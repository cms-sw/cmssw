#!/usr/bin/env python
#
# dump all fills into files.
# allfills.txt all the existing fills.
# fill_num.txt all the runs in the fill
# dumpFill -o outputdir
# dumpFill -f fillnum generate runlist for the given fill
#
import os,os.path,sys,math,array,datetime,time,re
import coral

from RecoLuminosity.LumiDB import argparse,lumiTime,CommonUtil,lumiQueryAPI,lumiCorrections
MINFILL=1800
allfillname='allfills.txt'

def listfilldir(indir):
    fillnamepat=r'^[0-9]{4}$'
    p=re.compile(fillnamepat)
    processedfills=[]
    dirList=os.listdir(indir)
    for fname in dirList:
        if p.match(fname) and os.path.isdir(os.path.join(indir,fname)):#found fill dir
            allfs=os.listdir(os.path.join(indir,fname))
            for myfile in allfs:
                sumfilenamepat=r'^[0-9]{4}_summary_CMS.txt$'
                s=re.compile(sumfilenamepat)
                if s.match(myfile):
                    #only if fill_summary_CMS.txt file exists
                    processedfills.append(int(fname))
    return processedfills

def lastcompleteFill(infile):
    lastfill=None
    hlinepat=r'(LASTCOMPLETEFILL )([0-9]{4})'
    h=re.compile(hlinepat)
    dqmfile=open(infile,'r')
    for line in dqmfile:
        result=h.match(line)
        if result:
            lastfill=result.group(2)
            break
    return int(lastfill)

def calculateSpecificLumi(lumi,lumierr,beam1intensity,beam1intensityerr,beam2intensity,beam2intensityerr):
    '''
    '''
    specificlumi=0.0
    specificlumierr=0.0
    if beam1intensity!=0.0 and  beam2intensity!=0.0:
        specificlumi=float(lumi)/(float(beam1intensity)*float(beam2intensity))
        specificlumierr=specificlumi*math.sqrt(lumierr**2/lumi**2+beam1intensityerr**2/beam1intensity**2+beam2intensityerr**2/beam2intensity**2)
    return (specificlumi,specificlumierr)

def getFillFromDB(dbsession,parameters,fillnum):
    '''
    output: {run:starttime}
    '''
    runtimesInFill={}
    q=dbsession.nominalSchema().newQuery()
    fillrundict=lumiQueryAPI.runsByfillrange(q,fillnum,fillnum)
    del q
    if len(fillrundict)>0:
        for fill,runs in  fillrundict.items():
            for run in runs:
                q=dbsession.nominalSchema().newQuery()
                rresult=lumiQueryAPI.runsummaryByrun(q,run)
                del q
                if len(rresult)==0: continue
                runtimesInFill[run]=rresult[3]
    return runtimesInFill

def getFillFromFile(fillnum,inputdir):
    runtimesInFill={}
    #look for files 'fill_num.txt' in inputdir
    for filename in os.listdir(inputdir):
        mpat=r'^fill_[0-9]{4}.txt$'
        m=re.compile(mpat)
        if m.match(filename) is None:
            continue
        filename=filename.strip()
        if filename.find('.')==-1: continue            
        basename,extension=filename.split('.')        
        if not extension or extension!='txt':
            continue
        if basename.find('_')==-1: continue
        prefix,number=basename.split('_')
        if not number : continue
        if fillnum!=int(number):continue
        f=open(os.path.join(inputdir,'fill_'+number+'.txt'),'r')
        for line in f:
            l=line.strip()
            fields=l.split(',')
            if len(fields)<2 : continue
            runtimesInFill[int(fields[0])]=fields[1]
        f.close()
    return runtimesInFill

def getSpecificLumi(dbsession,parameters,fillnum,inputdir,finecorrections=None):
    '''
    specific lumi in 1e-30 (ub-1s-1) unit
    lumidetail occlumi in 1e-27
    1309_lumi_401_CMS.txt
    time(in seconds since January 1,2011,00:00:00 UTC) stab(fraction of time spent in stable beams for this time bin) l(lumi in Hz/ub) dl(point-to-point error on lumi in Hz/ub) sl(specific lumi in Hz/ub) dsl(error on specific lumi)
    20800119.0 1 -0.889948 0.00475996848729 0.249009 0.005583287562 -0.68359 6.24140208607 0.0 0.0 0.0 0.0 0.0 0.0 0.0383576 0.00430892097862 0.0479095 0.00430892097862 66.6447 4.41269758764 0.0 0.0 0.0
    result [(time,beamstatusfrac,lumi,lumierror,speclumi,speclumierror)]
    '''
    #result=[]
    runtimesInFill=getFillFromFile(fillnum,inputdir)#{runnum:starttimestr}
    beamstatusDict={}#{runnum:{(startorbit,cmslsnum):beamstatus}}
    t=lumiTime.lumiTime()
    fillbypos={}#{bxidx:(lstime,beamstatusfrac,lumi,lumierror,specificlumi,specificlumierror)}
    #referencetime=time.mktime(datetime.datetime(2010,1,1,0,0,0).timetuple())
    referencetime=0
    if fillnum and len(runtimesInFill)==0:
        runtimesInFill=getFillFromDB(dbsession,parameters,fillnum)#{runnum:starttimestr}
    #precheck
    totalstablebeamLS=0
    for runnum in runtimesInFill.keys():
        q=dbsession.nominalSchema().newQuery()
        runinfo=lumiQueryAPI.lumisummaryByrun(q,runnum,'0001',beamstatus=None)
        del q
        lsbeamstatusdict={}
        for perlsdata in runinfo:
            cmslsnum=perlsdata[0]
            startorbit=perlsdata[3]
            beamstatus=perlsdata[4]
            lsbeamstatusdict[(startorbit,cmslsnum)]=beamstatus            
            #print (startorbit,cmslsnum),beamstatus
            if beamstatus=='STABLE BEAMS':
                totalstablebeamLS+=1
        beamstatusDict[runnum]=lsbeamstatusdict
    if totalstablebeamLS<10:#less than 10 LS in a fill has 'stable beam', it's no a good fill
        print 'fill ',fillnum,' , having less than 10 stable beam lS, is not good, skip'
        return fillbypos
    for runnum,starttime in runtimesInFill.items():
        #if not runtimesInFill.has_key(runnum):
        #    print 'run '+str(runnum)+' does not exist'
        #    continue
        q=dbsession.nominalSchema().newQuery()
        if finecorrections and finecorrections[runnum]:
            occlumidata=lumiQueryAPI.calibratedDetailForRunLimitresult(q,parameters,runnum,finecorrection=finecorrections[runnum])#{(startorbit,cmslsnum):[(bxidx,lumivalue,lumierr)]} #values after cut
        else:
            occlumidata=lumiQueryAPI.calibratedDetailForRunLimitresult(q,parameters,runnum)
        del q
        #print occlumidata
        q=dbsession.nominalSchema().newQuery()
        beamintensitydata=lumiQueryAPI.beamIntensityForRun(q,parameters,runnum)#{startorbit:[(bxidx,beam1intensity,beam2intensity),()]}
        #print 'beamintensity for run ',runnum,beamintensitydata
        del q
        for (startorbit,cmslsnum),lumilist in occlumidata.items():
            if len(lumilist)==0: continue
            beamstatusflag=beamstatusDict[runnum][(startorbit,cmslsnum)]
            beamstatusfrac=0.0
            if beamstatusflag=='STABLE BEAMS':
                beamstatusfrac=1.0
            lstimestamp=t.OrbitToUTCTimestamp(starttime,startorbit)
            for lumidata in lumilist:#loop over bx
                bxidx=lumidata[0]
                lumi=lumidata[1]
                lumierror=lumidata[2]
                speclumi=(0.0,0.0)
                if not fillbypos.has_key(bxidx):
                    fillbypos[bxidx]=[]
                if beamintensitydata.has_key(startorbit):
                    beaminfo=beamintensitydata[startorbit]
                    for beamintensitybx in beaminfo:
                        if beamintensitybx[0]==bxidx:                        
                            beam1intensity=beamintensitybx[1]
                            beam2intensity=beamintensitybx[2]
                            if beam1intensity<0:
                                beam1intensity=0
                            if beam2intensity<0:
                                beam2intensity=0
                            speclumi=calculateSpecificLumi(lumi,lumierror,beam1intensity,0.0,beam2intensity,0.0)
                            break
                fillbypos[bxidx].append((lstimestamp-referencetime,beamstatusfrac,lumi,lumierror,speclumi[0],speclumi[1]))

    #print 'fillbypos.keys ',fillbypos.keys()
    return fillbypos

#####output methods####
def filltofiles(allfills,runsperfill,runtimes,dirname):
    f=open(os.path.join(dirname,allfillname),'w')
    for fill in allfills:
        print >>f,'%d'%(fill)
    f.close()
    for fill,runs in runsperfill.items():
        filename='fill_'+str(fill)+'.txt'
        if len(runs)!=0:
            f=open(os.path.join(dirname,filename),'w')
            for run in runs:
                print >>f,'%d,%s'%(run,runtimes[run])
            f.close()
            
def specificlumiTofile(fillnum,filldata,outdir):
    timedict={}#{lstime:[[stablebeamfrac,lumi,lumierr,speclumi,speclumierr]]}
    #
    #check outdir/fillnum subdir exists; if not, create it; else outdir=outdir/fillnum
    #
    filloutdir=os.path.join(outdir,str(fillnum))
    print 'filloutdir ',filloutdir
    if not os.path.exists(filloutdir):
        os.mkdir(filloutdir)
    for cmsbxidx,perbxdata in filldata.items():
        lhcbucket=0
        if cmsbxidx!=0:
            lhcbucket=(cmsbxidx-1)*10+1
        a=sorted(perbxdata,key=lambda x:x[0])
        filename=str(fillnum)+'_lumi_'+str(lhcbucket)+'_CMS.txt'
        linedata=[]
        for perlsdata in a:
            ts=int(perlsdata[0])
            beamstatusfrac=perlsdata[1]
            lumi=perlsdata[2]
            lumierror=perlsdata[3]
            #beam1intensity=perlsdata[4]
            #beam2intensity=perlsdata[5]
            speclumi=perlsdata[4]
            speclumierror= perlsdata[5]
            if lumi>0:
                linedata.append([ts,beamstatusfrac,lumi,lumierror,speclumi,speclumierror])
            if not timedict.has_key(ts):
                timedict[ts]=[]
            timedict[ts].append([beamstatusfrac,lumi,lumierror,speclumi,speclumierror])
        if len(linedata)>10:#at least 10 good ls
            f=open(os.path.join(filloutdir,filename),'w')
            for line in linedata:
                print >>f, '%d\t%e\t%e\t%e\t%e\t%e'%(line[0],line[1],line[2],line[3],line[4],line[5])
            f.close()
    #print 'writing avg file'
    summaryfilename=str(fillnum)+'_lumi_CMS.txt'
    f=None
    lstimes=timedict.keys()
    lstimes.sort()
    fillseg=[]
    lscounter=0
    for lstime in lstimes:
        allvalues=timedict[lstime]
        transposedvalues=CommonUtil.transposed(allvalues,0.0)
        bstatfrac=transposedvalues[0][0]#beamstatus does not change with bx position
        lumivals=transposedvalues[1]
        lumitot=sum(lumivals)
        if bstatfrac==1.0 :
            fillseg.append([lstime,lumitot])
        lumierrs=transposedvalues[2]
        lumierrortot=math.sqrt(sum(map(lambda x:x**2,lumierrs)))
        specificvals=transposedvalues[3]
        specificavg=sum(specificvals)/float(len(specificvals))#avg spec lumi
        specificerrs=transposedvalues[4]
        specifictoterr=math.sqrt(sum(map(lambda x:x**2,specificerrs)))
        specificerravg=specifictoterr/float(len(specificvals))
        if lscounter==0:
            f=open(os.path.join(filloutdir,summaryfilename),'w')
        lscounter+=1
        print >>f,'%d\t%e\t%e\t%e\t%e\t%e'%(lstime,bstatfrac,lumitot,lumierrortot,specificavg,specificerravg)
    if f is not None:
        f.close()
    #print 'writing summary file'
    fillsummaryfilename=str(fillnum)+'_summary_CMS.txt'
    f=open(os.path.join(filloutdir,fillsummaryfilename),'w')    
    if len(fillseg)==0:
        print >>f,'%s'%('#no stable beams')
        f.close()
        return
    previoustime=fillseg[0][0]
    boundarytime=fillseg[0][0]
    #print 'boundary time ',boundarytime
    summaryls={}
    summaryls[boundarytime]=[]
    for [lstime,lumitot] in fillseg:#fillseg is everything with stable beam flag
        if lstime-previoustime>50.0:
            boundarytime=lstime
            #print 'found new boundary ',boundarytime
            summaryls[boundarytime]=[]
     #   print 'appending ',boundarytime,lstime,lumitot
        summaryls[boundarytime].append([lstime,lumitot])
        previoustime=lstime
    #print summaryls
   
    summarylstimes=summaryls.keys()
    summarylstimes.sort()
    for bts in summarylstimes:
        startts=bts
        tsdatainseg=summaryls[bts]
        #print 'tsdatainseg ',tsdatainseg
        stopts=tsdatainseg[-1][0]
        plu=max(CommonUtil.transposed(tsdatainseg,0.0)[1])
        lui=sum(CommonUtil.transposed(tsdatainseg,0.0)[1])*23.357
        print >>f,'%d\t%d\t%e\t%e'%(startts,stopts,plu,lui)
    f.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Dump Fill",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-i',dest='inputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('-o',dest='outputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('-f',dest='fillnum',action='store',required=False,help='specific fill',default=None)
    parser.add_argument('-norm',dest='norm',action='store',required=False,help='norm',default=None)
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    parser.add_argument('--with-correction',dest='withFineCorrection',action='store_true',required=False,help='with fine correction',default=None)
    parser.add_argument('--toscreen',dest='toscreen',action='store_true',help='dump to screen')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    parameters = lumiQueryAPI.ParametersObject()
    if options.norm!=None:
        parameters.normFactor=float(options.norm)
    session,svc =  lumiQueryAPI.setupSession (options.connect or \
                                              'frontier://LumiCalc/CMS_LUMI_PROD',
                                               options.siteconfpath,parameters,options.debug)

    ##
    #query DB for all fills and compare with allfills.txt
    #if found newer fills, store  in mem fill number
    #reprocess anyway the last 1 fill in the dir
    #redo specific lumi for all marked fills
    ##
    finecorrections=None
    allfillsFromFile=[]
    fillstoprocess=[]
    session.transaction().start(True)
    if options.fillnum: #if process a specific single fill
        fillstoprocess.append(int(options.fillnum))
    else: #if process fills automatically
        q=session.nominalSchema().newQuery()    
        allfillsFromDB=lumiQueryAPI.allfills(q)
        del q
        processedfills=listfilldir(options.outputdir)
        lastcompletedFill=lastcompleteFill(os.path.join(options.inputdir,'runtofill_dqm.txt'))
        print 'last complete fill : ',lastcompletedFill
        print 'processedfills in '+options.outputdir+' ',processedfills
        for pf in processedfills:
            if pf>lastcompletedFill:
                print '\tremove unfinished fill from processed list ',pf
                processedfills.remove(pf)
        print 'final processed fills : ',sorted(processedfills)
        for fill in allfillsFromDB:
            if fill not in processedfills :
                if fill<=lastcompletedFill:
                    #print 'fill less than last complet fill ',fill
                    if fill>MINFILL:
                        fillstoprocess.append(fill)
                else:
                    print 'ongoing fill...',fill               
    print 'fills to process : ',fillstoprocess
    if len(fillstoprocess)==0:
        print 'no fill to process, exit '
        exit(0)
    runsperfillFromDB={}
    q=session.nominalSchema().newQuery()
    runsperfillFromDB=lumiQueryAPI.runsByfillrange(q,int(min(fillstoprocess)),int(max(fillstoprocess)))
    del q
    #print 'runsperfillFromDB ',runsperfillFromDB
    runtimes={}
    runs=runsperfillFromDB.values()#list of lists
    allruns=[item for sublist in runs for item in sublist]
    allruns.sort()
    #print 'allruns ',allruns
    for run in allruns:
        q=session.nominalSchema().newQuery()
        runtimes[run]=lumiQueryAPI.runsummaryByrun(q,run)[3]
        del q
    if options.withFineCorrection:
         schema=session.nominalSchema()
         finecorrections=lumiCorrections.correctionsForRange(schema,allruns)
    #write specificlumi to outputdir
    #update inputdir
    print 'fillstoprocess ',fillstoprocess
    if len(fillstoprocess)!=0 and options.fillnum is None:
        filltofiles(allfillsFromDB,runsperfillFromDB,runtimes,options.inputdir)
    print '===== Start Processing Fills',fillstoprocess
    print '====='
    
    for fillnum in fillstoprocess:
        filldata=getSpecificLumi(session,parameters,fillnum,options.inputdir,finecorrections=finecorrections)
        specificlumiTofile(fillnum,filldata,options.outputdir)
    session.transaction().commit()

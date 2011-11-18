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

from RecoLuminosity.LumiDB import argparse,lumiTime,CommonUtil,lumiQueryAPI

allfillname='allfills.txt'
        
def calculateSpecificLumi(lumi,lumierr,beam1intensity,beam1intensityerr,beam2intensity,beam2intensityerr):
    '''
    '''
    specificlumi=0.0
    specificlumierr=0.0
    if lumi!=0.0 and beam1intensity!=0.0 and  beam2intensity!=0.0:
        specificlumi=float(lumi)/(float(beam1intensity)*float(beam2intensity))
        specificlumierr=specificlumi*math.sqrt(lumierr**2/lumi**2+beam1intensityerr**2/beam1intensity**2+beam2intensityerr**2/beam2intensity**2)
    return (specificlumi,specificlumierr)

def getFillFromDB(dbsession,parameters,fillnum):
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
    
def getSpecificLumi(dbsession,parameters,fillnum,inputdir):
    '''
    specific lumi in 1e-30 (ub-1s-1) unit
    lumidetail occlumi in 1e-27
    1309_lumireg_401_CMS.txt
    ip fillnum time l(lumi in Hz/ub) dl(point-to-point error on lumi in Hz/ub) sl(specific lumi in Hz/ub) dsl(error on specific lumi)
    5  1309 20800119.0 -0.889948 0.00475996848729 0.249009 0.005583287562 -0.68359 6.24140208607 0.0 0.0 0.0 0.0 0.0 0.0 0.0383576 0.00430892097862 0.0479095 0.00430892097862 66.6447 4.41269758764 0.0 0.0 0.0
    result [(time,lumi,lumierror,speclumi,speclumierror)]
    '''
    #result=[]
    runtimesInFill=getFillFromFile(fillnum,inputdir)#{runnum:starttimestr}
    t=lumiTime.lumiTime()
    fillbypos={}#{bxidx:(lstime,lumi,lumierror,specificlumi,specificlumierror)}
    #'referencetime=time.mktime(datetime.datetime(2009,12,31,23,0,0).timetuple())
    #referencetime=time.mktime(datetime.datetime(2010,1,1,0,0,0).timetuple())
    referencetime=1262300400-7232
    #for i in range(3564):
    #    fillbypos[i]=[]

    if fillnum and len(runtimesInFill)==0:
        runtimesInFill=getFillFromDB(dbsession,parameters,fillnum)#{runnum:starttimestr}
    #precheck
    totalstablebeamLS=0
    for runnum in runtimesInFill.keys():
        q=dbsession.nominalSchema().newQuery()
        runinfo=lumiQueryAPI.lumisummaryByrun(q,runnum,'0001',beamstatus='STABLE BEAMS')
        del q
        totalstablebeamLS+=len(runinfo)
    if totalstablebeamLS<10:#less than 10 LS in a fill has 'stable beam', it's no a good fill
        print 'fill ',fillnum,' , having less than 10 stable beam lS, is not good, skip'
        return fillbypos
    
    for runnum,starttime in runtimesInFill.items():
        if not runtimesInFill.has_key(runnum):
            print 'run '+str(runnum)+' does not exist'
            continue
        q=dbsession.nominalSchema().newQuery()
        occlumidata=lumiQueryAPI.calibratedDetailForRunLimitresult(q,parameters,runnum)#{(startorbit,cmslsnum):[(bxidx,lumivalue,lumierr)]} #values after cut
        del q
        #print occlumidata
        q=dbsession.nominalSchema().newQuery()
        beamintensitydata=lumiQueryAPI.beamIntensityForRun(q,parameters,runnum)#{startorbit:[(bxidx,beam1intensity,beam2intensity)]}
        del q
        for (startorbit,cmslsnum),lumilist in occlumidata.items():
            if len(lumilist)==0: continue
            lstimestamp=t.OrbitToTimestamp(starttime,startorbit)
            if beamintensitydata.has_key(startorbit) and len(beamintensitydata[startorbit])>0:
                for lumidata in lumilist:
                    bxidx=lumidata[0]
                    lumi=lumidata[1]
                    lumierror=lumidata[2]
                    for beamintensitybx in beamintensitydata[startorbit]:
                        if beamintensitybx[0]==bxidx:
                            if not fillbypos.has_key(bxidx):
                                fillbypos[bxidx]=[]
                            beam1intensity=beamintensitybx[1]
                            beam2intensity=beamintensitybx[2]
                            speclumi=calculateSpecificLumi(lumi,lumierror,beam1intensity,0.0,beam2intensity,0.0)
                            fillbypos[bxidx].append([lstimestamp-referencetime,lumi,lumierror,beam1intensity,beam2intensity,speclumi[0],speclumi[1]])
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
    timedict={}#{lstime:[[lumi,lumierr,speclumi,speclumierr]]}
    ipnumber=5
    for cmsbxidx,perbxdata in filldata.items():
        lhcbucket=0
        if cmsbxidx!=0:
            lhcbucket=(cmsbxidx-1)*10+1
        a=sorted(perbxdata,key=lambda x:x[0])
        lscounter=0
        filename=str(fillnum)+'_lumi_'+str(lhcbucket)+'_CMS.txt'
        for perlsdata in a:
            if perlsdata[-2]>0 and perlsdata[-1]>0 and perlsdata[1]>0:
                if lscounter==0:
                    f=open(os.path.join(outdir,filename),'w')
                print >>f, '%d\t%d\t%d\t%e\t%e\t%e\t%e\n'%(int(ipnumber),int(fillnum),int(perlsdata[0]),perlsdata[1],perlsdata[2],perlsdata[-2],perlsdata[-1])
                if not timedict.has_key(int(perlsdata[0])):
                    timedict[int(perlsdata[0])]=[]
                timedict[int(perlsdata[0])].append([perlsdata[1],perlsdata[2],perlsdata[-2],perlsdata[-1]])
                lscounter+=1
        f.close()
        summaryfilename=str(fillnum)+'_lumi_CMS.txt'
        f=open(os.path.join(outdir,summaryfilename),'w')
        lstimes=timedict.keys()
        lstimes.sort()
        for lstime in lstimes:
            allvalues=timedict[lstime]
            transposedvalues=CommonUtil.transposed(allvalues,0.0)
            lumivals=transposedvalues[0]
            lumitot=sum(lumivals)
            lumierrs=transposedvalues[1]
            lumierrortot=math.sqrt(sum(map(lambda x:x**2,lumierrs)))
            specificvals=transposedvalues[2]
            specificavg=sum(specificvals)/float(len(specificvals))#avg spec lumi
            specificerrs=transposedvalues[3]
            specifictoterr=math.sqrt(sum(map(lambda x:x**2,specificerrs)))
            specificerravg=specifictoterr/float(len(specificvals))
            print >>f,'%d\t%d\t%d\t%e\t%e\t%e\t%e\n'%(int(ipnumber),int(fillnum),lstime,lumitot,lumierrortot,specificavg,specificerravg)
        f.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Dump Fill",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-i',dest='inputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('-o',dest='outputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('-f',dest='fillnum',action='store',required=False,help='specific fill',default=None)
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    parser.add_argument('--toscreen',dest='toscreen',action='store_true',help='dump to screen')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    parameters = lumiQueryAPI.ParametersObject()
    session,svc =  lumiQueryAPI.setupSession (options.connect or \
                                              'frontier://LumiCalc/CMS_LUMI_PROD',
                                               options.siteconfpath,parameters,options.debug)

    ##
    #query DB for all fills and compare with allfills.txt
    #if found newer fills, store  in mem fill number
    #reprocess anyway the last 5 fills in the dir
    #redo specific lumi for all marked fills
    ##
 
    allfillsFromFile=[]
    fillstoprocess=[]
    session.transaction().start(True)
    if options.fillnum: #if process a specific single fill
        fillstoprocess.append(int(options.fillnum))
    else: #if process fills automatically
        q=session.nominalSchema().newQuery()    
        allfillsFromDB=lumiQueryAPI.allfills(q)
        del q
        if os.path.exists(os.path.join(options.inputdir,allfillname)):
            allfillF=open(os.path.join(options.inputdir,allfillname),'r')
            for line in allfillF:
                l=line.strip()
                allfillsFromFile.append(int(l))
            allfillF.close()
            if len(allfillsFromDB)==0:
                print 'no fill found in DB, exit'
                sys.exit(-1)
            if len(allfillsFromDB)!=0:
                allfillsFromDB.sort()
            if len(allfillsFromFile) != 0:
                allfillsFromFile.sort()
            #print 'allfillsFromFile ',allfillsFromFile
            if max(allfillsFromDB)>max(allfillsFromFile) : #need not to be one to one match because data can be deleted in DB
                print 'found new fill '
                for fill in allfillsFromDB:
                    if fill>max(allfillsFromFile):
                        fillstoprocess.append(fill)
            else:
                print 'no new fill '
                fillstoprocess+=allfillsFromFile[-1:]
            #if len(allfillsFromFile)>5: #reprocess anyway old fills
            #    fillstoprocess+=allfillsFromFile[-5:]
        else:
            fillstoprocess=allfillsFromDB #process everything from scratch
    #print 'fillstoprocess ',fillstoprocess
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
    #write specificlumi to outputdir
    #update inputdir
    if len(fillstoprocess)!=0 and options.fillnum is None:
        filltofiles(allfillsFromDB,runsperfillFromDB,runtimes,options.inputdir)
    print '===== Start Processing Fills',fillstoprocess
    print '====='
    for fillnum in fillstoprocess:
        filldata=getSpecificLumi(session,parameters,fillnum,options.inputdir)
        specificlumiTofile(fillnum,filldata,options.outputdir)
    session.transaction().commit()

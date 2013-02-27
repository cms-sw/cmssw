#!/usr/bin/env python
########################################################################
# Command to produce perbunch and specific lumi                        #
#                                                                      #
# Author:      Zhen Xie                                                #
########################################################################
#
# dump all fills into files.
# allfills.txt all the existing fills.
# fill_num.txt all the runs in the fill
# dumpFill -o outputdir
# dumpFill -f fillnum generate runlist for the given fill
#
import os,os.path,sys,math,array,datetime,time,calendar,re
import coral

from RecoLuminosity.LumiDB import argparse,sessionManager,lumiTime,CommonUtil,lumiCalcAPI,lumiParameters,revisionDML,normDML
MINFILL=1800
MAXFILL=9999
allfillname='allfills.txt'

def getFillFromDB(schema,fillnum):
    '''
    output: {run:starttime}
    '''
    runtimesInFill={}
    fillrundict=lumiCalcAPI.fillrunMap(schema,fillnum)
    if len(fillrundict)>0:
        runs=fillrundict.values()[0]
        runlsdict=dict(zip(runs,[None]*len(runs)))
        runresult=lumiCalcAPI.runsummary(schema,runlsdict)    
        for perrundata in runresult:
            runtimesInFill[perrundata[0]]=perrundata[7]
    return runtimesInFill

def listfilldir(indir):
    '''
    list all fills contained in the given dir
    input: indir
    output: [fill]
    '''
    fillnamepat=r'^[0-9]{4}$'
    p=re.compile(fillnamepat)
    processedfills=[]
    dirList=os.listdir(indir)
    for fname in dirList:
        if p.match(fname) and os.path.isdir(os.path.join(indir,fname)):#found fill dir
            allfs=os.listdir(os.path.join(indir,fname))
            for myfile in allfs:
                sumfilenamepat=r'^[0-9]{4}_bxsum_CMS.txt$'
                s=re.compile(sumfilenamepat)
                if s.match(myfile):
                    #only if fill_summary_CMS.txt file exists
                    processedfills.append(int(fname))
    return processedfills

def lastcompleteFill(infile):
    '''
    parse infile to find LASTCOMPLETEFILL
    input: input file name
    output: last completed fill number
    '''
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
    calculate specific lumi
    input: instlumi, instlumierror,beam1intensity,beam1intensityerror,beam2intensity,beam2intensityerror
    output (specific lumi value,specific lumi error)
    '''
    specificlumi=0.0
    specificlumierr=0.0
    if beam1intensity<0: beam1intensity=0
    if beam2intensity<0: beam2intensity=0
    if beam1intensity>0.0 and  beam2intensity>0.0:
        specificlumi=float(lumi)/(float(beam1intensity)*float(beam2intensity))
        specificlumierr=specificlumi*math.sqrt(lumierr**2/lumi**2+beam1intensityerr**2/beam1intensity**2+beam2intensityerr**2/beam2intensity**2)
    return (specificlumi,specificlumierr)

def getFillFromFile(fillnum,inputdir):
    '''
    parse fill_xxx.txt files in the input directory for runs, starttime in the fill
    input: fillnumber, input dir
    output: {run:tarttime}
    '''
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

#####output methods####
def filltofiles(allfills,runsperfill,runtimes,dirname):
    '''
    write runnumber:starttime map per fill to files
    '''
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
    #
    #input : fillnum
    #        filldata: {bxidx:[[lstime,beamstatusfrac,lumivalue,lumierror,speclumi,speclumierr]],[]}
    #sorted by bxidx, sorted by lstime inside list
    #check outdir/fillnum subdir exists; if not, create it; else outdir=outdir/fillnum
    #
    if not filldata:
        print 'empty input data, do nothing for fill ',fillnum
        return
    timedict={}#{lstime:[[stablebeamfrac,lumi,lumierr,speclumi,speclumierr]]}
    filloutdir=os.path.join(outdir,str(fillnum))
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
    fillsummaryfilename=str(fillnum)+'_bxsum_CMS.txt'
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
    lumip=lumiParameters.ParametersObject()
    for bts in summarylstimes:
        startts=bts
        tsdatainseg=summaryls[bts]
        #print 'tsdatainseg ',tsdatainseg
        stopts=tsdatainseg[-1][0]
        plu=max(CommonUtil.transposed(tsdatainseg,0.0)[1])
        lui=sum(CommonUtil.transposed(tsdatainseg,0.0)[1])*lumip.lslengthsec()
        print >>f,'%d\t%d\t%e\t%e'%(startts,stopts,plu,lui)
    f.close()

def getSpecificLumi(schema,fillnum,inputdir,dataidmap,normmap,xingMinLum=0.0,amodetag='PROTPHYS',bxAlgo='OCC1'):
    '''
    specific lumi in 1e-30 (ub-1s-1) unit
    lumidetail occlumi in 1e-27
    1309_lumi_401_CMS.txt
    time(in seconds since January 1,2011,00:00:00 UTC) stab(fraction of time spent in stable beams for this time bin) l(lumi in Hz/ub) dl(point-to-point error on lumi in Hz/ub) sl(specific lumi in Hz/ub) dsl(error on specific lumi)
    20800119.0 1 -0.889948 0.00475996848729 0.249009 0.005583287562 -0.68359 6.24140208607 0.0 0.0 0.0 0.0 0.0 0.0 0.0383576 0.00430892097862 0.0479095 0.00430892097862 66.6447 4.41269758764 0.0 0.0 0.0
    result [(time,beamstatusfrac,lumi,lumierror,speclumi,speclumierror)]
    '''
    t=lumiTime.lumiTime()
    fillbypos={}#{bxidx:[[ts,beamstatusfrac,lumi,lumierror,spec1,specerror],[]]}
    runtimesInFill=getFillFromDB(schema,fillnum)#{runnum:starttimestr}
    runlist=runtimesInFill.keys()
    if not runlist: return fillbypos
    irunlsdict=dict(zip(runlist,[None]*len(runlist)))
    #print irunlsdict
    GrunsummaryData=lumiCalcAPI.runsummaryMap(session.nominalSchema(),irunlsdict)
    lumidetails=lumiCalcAPI.deliveredLumiForIds(schema,irunlsdict,dataidmap,GrunsummaryData,beamstatusfilter=None,normmap=normmap,withBXInfo=True,bxAlgo=bxAlgo,xingMinLum=xingMinLum,withBeamIntensity=True,lumitype='HF')
    
    #
    #output: {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),calibratedlumierr(6),(bxvalues,bxerrs)(7),(bxidx,b1intensities,b2intensities)(8),fillnum(9)]}
    #
    totalstablebeamls=0
    orderedrunlist=sorted(lumidetails)
    for run in orderedrunlist:
        perrundata=lumidetails[run]
        for perlsdata in perrundata:
            beamstatus=perlsdata[3]
            if beamstatus=='STABLE BEAMS':
                totalstablebeamls+=1
    #print 'totalstablebeamls in fill ',totalstablebeamls
    if totalstablebeamls<10:#less than 10 LS in a fill has 'stable beam', it's no a good fill
        print 'fill ',fillnum,' , having less than 10 stable beam lS, is not good, skip'
        return fillbypos
    lumiparam=lumiParameters.ParametersObject()
    for run in orderedrunlist:
        perrundata=lumidetails[run]
        for perlsdata in perrundata:
            beamstatusfrac=0.0
            tsdatetime=perlsdata[2]
            ts=calendar.timegm(tsdatetime.utctimetuple())
            beamstatus=perlsdata[3]
            if beamstatus=='STABLE BEAMS':
                beamstatusfrac=1.0
            (bxidxlist,bxvaluelist,bxerrolist)=perlsdata[7]
            #instbxvaluelist=[x/lumiparam.lslengthsec() for x in bxvaluelist if x]
            instbxvaluelist=[x for x in bxvaluelist if x]
            maxlumi=0.0
            if len(instbxvaluelist)!=0:
                maxlumi=max(instbxvaluelist)
            avginstlumi=0.0
            if len(instbxvaluelist)!=0:
                avginstlumi=sum(instbxvaluelist)
            (intbxidxlist,b1intensities,b2intensities)=perlsdata[8]#contains only non-zero bx
            for bxidx in bxidxlist:
                idx=bxidxlist.index(bxidx)
                instbxvalue=instbxvaluelist[idx]
                bxerror=bxerrolist[idx]
                if instbxvalue<max(xingMinLum,maxlumi*0.2):
                    continue
                bintensityPos=-1
                try:
                    bintensityPos=intbxidxlist.index(bxidx)
                except ValueError:
                    pass
                if bintensityPos<=0:
                    fillbypos.setdefault(bxidx,[]).append([ts,beamstatusfrac,instbxvalue,bxerror,0.0,0.0])
                    continue
                b1intensity=b1intensities[bintensityPos]
                b2intensity=b2intensities[bintensityPos]
                speclumi=calculateSpecificLumi(instbxvalue,bxerror,b1intensity,0.0,b2intensity,0.0)
                fillbypos.setdefault(bxidx,[]).append([ts,beamstatusfrac,instbxvalue,bxerror,speclumi[0],speclumi[1]])
    return fillbypos


##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "specific lumi",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    amodetagChoices = [ "PROTPHYS","IONPHYS",'PAPHYS' ]
    xingAlgoChoices =[ "OCC1","OCC2","ET"]
    # parse arguments
    parser.add_argument('-c',dest='connect',
                        action='store',
                        required=False,
                        help='connect string to lumiDB,optional',
                        default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',
                        action='store',
                        help='path to authentication file,optional')
    parser.add_argument('-i',dest='inputdir',
                        action='store',
                        required=False,
                        help='output dir',
                        default='.')
    parser.add_argument('-o',dest='outputdir',
                        action='store',
                        required=False,
                        help='output dir',
                        default='.')
    parser.add_argument('-f','--fill',dest='fillnum',
                        action='store',
                        required=False,
                        help='specific fill',
                        default=None)
    parser.add_argument('--minfill',dest='minfill',
                        type=int,
                        action='store',
                        required=False,
                        default=MINFILL,
                        help='min fill')
    parser.add_argument('--maxfill',dest='maxfill',
                        type=int,
                        action='store',
                        required=False,
                        default=MAXFILL,
                        help='maximum fillnumber '
                        )
    parser.add_argument('--amodetag',dest='amodetag',
                        action='store',
                        choices=amodetagChoices,
                        required=False,
                        help='specific accelerator mode choices [PROTOPHYS,IONPHYS,PAPHYS] (optional)')
    parser.add_argument('--xingMinLum', dest = 'xingMinLum',
                        type=float,
                        default=1e-03,
                        required=False,
                        help='Minimum luminosity considered for lumibylsXing action')
    parser.add_argument('--xingAlgo', dest = 'bxAlgo',
                        default='OCC1',
                        required=False,
                        help='algorithm name for per-bunch lumi ')
    parser.add_argument('--normtag',dest='normtag',action='store',
                        required=False,
                        help='norm tag',
                        default=None)
    parser.add_argument('--datatag',dest='datatag',action='store',
                        required=False,
                        help='data tag',
                        default=None)
    #
    #command configuration 
    #
    parser.add_argument('--siteconfpath',dest='siteconfpath',action='store',
                        help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    #
    #switches
    #
    parser.add_argument('--without-correction',dest='withoutNorm',action='store_true',
                        help='without any correction/calibration' )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    ##
    #query DB for all fills and compare with allfills.txt
    #if found newer fills, store  in mem fill number
    #reprocess anyway the last 1 fill in the dir
    #redo specific lumi for all marked fills
    ##    
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])

    fillstoprocess=[]
    maxfillnum=options.maxfill
    minfillnum=options.minfill
    if options.fillnum is not None: #if process a specific single fill
        fillstoprocess.append(int(options.fillnum))
    else:
        session.transaction().start(True)
        schema=session.nominalSchema()
        allfillsFromDB=lumiCalcAPI.fillInRange(schema,fillmin=minfillnum,fillmax=maxfillnum,amodetag=options.amodetag)
        processedfills=listfilldir(options.outputdir)
        lastcompletedFill=lastcompleteFill(os.path.join(options.inputdir,'runtofill_dqm.txt'))
        for pf in processedfills:
            if pf>lastcompletedFill:
                print '\tremove unfinished fill from processed list ',pf
                processedfills.remove(pf)
        for fill in allfillsFromDB:
            if fill not in processedfills :
                if int(fill)<=lastcompletedFill:
                    if int(fill)>minfillnum and int(fill)<maxfillnum:
                        fillstoprocess.append(fill)
                else:
                    print 'ongoing fill...',fill
        session.transaction().commit()
    print 'fills to process : ',fillstoprocess
    if len(fillstoprocess)==0:
        print 'no fill to process, exit '
        exit(0)

    
    print '===== Start Processing Fills',fillstoprocess
    print '====='
    filldata={}
    #
    # check datatag
    #
    reqfillmin=min(fillstoprocess)
    reqfillmax=max(fillstoprocess)
    session.transaction().start(True)
    runlist=lumiCalcAPI.runList(session.nominalSchema(),options.fillnum,runmin=None,runmax=None,fillmin=reqfillmin,fillmax=reqfillmax,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=options.amodetag,nominalEnergy=None,energyFlut=None,requiretrg=False,requirehlt=False)
    
    datatagname=options.datatag
    if not datatagname:
        (datatagid,datatagname)=revisionDML.currentDataTag(session.nominalSchema())
        dataidmap=revisionDML.dataIdsByTagId(session.nominalSchema(),datatagid,runlist=runlist,withcomment=False)
        #{run:(lumidataid,trgdataid,hltdataid,())}
    else:
        dataidmap=revisionDML.dataIdsByTagName(session.nominalSchema(),datatagname,runlist=runlist,withcomment=False)

    #
    # check normtag and get norm values if required
    #
    normname='NONE'
    normid=0
    normvalueDict={}
    if not options.withoutNorm:
        normname=options.normtag
        if not normname:
            normmap=normDML.normIdByType(session.nominalSchema(),lumitype='HF',defaultonly=True)
            if len(normmap):
                normname=normmap.keys()[0]
                normid=normmap[normname]
        else:
            normid=normDML.normIdByName(session.nominalSchema(),normname)
        if not normid:
            raise RuntimeError('[ERROR] cannot resolve norm/correction')
            sys.exit(-1)
        normvalueDict=normDML.normValueById(session.nominalSchema(),normid) #{since:[corrector(0),{paramname:paramvalue}(1),amodetag(2),egev(3),comment(4)]}
    session.transaction().commit()
    for fillnum in fillstoprocess:# process per fill
        session.transaction().start(True)
        filldata=getSpecificLumi(session.nominalSchema(),fillnum,options.inputdir,dataidmap,normvalueDict,xingMinLum=options.xingMinLum,amodetag=options.amodetag,bxAlgo=options.bxAlgo)
        specificlumiTofile(fillnum,filldata,options.outputdir)
        session.transaction().commit()



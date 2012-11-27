import os,coral,datetime,fnmatch,time
from RecoLuminosity.LumiDB import nameDealer,revisionDML,dataDML,lumiTime,CommonUtil,selectionParser,hltTrgSeedMapper,normFunctors,lumiParameters

########################################################################
# Lumi data management and calculation API                             #
#                                                                      #
# Author:      Zhen Xie                                                #
########################################################################

def runsummary(schema,irunlsdict):
    '''
    output  [[run(0),l1key(1),amodetag(2),egev(3),hltkey(4),fillnum(5),fillscheme(6),starttime(7),stoptime(8)]]
    '''
    result=[]
    for run in sorted(irunlsdict):
        runinfo=dataDML.runsummary(schema,run)
        runinfo.insert(0,run)
        result.append(runinfo)
    return result

def runsummaryMap(schema,irunlsdict):
    '''
    output  {run:[l1key(0),amodetag(1),egev(2),hltkey(3),fillnum(4),fillscheme(5),starttime(6),stoptime(7)]}
    '''
    result={}
    seqresult=runsummary(schema,irunlsdict)
    for [run,l1key,amodetag,egev,hltkey,fillnum,fillscheme,starttime,stoptime] in seqresult:
        result[run]=[l1key,amodetag,egev,hltkey,fillnum,fillscheme,starttime,stoptime]
    return result

def fillInRange(schema,fillmin=1000,fillmax=9999,amodetag='PROTPHYS',startT=None,stopT=None):
    '''
    output [fill]
    '''
    fills=dataDML.fillInRange(schema,fillmin,fillmax,amodetag,startT,stopT)
    return fills
def fillrunMap(schema,fillnum=None,runmin=None,runmax=None,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=None):
    '''
    output: {fill:[runnum,...]}
    '''
    return dataDML.fillrunMap(schema,fillnum=fillnum,runmin=runmin,runmax=runmax,startT=startT,stopT=stopT,l1keyPattern=l1keyPattern,hltkeyPattern=hltkeyPattern,amodetag=amodetag)
             
def runList(schema,fillnum=None,runmin=None,runmax=None,fillmin=None,fillmax=None,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=None,nominalEnergy=None,energyFlut=0.2,requiretrg=True,requirehlt=True,lumitype='HF'):
    '''
    output: [runnumber,...]
    '''
    return dataDML.runList(schema,fillnum=fillnum,runmin=runmin,runmax=runmax,fillmin=None,fillmax=None,startT=startT,stopT=stopT,l1keyPattern=l1keyPattern,hltkeyPattern=hltkeyPattern,amodetag=amodetag,nominalEnergy=nominalEnergy,energyFlut=energyFlut,requiretrg=requiretrg,requirehlt=requirehlt,lumitype=lumitype)

def hltpathsForRange(schema,runlist,hltpathname=None,hltpathpattern=None):
    '''
    input: runlist [run],     (required)      
           datatag: data version (optional)
    output : {runnumber,[(hltpath,l1seedexpr,l1bitname)...]}
    '''
    result={}
    for run in runlist:
        hlttrgmap=dataDML.hlttrgMappingByrun(schema,run,hltpathname=hltpathname,hltpathpattern=hltpathpattern)
        result[run]=[]
        for hltpath in sorted(hlttrgmap):
            l1seedexpr=hlttrgmap[hltpath]
            (exptype,l1bits)=hltTrgSeedMapper.findUniqueSeed(hltpath,l1seedexpr)
            l1bitname='n/a'
            if l1bits:
                if exptype:
                    l1bitname=l1seedexpr
                else:
                    l1bitname=l1bits[0]
            result[run].append((hltpath,l1seedexpr,l1bitname))
    return result

def beamForRange(schema,inputRange,withBeamIntensity=False,minIntensity=0.1,tableName=None,branchName=None):
    '''
    input:
           inputRange: {run:[cmsls]} (required)
    output : {runnumber:[(lumicmslnum,cmslsnum,beamenergy,beamstatus,[(ibx,b1,b2)])...](4)}
    '''
    if tableName is None:
        tableName=nameDealer.lumidataTableName()
    if branchName is None:
        branchName='DATA'
    result={}
    for run in inputRange.keys():
        lslist=inputRange[run]
        if lslist is not None and len(lslist)==0:
            result[run]=[]#if no LS is selected for a run
            continue
        lumidataid=dataDML.guessLumiDataIdByRun(schema,run,tableName)
        if lumidataid is None:
            result[run]=None
            continue #run non exist
        lumidata=dataDML.beamInfoById(schema,lumidataid,withBeamIntensity=withBeamIntensity,minIntensity=minIntensity)
        #(runnum,[(lumilsnum(0),cmslsnum(1),beamstatus(2),beamenergy(3),beaminfolist(4)),..])
        result[run]=[]
        perrundata=lumidata[1]
        if not perrundata:
            result[run]=[]
            continue
        for perlsdata in perrundata:
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            if lslist is not None and cmslsnum not in lslist:
                continue
            beamstatus=perlsdata[2]
            beamenergy=perlsdata[3]
            beamintInfolist=[]
            if withBeamIntensity:
                beamintInfolist=perlsdata[4]
            result[run].append((lumilsnum,cmslsnum,beamstatus,beamenergy,beamintInfolist))        
    return result

def beamForIds(schema,irunlsdict,dataidmap,withBeamIntensity=False,minIntensity=0.1):
    '''
    input:
           inputRange: {run:[cmsls]} (required)
           dataidmap: {run:(lumiid,trgid,hltid)}
    output : {runnumber:[(lumicmslnum(0),cmslsnum(1),beamenergy(2),beamstatus(3),ncollidingbx(4),[(ibx,b1,b2)])...](5)}
    '''
    result={}
    for run in irunlsdict.keys():
        result[run]=[]
        lslist=irunlsdict[run]
        if lslist is not None and len(lslist)==0:
            continue
        if not dataidmap.has_key(run):
            continue #run non exist
        lumidataid=dataidmap[run][0]
        if lumidataid is None:
            result[run]=None
            continue
        lumidata=dataDML.beamInfoById(schema,lumidataid,withBeamIntensity=withBeamIntensity,minIntensity=minIntensity)
        #(runnum,[(lumilsnum(0),cmslsnum(1),beamstatus(2),beamenergy(3),ncollidingbunches(4),beaminfolist(5),..])
        if lumidata and lumidata[1]:
            perrundata=lumidata[1]
            for perlsdata in perrundata:
                lumilsnum=perlsdata[0]
                cmslsnum=perlsdata[1]
                if lslist is not None and cmslsnum not in lslist:
                    continue
                beamstatus=perlsdata[2]
                beamenergy=perlsdata[3]
                ncollidingbunches=perlsdata[4]
                beamintInfolist=[]
                if withBeamIntensity:
                    beamintInfolist=perlsdata[5]
                result[run].append((lumilsnum,cmslsnum,beamstatus,beamenergy,ncollidingbunches,beamintInfolist))        
    return result

def hltForIds(schema,irunlsdict,dataidmap,hltpathname=None,hltpathpattern=None,withL1Pass=False,withHLTAccept=False):
    '''
    input:
           irunlsdict: {run:[cmsls]} (required)
           dataidmap: {run:(lumiid,trgid,hltid)}
           hltpathname: exact match hltpathname  (optional) 
           hltpathpattern: regex match hltpathpattern (optional)
           withL1Pass: with L1 pass count
           withHLTAccept: with HLT accept
    output: {runnumber:[(cmslsnum,[(hltpath,hltprescale,l1pass,hltaccept),...]),(cmslsnum,[])})}
    '''
    result={}
    for run in irunlsdict.keys():
        lslist=irunlsdict[run]
        if lslist is not None and len(lslist)==0:
            result[run]=[]#if no LS is selected for a run
            continue
        if not dataidmap.has_key(run):
            continue
        hltdataid=dataidmap[run][2]
        if hltdataid is None:
            result[run]=None
            continue #run non exist
        hltdata=dataDML.hltLSById(schema,hltdataid,hltpathname=hltpathname,hltpathpattern=hltpathpattern,withL1Pass=withL1Pass,withHLTAccept=withHLTAccept)
        #(runnum,{cmslsnum:[(pathname,prescale,l1pass,hltaccept),...]})
        result[run]=[]            
        if hltdata and hltdata[1]:
            lsdict={}            
            for cmslsnum in sorted(hltdata[1]):
                if lslist is not None and cmslsnum not in lslist:
                    continue
                lsdata=[]
                for perpathdata in hltdata[1][cmslsnum]:
                    pathname=perpathdata[0]
                    prescale=perpathdata[1]
                    l1pass=None
                    hltaccept=None
                    if withL1Pass:
                        l1pass=perpathdata[2]
                    if withHLTAccept:
                        hltaccept=perpathdata[3]
                    lsdata.append((pathname,prescale,l1pass,hltaccept))
                result[run].append((cmslsnum,lsdata))
    return result

def trgForIds(schema,irunlsdict,dataidmap,trgbitname=None,trgbitnamepattern=None,withL1Count=False,withPrescale=False):
    '''
    input :
            irunlsdict  {run:[cmsls]} (required)
            dataidmap: {run:(lumiid,trgid,hltid)}
            trgbitname exact match  trgbitname (optional)
            trgbitnamepattern match trgbitname (optional)
    output
            result {run:[[cmslsnum(0),deadfrac(1),deadtimecount(2),bitzero_count(3),bitzero_prescale(4),[(bitname,prescale,counts)](5)]]}
    '''
    result={}
    for run in irunlsdict.keys():
        result[run]=[]
        lslist=irunlsdict[run]
        if lslist is not None and len(lslist)==0:
            #if no LS is selected for a run
            continue
        if not dataidmap.has_key(run):
            continue
        trgdataid=dataidmap[run][1]
        if trgdataid is None:
            result[run]=None
            continue        #if run non exist
        trgdata=dataDML.trgLSById(schema,trgdataid,trgbitname=trgbitname,trgbitnamepattern=trgbitnamepattern,withL1Count=withL1Count,withPrescale=withPrescale)
    
        #(runnum,{cmslsnum:[deadtimecount(0),bitzerocount(1),bitzeroprescale(2),deadfrac(3),[(bitname,trgcount,prescale)](4)]})
        if trgdata and trgdata[1]:
            lsdict={}
            for cmslsnum in sorted(trgdata[1]):
                if lslist is not None and cmslsnum not in lslist:
                    continue
                lsdata=[]
                #print trgdata[1][cmslsnum]
                deadtimecount=trgdata[1][cmslsnum][0]
                #bitzerocount=trgdata[1][cmslsnum][1]
                #bitzeroprescale=trgdata[1][cmslsnum][2]
                bitzerocount=0
                bitzeroprescale=0
                deadfrac=trgdata[1][cmslsnum][3]
                if deadfrac<0 or deadfrac>1.0:
                    deadfrac=1.0
                allbitsinfo=trgdata[1][cmslsnum][4]
                lsdata.append(cmslsnum)
                lsdata.append(deadfrac)
                lsdata.append(deadtimecount)
                lsdata.append(bitzerocount)
                lsdata.append(bitzeroprescale)
                lsdata.append(allbitsinfo)
                result[run].append(lsdata)
    return result

def instLumiForIds(schema,irunlsdict,dataidmap,runsummaryMap,beamstatusfilter=None,timeFilter=None,withBXInfo=False,bxAlgo=None,xingMinLum=None,withBeamIntensity=False,lumitype='HF'):
    '''
    FROM ROOT FILE NO CORRECTION AT ALL 
    input:
           irunlsdict: {run:[cmsls]} 
           dataidmap: {run:(lumiid,trgid,hltid)}
           runsummaryMap: {run:[l1key(0),amodetag(1),egev(2),hltkey(3),fillnum(4),fillscheme(5),starttime(6),stoptime(7)]}
           beamstatus: LS filter on beamstatus (optional)
           timeFilter: (minLSBegTime,maxLSBegTime)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: None means apply no cut
           withBeamIntensity: get beam intensity info (optional)
           lumitype: luminosity measurement source
    output:
           result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),instlumi(5),instlumierr(6),startorbit(7),numorbit(8),(bxidx,bxvalues,bxerrs)(9),(bxidx,b1intensities,b2intensities)(10),fillnum(11)]}}
           
           special meanings:
           {run:None}  None means selected run not in lumiDB, 
           {run:[]} [] means no lumi data for this run in lumiDB
           {run:cmslsnum(1)==0} means either not cmslsnum or iscms but not selected
           instlumi unit in Hz/ub
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    lumitableName=''
    lumilstableName=''
    if lumitype=='HF':
        lumitableName=nameDealer.lumidataTableName()
        lumilstableName=nameDealer.lumisummaryv2TableName()
    else:
        lumitableName=nameDealer.pixellumidataTableName()
        lumilstableName=nameDealer.pixellumisummaryv2TableName()
    result={}
    for run in irunlsdict.keys():
    #for run,(lumidataid,trgid,hltid ) in dataidmap.items():
        lslist=irunlsdict[run]
        if lslist is not None and len(lslist)==0:
            result[run]=[]#no lumi data for this run in lumiDB
            continue
        fillnum=runsummaryMap[run][4]
        runstarttimeStr=runsummaryMap[run][6]
        if not dataidmap.has_key(run):
            result[run]=[]#no lumi data for this run in lumiDB
            continue
        (lumidataid,trgid,hltid )=dataidmap[run]
        if lumidataid is None: #selected run not in lumiDB
            result[run]=None
            continue
        (lumirunnum,perlsresult)=dataDML.lumiLSById(schema,lumidataid,beamstatus=beamstatusfilter,withBXInfo=withBXInfo,bxAlgo=bxAlgo,withBeamIntensity=withBeamIntensity,tableName=lumilstableName)
        lsresult=[]
        c=lumiTime.lumiTime()
        for lumilsnum in perlsresult.keys():
            perlsdata=perlsresult[lumilsnum]
            cmslsnum=perlsdata[0]
            if lslist is not None and cmslsnum not in lslist: #ls exists but not selected
                cmslsnum=0
            numorbit=perlsdata[6]
            startorbit=perlsdata[7]
            orbittime=c.OrbitToTime(runstarttimeStr,startorbit,0)
            if timeFilter:
                if timeFilter[0]:
                    if orbittime<timeFilter[0]: continue
                if timeFilter[1]:
                    if orbittime>timeFilter[1]: continue
            if lumitype=='HF':
                instlumi=perlsdata[1]*1000.0 #HF db avg values are in Hz/mb,change it to Hz/ub
                instlumierr=perlsdata[2]*1000.0
            else:
                instlumi=perlsdata[1] #PIXEL avg values are in Hz/ub, need no conversion
                instlumierr=perlsdata[2]
            beamstatus=perlsdata[4]
            beamenergy=perlsdata[5]
            bxidxlist=[]
            bxvaluelist=[]
            bxerrorlist=[]
            bxdata=None
            beamdata=None
            if withBXInfo:
                bxinfo=perlsdata[8]                
                bxvalueArray=None
                bxerrArray=None
                if bxinfo:
                    bxvalueArray=bxinfo[0]
                    bxerrArray=bxinfo[1]
                    if xingMinLum :
                        for idx,bxval in enumerate(bxvalueArray):
                            if bxval>xingMinLum:
                                bxidxlist.append(idx)
                                bxvaluelist.append(bxval)
                                bxerrorlist.append(bxerrArray[idx])
                    else:
                        bxidxlist=range(0,len(bxvalueArray))
                        bxvaluelist=bxvalueArray.tolist()
                        bxerrorlist=bxerrArray.tolist()
                    del bxvalueArray[:]
                    del bxerrArray[:]
                bxdata=(bxidxlist,bxvaluelist,bxerrorlist)
            if withBeamIntensity:
                beaminfo=perlsdata[9]
                bxindexlist=[]
                b1intensitylist=[]
                b2intensitylist=[]
                if beaminfo[0] and beaminfo[1] and beaminfo[2]:
                    bxindexarray=beaminfo[0]
                    beam1intensityarray=beaminfo[1]
                    beam2intensityarray=beaminfo[2]                    
                    bxindexlist=bxindexarray.tolist()
                    b1intensitylist=beam1intensityarray.tolist()
                    b2intensitylist=beam2intensityarray.tolist()
                    del bxindexarray[:]
                    del beam1intensityarray[:]
                    del beam2intensityarray[:]                    
                beamdata=(bxindexlist,b1intensitylist,b2intensitylist)
            lsresult.append([lumilsnum,cmslsnum,orbittime,beamstatus,beamenergy,instlumi,instlumierr,startorbit,numorbit,bxdata,beamdata,fillnum])         
            del perlsdata[:]
        result[run]=lsresult
    return result

def deliveredLumiForIds(schema,irunlsdict,dataidmap,runsummaryMap,beamstatusfilter=None,timeFilter=None,normmap=None,withBXInfo=False,bxAlgo=None,xingMinLum=None,withBeamIntensity=False,lumitype='HF',minbiasXsec=None):
    '''
    delivered lumi (including calibration,time integral)
    input:
       irunlsdict:  {run:[lsnum]}, where [lsnum]==None means all ; [lsnum]==[] means selected ls
       dataidmap : {run:(lumiid,trgid,hltid)}
       runsummaryMap: {run:[l1key(0),amodetag(1),egev(2),hltkey(3),fillnum(4),fillscheme(5),starttime(6),stoptime(7)]}
       beamstatus: LS filter on beamstatus 
       normmap: {since:[corrector(0),{paramname:paramvalue}(1),amodetag(2),egev(3),comment(4)]} if normmap empty, means without-correction , if notnormmap means without-correction
       withBXInfo: get per bunch info (optional)
       bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
       xingMinLum: cut on bx lumi value (optional)
       withBeamIntensity: get beam intensity info (optional)
       lumitype: luminosity source
    output:
       result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),calibratedlumierr(6),(bxidxlist,bxvalues,bxerrs)(7),(bxidx,b1intensities,b2intensities)(8),fillnum(9),pu(10)]}
       
       special meanings:
       {run:None}  None means no run in lumiDB, 
       {run:[]} [] means no lumi for this run in lumiDB
       {run:cmslsnum(1)==0} means either not cmslsnum or iscms but not selected 
       lumi unit: /ub
    '''
    result = {}
    lumip=lumiParameters.ParametersObject()
    lumirundata=dataDML.lumiRunByIds(schema,dataidmap,lumitype=lumitype)
    instresult=instLumiForIds(schema,irunlsdict,dataidmap,runsummaryMap,beamstatusfilter=beamstatusfilter,timeFilter=timeFilter,withBXInfo=withBXInfo,bxAlgo=bxAlgo,withBeamIntensity=withBeamIntensity,lumitype=lumitype)
    
    intglumimap={}
    if lumitype=='HF':
        intglumimap=dataDML.intglumiForRange(schema,irunlsdict.keys())#some runs need drift correction
    allsince=[]
    if normmap:
        allsince=normmap.keys()
        allsince.sort()        
    correctorname='fPoly' #HF default
    correctionparams={'a0':1.0}
    runfillschemeMap={}
    fillschemePatternMap={}
    if lumitype=='PIXEL':
        correctorname='fPolyScheme' #PIXEL default
        fillschemePatternMap=dataDML.fillschemePatternMap(schema,'PIXEL')
    for run,perrundata in instresult.items():
        if perrundata is None:
            result[run]=None
            continue
        intglumi=0.
        if normmap and intglumimap and intglumimap.has_key(run) and intglumimap[run]:
            intglumi=intglumimap[run]
        nBXs=0
        if normmap and lumirundata and lumirundata.has_key(run) and lumirundata[run][2]:
            nBXs=lumirundata[run][2]
        fillschemeStr=''
        if normmap and runsummaryMap and runsummaryMap.has_key(run) and runsummaryMap[run][5]:
            fillschemeStr=runsummaryMap[run][5]
        if allsince:
            lastsince=allsince[0]
            for since in allsince:
                if run>=since:
                    lastsince=since
            correctorname=normmap[lastsince][0]
            correctionparams=normmap[lastsince][1]
            
        correctioninput=[0.,intglumi,nBXs,fillschemeStr,fillschemePatternMap]
        result[run]=[]
        for perlsdata in perrundata:#loop over ls
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            timestamp=perlsdata[2]
            bs=perlsdata[3]
            beamenergy=perlsdata[4]
            instluminonorm=perlsdata[5]
            correctioninput[0]=instluminonorm
            totcorrectionFac=normFunctors.normFunctionCaller(correctorname,*correctioninput,**correctionparams)
            fillnum=perlsdata[11]
            instcorrectedlumi=totcorrectionFac*instluminonorm
            numorbit=perlsdata[8]
            numbx=lumip.NBX
            lslen=lumip.lslengthsec()
            deliveredlumi=instcorrectedlumi*lslen
            calibratedbxdata=None
            beamdata=None
            pu=0.#avgPU
            if nBXs and minbiasXsec:
                pu=(instcorrectedlumi/nBXs)*minbiasXsec/lumip.rotationRate                
            if withBXInfo:                
                (bxidxData,bxvaluesData,bxerrsData)=perlsdata[9]
                if lumitype=='HF':
                    if xingMinLum:
                        bxidxList=[]
                        bxvalueList=[]
                        bxerrList=[]
                        for idx,bxval in enumerate(bxvaluesData):
                            correctedbxintlumi=totcorrectionFac*bxval
                            correctedbxintlumierr=totcorrectionFac*bxerrsData[idx]
                            if correctedbxintlumi>xingMinLum:
                                bxidxList.append(bxidxData[idx])
                                bxvalueList.append(correctedbxintlumi)
                                bxerrList.append(correctedbxintlumierr)
                        calibratedbxdata=(bxidxList,bxvalueList,bxerrList)
                    else:
                        calibratedbxvalue=[totcorrectionFac*x for x in bxvaluesData]
                        calibratedlumierr=[totcorrectionFac*x for x in bxerrsData]
                        calibratedbxdata=(bxidxData,calibratedbxvalue,calibratedlumierr)
            if withBeamIntensity:
                beamdata=perlsdata[10]
            calibratedlumierr=0.0
            result[run].append([lumilsnum,cmslsnum,timestamp,bs,beamenergy,deliveredlumi,calibratedlumierr,calibratedbxdata,beamdata,fillnum,pu])
            del perlsdata[:]
    return result

def lumiForIds(schema,irunlsdict,dataidmap,runsummaryMap,beamstatusfilter=None,timeFilter=None,normmap=None,withBXInfo=False,bxAlgo=None,xingMinLum=None,withBeamIntensity=False,lumitype='HF',minbiasXsec=None):
    '''
    delivered/recorded lumi  (including calibration,time integral)
    input:
       irunlsdict:  {run:[lsnum]}, where [lsnum]==None means all ; [lsnum]==[] means no selected ls
       dataidmap : {run:(lumiid,trgid,hltid)}
       runsummaryMap: {run:[l1key(0),amodetag(1),egev(2),hltkey(3),fillnum(4),fillscheme(5),starttime(6),stoptime(7)]}
       beamstatus: LS filter on beamstatus 
       normmap: 
       withBXInfo: get per bunch info (optional)
       bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
       xingMinLum: cut on bx lumi value (optional)
       withBeamIntensity: get beam intensity info (optional)
       lumitype: luminosity source
    output:
       result {run:[[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),(bxidx,b1intensities,b2intensities)(9),fillnum(10),ncollidingbunches(11)]...]}
       special meanings:
       {run:None}  None means no run in lumiDB, 
       {run:[]} [] means no lumi for this run in lumiDB
       {run:[....deliveredlumi(5),recordedlumi(6)None]} means no trigger in lumiDB
       {run:cmslsnum(1)==0} means either not cmslsnum or is cms but not selected, therefore set recordedlumi=0,efflumi=0
       lumi unit: 1/ub
    '''
    deliveredresult=deliveredLumiForIds(schema,irunlsdict,dataidmap,runsummaryMap,beamstatusfilter=beamstatusfilter,timeFilter=timeFilter,normmap=normmap,withBXInfo=withBXInfo,bxAlgo=bxAlgo,xingMinLum=xingMinLum,withBeamIntensity=withBeamIntensity,lumitype=lumitype,minbiasXsec=minbiasXsec)
    trgresult=trgForIds(schema,irunlsdict,dataidmap)
    for run in deliveredresult.keys():#loop over delivered,already selected
        perrundata=deliveredresult[run]
        if perrundata is None or len(perrundata)==0: #pass through 
            continue
        alltrgls=[]
        if trgresult.has_key(run) and trgresult[run]:
            alltrgls=[x[0] for x in trgresult[run]]
        for perlsdata in perrundata:#loop over ls
            if not perlsdata: continue #no lumi data for this ls
            perlsdata.insert(6,None)
            if not alltrgls: continue  #no trg for this run,recorded=None 
            cmslsnum=perlsdata[1]
            if cmslsnum==0:#if not a cmsls or not selected by cms list, set recordedlumi to 0
                recordedlumi=0.0
            else:
                try:
                    trglsidx=alltrgls.index(cmslsnum)
                    deadfrac=trgresult[run][trglsidx][1]
                    if deadfrac<0 or deadfrac>1.0: deadfrac=1.0
                    deliveredlumi=perlsdata[5]
                    recordedlumi=(1.0-deadfrac)*deliveredlumi
                except ValueError:
                    #print '[WARNING] no trigger for LS=',cmslsnum
                    recordedlumi=None
            perlsdata[6]=recordedlumi
    return deliveredresult

def effectiveLumiForIds(schema,irunlsdict,dataidmap,runsummaryMap=None,beamstatusfilter=None,timeFilter=None,normmap=None,hltpathname=None,hltpathpattern=None,withBXInfo=False,bxAlgo=None,xingMinLum=None,withBeamIntensity=False,lumitype='HF',minbiasXsec=None):
    '''
    delivered/recorded/eff lumi in selected hlt path  (including calibration,time integral)
    input:
           irunlsdict: {run:[lsnum]}, where [lsnum]==None means all ; [lsnum]==[] means selected ls
           dataidmap : {run:(lumiid,trgid,hltid)}
           runsummaryMap: {run:[l1key(0),amodetag(1),egev(2),hltkey(3),fillnum(4),fillscheme(5),starttime(6),stoptime(7)]}
           beamstatusfilter: LS filter on beamstatus
           normmap: {since:[corrector(0),{paramname:paramvalue}(1),amodetag(2),egev(3),comment(4)]} if normmap empty, means without-correction , if notnormmap means without-correction
           hltpathname: selected hltpathname
           hltpathpattern: regex select hltpaths           
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamIntensity: get beam intensity info (optional)
           lumitype: luminosity source
    output:
           result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]}(8),bxdata(9),beamdata(10),fillnum(11),ncollidingbunches(12)]}
           {run:None}  None means no run in lumiDB, 
           {run:[]} [] means no lumi for this run in lumiDB
           {run:[....deliveredlumi(5),recorded(6)==None,]} means no trigger in lumiDB
           {run:[....deliveredlumi(5),recorded(6),calibratedlumierror(7)==None]} means no hlt in lumiDB
           
           lumi unit: 1/ub
    '''
    deliveredresult=deliveredLumiForIds(schema,irunlsdict,dataidmap,runsummaryMap,beamstatusfilter=beamstatusfilter,timeFilter=timeFilter,normmap=normmap,withBXInfo=withBXInfo,bxAlgo=bxAlgo,xingMinLum=xingMinLum,withBeamIntensity=withBeamIntensity,lumitype=lumitype,minbiasXsec=minbiasXsec)
    trgresult=trgForIds(schema,irunlsdict,dataidmap,withPrescale=True) #{run:[cmslsnum,deadfrac,deadtimecount,bitzero_count,bitzero_prescale,[(bitname,prescale,counts)]]}
    hltresult=hltForIds(schema,irunlsdict,dataidmap,hltpathname=hltpathname,hltpathpattern=hltpathpattern,withL1Pass=False,withHLTAccept=False) #{runnumber:[(cmslsnum,[(hltpath,hltprescale,l1pass,hltaccept),...]),(cmslsnum,[])})}
    for run in deliveredresult.keys(): #loop over delivered
        perrundata=deliveredresult[run]
        if perrundata is None or len(perrundata)==0:#pass through 
            continue
        alltrgls=[]
        if trgresult.has_key(run) and trgresult[run]:
            alltrgls=[x[0] for x in trgresult[run]]
        allhltls=[]
        if hltresult.has_key(run) and hltresult[run]:
            allhltls=[x[0] for x in hltresult[run]]            
        l1bitinfo=[]
        hltpathinfo=[]
        hlttrgmap=dataDML.hlttrgMappingByrun(schema,run,hltpathname=hltpathname,hltpathpattern=hltpathpattern)
        for perlsdata in perrundata: #loop over ls
            if not perlsdata: continue #no lumi for this ls
            perlsdata.insert(6,None)
            perlsdata.insert(8,None)
            if not alltrgls: continue  #no trg for this run
            cmslsnum=perlsdata[1]
            recordedlumi=0.0
            if cmslsnum==0:#if not a cmsls or not selected by cms list, set recordedlumi,efflumi to 0
                continue
            else:
                try:
                    trglsidx=alltrgls.index(cmslsnum)
                    deadfrac=trgresult[run][trglsidx][1]
                    l1bitinfo=trgresult[run][trglsidx][5]
                    if deadfrac<0 or deadfrac>1.0:deadfrac=1.0
                    deliveredlumi=perlsdata[5]
                    recordedlumi=(1.0-deadfrac)*deliveredlumi
                except ValueError:
                    #print '[WARNING] no trigger for LS=',cmslsnum
                    continue #do not go further
            perlsdata[6]=recordedlumi
            if not allhltls: continue #no hlt for this run
            try:
                hltlsidx=allhltls.index(cmslsnum)
            except ValueError:
                #print '[WARNING] no hlt for LS=',cmslsnum
                continue #do not go further
            trgprescalemap={} #{bitname:l1prescale} for this lumi section
            if l1bitinfo:
                for thisbitinfo in l1bitinfo:
                    thisbitname=thisbitinfo[0]
                    thisbitprescale=thisbitinfo[2]
                    trgprescalemap['"'+thisbitname+'"']=thisbitprescale
            else:
                continue
            hltpathdata=hltresult[run][hltlsidx][1]
            efflumidict={}#{pathname:[[l1bitname,l1prescale,hltprescale,efflumi]]}       
            for pathidx,thispathinfo in enumerate(hltpathdata):
                thispathname=thispathinfo[0]
                thisprescale=thispathinfo[1]
                thisl1seed=None
                l1bitname=None
                l1prescale=None
                try:
                    thisl1seed=hlttrgmap[thispathname]
                except KeyError:
                    thisl1seed=None
                if thisl1seed:
                    try:
                        (exptype,l1bits)=hltTrgSeedMapper.findUniqueSeed(thispathname,thisl1seed)
                        if l1bits:
                            if not exptype:
                                l1bitname=l1bits[0]
                                l1prescale=trgprescalemap[l1bits[0]]#need to match double quoted string!                                
                            else:
                                pmin=99999999
                                pmax=0                                
                                for bit in l1bits:
                                    l1p=trgprescalemap[bit]
                                    if exptype=='OR':
                                        if l1p!=0 and l1p<pmin:
                                            pmin=l1p
                                            l1prescale=l1p
                                            l1bitname=bit
                                    if exptype=='AND':
                                        if l1p!=0 and l1p>pmax:
                                            pmax=l1p
                                            l1prescale=l1p
                                            l1bitname=bit
                        else:
                            l1prescale=None
                    except KeyError:
                        l1prescale=None
            
                efflumi=0.0
                if l1prescale and thisprescale:#normal both prescaled
                    efflumi=recordedlumi/(float(l1prescale)*float(thisprescale))
                    efflumidict[thispathname]=[l1bitname,l1prescale,thisprescale,efflumi]
                elif l1prescale and thisprescale==0: #hltpath in menu but masked
                    efflumidict[thispathname]=[l1bitname,l1prescale,thisprescale,efflumi]
                else:
                    efflumidict[thispathname]=[None,0,thisprescale,efflumi]
            perlsdata[8]=efflumidict
    return deliveredresult


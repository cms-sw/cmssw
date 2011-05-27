import os,coral,datetime,fnmatch
from RecoLuminosity.LumiDB import nameDealer,revisionDML,dataDML,lumiTime,CommonUtil,selectionParser,hltTrgSeedMapper


#internal functions
#
#to decide on the norm value to use
#
def _getnorm(schema,norm):
    if isinstance(norm,int) or isinstance(norm,float) or CommonUtil.is_floatstr(norm) or CommonUtil.is_intstr(norm):
        return float(norm)
    if not isinstance(norm,str):
        raise ValueError('wrong parameter type')
    normdataid=dataDML.guessnormIdByName(schema,norm)
    normresult=dataDML.luminormById(schema,normdataid)
    return normresult[2]
def _decidenormFromContext(schema,amodetag,egev):
    normdataid=dataDML.guessnormIdByContext(schema,amodetag,egev)
    normresult=dataDML.luminormById(schema,normdataid)
    return normresult[2]
def _decidenormForRun(schema,run):
    rundata=dataDML.runsummary(schema,run)
    amodetagForRun=rundata[1]
    egevForRun=rundata[2]
    normdataid=dataDML.guessnormIdByContext(schema,amodetagForRun,egevForRun)
    normresult=dataDML.luminormById(schema,normdataid)
    return normresult[2]
#public functions
def runList(schema,fillnum=None,runmin=None,runmax=None,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=None,nominalEnergy=None,energyFlut=0.2,requiretrg=True,requirehlt=True):
    return dataDML.runList(schema,fillnum,runmin,runmax,startT,stopT,l1keyPattern,hltkeyPattern,amodetag,nominalEnergy,energyFlut,requiretrg,requirehlt)

def lslengthsec(numorbit, numbx):
    '''
    input:
       numorbit : number of orbit in the lumi section
       numbx : number of orbits
    output:
       lumi section length in sec
    '''
    l = numorbit * numbx * 25.0e-09
    return l
def hltpathsForRange(schema,runlist):
    '''
    input: runlist [run],     (required)      
           datatag: data version (optional)
    output : {runnumber,{hltpath:(l1bitname,l1seedexpr)}}
    '''
    result={}
    if isinstance(inputRange,list):
        for run in runlist:
            extendedmap={}
            hlttrgmap=dataDML.hlttrgMappingByrun(run)
            for hltpath,l1seedexpr in  hlttrgmap.items():
                l1bitname=hltTrgSeedMapper(hltpath,l1seedexpr)
                extendedmap[hltpath]=(l1bitname,l1seedexpr)
            result[run]=extendedmap
    return result
def trgbitsForRange(schema,runlist,datatag=None):
    '''
    input: runlist [run],(required)
           datatag: data version (optional)
    output: {runnumber:bitzeroname,[bitnames]}
    '''
    pass

def hltForRange(schema,inputRange,hltpathname=None,hltpathpattern=None,datatag=None):
    '''
    input:
           inputRange: {run:[cmsls]} (required)
           hltpathname: exact match hltpathname  (optional) 
           hltpathpattern: regex match hltpathpattern (optional)
           datatag : data version
    output: {runnumber:{hltpath:[[cmslsnum,l1pass,hltaccept,hltprescale]]})}
    '''
    pass

def trgForRange(schema,inputRange,trgbitname=None,trgbitnamepattern=None,datatag=None):
    '''
    input :
            inputRange  {run:[cmsls]} (required)
            trgbitname exact match  trgbitname (optional)
            trgbitnamepattern regex match trgbitname (optional)
            datatag : data version
    output
            result {run,{cmslsnum:[deadtimecount,bitzero_count,bitzero_prescale,deadfraction,{bitname:[prescale,counts]}]}}
    '''
    pass

def instLumiForRange(schema,inputRange,beamstatusfilter=None,withBXInfo=False,bxAlgo=None,xingMinLum=0.0,withBeamIntensity=False,datatag=None):
    '''
    lumi raw data. beofore normalization and time integral
    input:
           inputRange  {lumidataid:[cmsls]} (required)
           beamstatus: LS filter on beamstatus (optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamIntensity: get beam intensity info (optional)
           datatag: data version
    output:
           result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),instlumi(5),instlumierr(6),startorbit(7),numorbit(8),(bxvalues,bxerrs)(9),(bxidx,b1intensities,b2intensities)(10)]}}
           lumi unit: HZ/ub
    '''
    result={}
    for run in inputRange.keys():
        lslist=inputRange[run]
        if lslist is not None and len(lslist)==0:
            result[run]=[]#if no LS is selected for a run
            continue
        runsummary=dataDML.runsummary(schema,run)
        if len(runsummary)==0:#if run not found in runsummary
            result[run]=None
            continue
        runstarttimeStr=runsummary[6]
        lumidataid=dataDML.guessLumiDataIdByRun(schema,run)
        if lumidataid is None: #if run not found in lumidata
            result[run]=None
            continue
        perlsresult=dataDML.lumiLSById(schema,lumidataid,beamstatusfilter,withBXInfo=withBXInfo,bxAlgo=bxAlgo,withBeamIntensity=withBeamIntensity)[1]
        lsresult=[]
        c=lumiTime.lumiTime()
        for lumilsnum,perlsdata in perlsresult.items():
            cmslsnum=perlsdata[0]
            if lslist is not None and lumilsnum not in lslist:
                cmslsnum=0
                recordedlumi=0.0
            numorbit=perlsdata[6]
            startorbit=perlsdata[7]
            orbittime=c.OrbitToTime(runstarttimeStr,startorbit,0)
            instlumi=perlsdata[1]
            instlumierr=perlsdata[2]
            beamstatus=perlsdata[4]
            beamenergy=perlsdata[5]
            bxvaluelist=[]
            bxerrorlist=[]
            bxdata=None
            beamdata=None
            if withBXInfo:
                bxinfo=perlsdata[8]                
                bxvaluesArray=None
                bxerrorsArray=None
                if bxinfo:
                    bxvalueArray=bxinfo[0]
                    bxerrArray=bxinfo[1]
                    for idx,bxval in enumerate(bxvalueArray):
                        if bxval>xingMinLum:
                            bxidxlist.append(idx)
                            bxvaluelist.append(bxval)
                            bxerrorlist.append(bxerrArray[idx])
                    del bxvalueArray[:]
                    del bxerrArray[:]
                bxdata=(bxidxlist,bxvaluelist,bxerrorlist)
            if withBeamIntensity:
                beaminfo=perlsdata[9]
                bxindexlist=[]
                b1intensitylist=[]
                b2intensitylist=[]
                if beaminfo:
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
            lsresult.append([lumilsnum,cmslsnum,orbittime,beamstatus,beamenergy,instlumi,instlumierr,startorbit,numorbit,bxdata,beamdata])         
            del perlsdata[:]
        result[run]=lsresult
    return result

def instCalibratedLumiForRange(schema,inputRange,beamstatus=None,amodetag=None,egev=None,withBXInfo=False,bxAlgo=None,xingMinLum=0.0,withBeamIntensity=False,norm=None,datatag=None):
    '''
    Inst luminosity after calibration, not time integrated
    input:
           inputRange  {run:[cmsls]} (required)
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','IONPHYS']
           beamstatus: LS filter on beamstatus (optional)
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamIntensity: get beam intensity info (optional)
           norm: if norm is a float, use it directly; if it is a string, consider it norm factor name to use (optional)
           datatag: data version
    output:
           result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),calibratedlumi(5),calibratedlumierr(6),startorbit(7),numorbit(8),(bxvalues,bxerrs)(9),(bxidx,b1intensities,b2intensities)(10)]}}
           lumi unit: HZ/ub
    '''
    result = {}
    normval=None
    perbunchnormval=None
    if norm:
        normval=_getnorm(schema,norm)
        perbunchnormval=float(normval)/float(1000)
    elif amodetag and egev:
        normval=_decidenormFromContex(schema,amodetag,egev)
        perbunchnormval=float(normval)/float(1000)
    instresult=instLumiForRange(schema,inputRange,beamstatusfilter=beamstatus,withBXInfo=withBXInfo,bxAlgo=bxAlgo,xingMinLum=xingMinLum,withBeamIntensity=withBeamIntensity,datatag=datatag)
    for run,perrundata in instresult.items():
        if perrundata is None:
            result[run]=None
            continue
        result[run]=[]
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        for perlsdata in perrundata:
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            timestamp=perlsdata[2]
            bs=perlsdata[3]
            beamenergy=perlsdata[4]
            calibratedlumi=perlsdata[5]*normval             
            calibratedlumierr=perlsdata[6]*normval
            startorbit=perlsdata[7]
            numorbit=perlsdata[8]
            calibratedbxdata=None
            beamdata=None
            if withBXInfo:
                bxdata=perlsdata[9]
                if bxdata:
                    calibratedbxdata=([x*perbunchnormval for x in bxdata[0]],[x*perbunchnormval for x in bxdata[1]])
                    del bxdata[0][:]
                    del bxdata[1][:]
            if withBeamIntensity:
                beamdata=perlsdata[10]                
            result[run].append([lumilsnum,cmslsnum,timestamp,bs,beamenergy,calibratedlumi,calibratedlumierr,startorbit,numorbit,calibratedbxdata,beamdata])
            del perlsdata[:]
    return result
         
def deliveredLumiForRange(schema,inputRange,beamstatus=None,amodetag=None,egev=None,withBXInfo=False,bxAlgo=None,xingMinLum=0.0,withBeamIntensity=False,norm=None,datatag=None):
    '''
    delivered lumi (including calibration,time integral)
    input:
           inputRange  {run:[lsnum]} (required) [lsnum]==None means all ; [lsnum]==[] means selected ls 
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','IONPHYS']
           beamstatus: LS filter on beamstatus (optional)
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamIntensity: get beam intensity info (optional)
           norm: norm factor name to use: if float, apply directly, if str search norm by name (optional)
           datatag: data version
    output:
           result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),calibratedlumierr(6),(bxvalues,bxerrs)(7),(bxidx,b1intensities,b2intensities)(8)]}
           avg lumi unit: 1/ub
    '''
    result = {}
    normval=None
    perbunchnormval=None
    if norm:
        normval=_getnorm(schema,norm)
        perbunchnormval=float(normval)/float(1000)
    elif amodetag and egev:
        normval=_decidenormFromContext(schema,amodetag,egev)
        perbunchnormval=float(normval)/float(1000)
    instresult=instLumiForRange(schema,inputRange,beamstatusfilter=beamstatus,withBXInfo=withBXInfo,bxAlgo=bxAlgo,xingMinLum=xingMinLum,withBeamIntensity=withBeamIntensity,datatag=datatag)
    #instLumiForRange should have aleady handled the selection,unpackblob    
    for run,perrundata in instresult.items():
        if perrundata is None:
            result[run]=None
            continue
        result[run]=[]
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        for perlsdata in perrundata:#loop over ls
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            timestamp=perlsdata[2]
            bs=perlsdata[3]
            beamenergy=perlsdata[4]
            calibratedlumi=perlsdata[5]*normval
            calibratedlumierr=perlsdata[6]*normval
            numorbit=perlsdata[8]
            numbx=3564
            lslen=lslengthsec(numorbit,numbx)
            deliveredlumi=calibratedlumi*lslen
            calibratedbxdata=None
            beamdata=None
            if withBXInfo:
                bxdata=perlsdata[9]
                if bxdata:
                    calibratedbxdata=([x*perbunchnormval for x in bxdata[0]],[x*perbunchnormval for x in bxdata[1]])
                del bxdata[0][:]
                del bxdata[1][:]
            if withBeamIntensity:
                beamdata=perlsdata[10]             
            result[run].append([lumilsnum,cmslsnum,timestamp,bs,beamenergy,deliveredlumi,calibratedlumierr,calibratedbxdata,beamdata])
            del perlsdata[:]
    return result
                       
def lumiForRange(schema,inputRange,beamstatus=None,amodetag=None,egev=None,withBXInfo=False,bxAlgo=None,xingMinLum=0.0,withBeamIntensity=False,norm=None,datatag=None):
    '''
    delivered/recorded lumi
    input:
           inputRange  {run:[cmsls]} (required)
           beamstatus: LS filter on beamstatus (optional)
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamIntensity: get beam intensity info (optional)
           normname: norm factor name to use (optional)
           datatag: data version
    output:
           result {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
           lumi unit: 1/ub
    '''
    result = {}
    normval=None
    perbunchnormval=None
    if norm:
        normval=_getnorm(schema,norm)
        perbunchnormval=float(normval)/float(1000)
    elif amodetag and egev:
        normval=_decidenormFromContext(schema,amodetag,egev)
        perbunchnormval=float(normval)/float(1000)
    c=lumiTime.lumiTime()
    for run in inputRange.keys():#loop over run
        lslist=inputRange[run]
        if lslist is not None and len(lslist)==0:#no selected ls, do nothing for this run
            result[run]=[]
            continue
        cmsrunsummary=dataDML.runsummary(schema,run)
        if len(cmsrunsummary)==0:#non existing run
            result[run]=None
            continue
        startTimeStr=cmsrunsummary[6]
        lumidataid=None
        trgdataid=None
        lumidataid=dataDML.guessLumiDataIdByRun(schema,run)
        if lumidataid is None :
            result[run]=None
            continue
        trgdataid=dataDML.guessTrgDataIdByRun(schema,run)
        (lumirunnum,lumidata)=dataDML.lumiLSById(schema,lumidataid,beamstatus=beamstatus,withBXInfo=withBXInfo,bxAlgo=bxAlgo,withBeamIntensity=withBeamIntensity)
        if trgdataid is None :
            trgdata={}
        else:
            (trgrunnum,trgdata)=dataDML.trgLSById(schema,trgdataid,withblobdata=False)
            
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        
        perrunresult=[]
        #print 'run,lslist ',run,lslist
        for lumilsnum,perlsdata in lumidata.items():
            cmslsnum=perlsdata[0]
            triggeredls=cmslsnum
            if lslist is not None and cmslsnum not in lslist:
                cmslsnum=0
                triggeredls=0
                recordedlumi=0.0
            instlumi=perlsdata[1]
            instlumierror=perlsdata[2]
            calibratedlumi=instlumi*normval
            calibratedlumierror=instlumierror*normval
            bstatus=perlsdata[4]
            begev=perlsdata[5]
            numorbit=perlsdata[6]
            startorbit=perlsdata[7]
            timestamp=c.OrbitToTime(startTimeStr,startorbit,0)
            numbx=3564
            lslen=lslengthsec(numorbit,numbx)
            deliveredlumi=calibratedlumi*lslen
            recordedlumi=0.0
            if cmslsnum!=0:
                if not trgdata.has_key(cmslsnum):                    
                   # triggeredls=0 #if no trigger, set back to non-cms-active ls
                    recordedlumi=0.0 # no trigger->nobeam recordedlumi=None
                else:
                    deadcount=trgdata[cmslsnum][0] ##subject to change !!
                    bitzerocount=trgdata[cmslsnum][1]
                    bitzeroprescale=trgdata[cmslsnum][2]
                    if float(bitzerocount)*float(bitzeroprescale)==0.0:
                        deadfrac=1.0
                    else:
                        deadfrac=float(deadcount)/(float(bitzerocount)*float(bitzeroprescale))
                    if deadfrac>1.0:
                        deadfrac=1.0  #artificial correction in case of deadfrac>1
                    recordedlumi=deliveredlumi*(1.0-deadfrac)
                    del trgdata[cmslsnum][:]
            bxdata=None
            if withBXInfo:
                bxinfo=perlsdata[8]
                bxvalueArray=None
                bxerrArray=None
                bxidxlist=[]
                bxvaluelist=[]
                bxerrorlist=[]
                if bxinfo:
                    bxvalueArray=bxinfo[0]
                    bxerrArray=bxinfo[1]
                    #if cmslsnum==1:
                    #    print 'bxvalueArray ',bxvalueArray
                    for idx,bxval in enumerate(bxvalueArray):
                        if bxval*perbunchnormval>xingMinLum:
                            bxidxlist.append(idx)
                            bxvaluelist.append(bxval*perbunchnormval)
                            bxerrorlist.append(bxerrArray[idx]*perbunchnormval)
                    del bxvalueArray[:]
                    del bxerrArray[:]
                bxdata=(bxidxlist,bxvaluelist,bxerrorlist)
            beamdata=None
            if withBeamIntensity:
                beaminfo=perlsdata[9]
                bxindexlist=[]
                b1intensitylist=[]
                b2intensitylist=[]                
                if beaminfo:
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
            perrunresult.append([lumilsnum,triggeredls,timestamp,bstatus,begev,deliveredlumi,recordedlumi,calibratedlumierror,bxdata,beamdata])
            del perlsdata[:]
        result[run]=perrunresult    
    return result
       
def effectiveLumiForRange(schema,inputRange,hltpathname=None,hltpathpattern=None,amodetag=None,beamstatus=None,egev=None,withBXInfo=False,xingMinLum=0.0,bxAlgo=None,withBeamIntensity=False,norm=None,datatag=None):
    '''
    input:
           inputRange  {run:[cmsls]} (required)
           hltpathname: selected hltpathname
           hltpathpattern: regex select hltpaths           
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamIntensity: get beam intensity info (optional)
           normname: norm factor name to use (optional)
           datatag: data version
    output:
    result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]},bxdata,beamdata]}
           lumi unit: 1/ub
    '''
    result = {}
    normval=None
    perbunchnormval=None
    if norm:
        normval=_getnorm(schema,norm)
        perbunchnormval=float(normval)/float(1000)
    elif amodetag and egev:
        normval=_decidenormFromContext(schema,amodetag,egev)
        perbunchnormval=float(normval)/float(1000)
    c=lumiTime.lumiTime()
    for run in sorted(inputRange):
        lslist=inputRange[run]
        if lslist is not None and len(lslist)==0:#no selected ls, do nothing for this run
            result[run]=[]
            continue
        cmsrunsummary=dataDML.runsummary(schema,run)
        if len(cmsrunsummary)==0:#non existing run
            result[run]=None
            continue
        startTimeStr=cmsrunsummary[6]
        lumidataid=None
        trgdataid=None
        hltdataid=None
        (lumidataid,trgdataid,hltdataid)=dataDML.guessAllDataIdByRun(schema,run)
        if lumidataid is None or trgdataid is None or hltdataid is None:
            result[run]=None
            continue
        (lumirunnum,lumidata)=dataDML.lumiLSById(schema,lumidataid,beamstatus)
        (trgrunnum,trgdata)=dataDML.trgLSById(schema,trgdataid,withblobdata=True)
        (hltrunnum,hltdata)=dataDML.hltLSById(schema,hltdataid)
        trgrundata=dataDML.trgRunById(schema,trgdataid)
        hltrundata=dataDML.hltRunById(schema,hltdataid)
        bitnames=trgrundata[3].split(',')
        hlttrgmap=dataDML.hlttrgMappingByrun(schema,run)
        pathnames=hltrundata[3].split(',')
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        perrunresult=[]
        numbx=3564
        for lumilsnum,perlsdata in lumidata.items():
            cmslsnum=perlsdata[0]            
            triggeredls=cmslsnum
            if lslist is not None and cmslsnum not in lslist:
                cmslsnum=0
                triggeredls=0
                recordedlumi=0.0                
            instlumi=perlsdata[1]
            instlumierror=perlsdata[2]
            calibratedlumi=instlumi*normval
            calibratedlumierror=instlumierror*normval
            bstatus=perlsdata[4]
            begev=perlsdata[5]
            numorbit=perlsdata[6]
            startorbit=perlsdata[7]
            timestamp=c.OrbitToUTCTimestamp(startTimeStr,startorbit,0)
            lslen=lslengthsec(numorbit,numbx)
            deliveredlumi=calibratedlumi*lslen
            recordedlumi=0.0
            l1prescaleblob=None
            if cmslsnum!=0:
                if not trgdata.has_key(cmslsnum):
                    #triggeredls=0 #if no trigger, set back to non-cms-active ls
                    recordedlumi=0.0 # no trigger->nobeam recordedlumi=None
                else:
                    deadcount=trgdata[cmslsnum][0] ##subject to change !!
                    bitzerocount=trgdata[cmslsnum][1]
                    bitzeroprescale=trgdata[cmslsnum][2]
                    if float(bitzerocount)*float(bitzeroprescale)==0.0:
                        deadfrac=1.0
                    else:
                        deadfrac=float(deadcount)/(float(bitzerocount)*float(bitzeroprescale))
                    if deadfrac>1.0:
                        deadfrac=1.0  #artificial correction in case of deadfrac>1
                    recordedlumi=deliveredlumi*(1.0-deadfrac)
                    l1prescaleblob=trgdata[cmslsnum][4]
            efflumidict={}
            l1prescalearray=None
            if l1prescaleblob:
                l1prescalearray=CommonUtil.unpackBlobtoArray(l1prescaleblob,'I')
            hltprescaleblob=None
            hltprescalearray=None
            if cmslsnum!=0 and hltdata.has_key(cmslsnum):                
                hltprescaleblob=hltdata[cmslsnum][0]
            if hltprescaleblob:
                hltprescalearray=CommonUtil.unpackBlobtoArray(hltprescaleblob,'I')
            if cmslsnum!=0 and l1prescalearray is not None and hltprescalearray is not None:
                trgprescalemap={} #build trg prescale map for this cmsls {bitname:l1prescale}
                for bitnum,bitprescale in enumerate(l1prescalearray):
                    thisbitname=bitnames[bitnum]
                    trgprescalemap['"'+thisbitname+'"']=bitprescale#note:need to double quote bit name!!
                if hltpathname is None and hltpathpattern is None: #get all paths
                    for hpath,l1seedexpr in hlttrgmap.items():
                        hltprescale=None
                        l1prescale=None
                        efflumi=None     
                        for pathidx,nm in enumerate(pathnames):
                            if nm==hpath:
                                hltprescale=hltprescalearray[pathidx]
                                break
                        try:
                            l1bitname=hltTrgSeedMapper.findUniqueSeed(hpath,l1seedexpr)
                            if l1bitname:
                                l1prescale=trgprescalemap[l1bitname]#need to match double quoted string!
                        except KeyError:
                            l1prescale=None
                        if l1prescale and hltprescale:
                            if l1prescale!=0 and hltprescale!=0:
                                efflumi=recordedlumi/(float(l1prescale)*float(hltprescale))
                            else:
                                efflumi=0.0
                        efflumidict[hpath]=[l1bitname,l1prescale,hltprescale,efflumi]                       
                elif hltpathname is not None and hltpathpattern is None:  #get exact path
                    hltprescale=None
                    l1prescale=None
                    efflumi=None
                    l1seedexpr=hlttrgmap[hltpathname]
                    for pathidx,nm in enumerate(pathnames):
                        if nm==hltpathname:
                            hltprescale=hltprescalearray[pathidx]
                            break
                    try:
                        l1bitname=hltTrgSeedMapper.findUniqueSeed(hltpathname,l1seedexpr)
                        if l1bitname:
                            l1prescale=trgprescalemap[l1bitname]
                    except KeyError:
                        l1prescale=None
                    if l1prescale and hltprescale:
                        if l1prescale!=0 and hltprescale!=0:
                            efflumi=recordedlumi/(float(l1prescale)*float(hltprescale))
                        else:
                            efflumi=0.0
                    efflumidict[hltpathname]=[l1bitname,l1prescale,hltprescale,efflumi]
                elif hltpathpattern is not None: #get paths matching fnmatch pattern
                    for hpath,l1seexexpr in hlttrgmap.items():
                        hltprescale=None
                        l1prescale=None
                        efflumi=None     
                        if fnmatch.fnmatch(hpath,hltpathpattern):#use fnmatch rules
                            for pathidx,nm in enumerate(pathnames):
                                if nm==hpath:
                                    hltprescale=hltprescalearray[pathidx]
                                    break
                            l1seedexpr=hlttrgmap[hpath]
                            try:
                                l1bitname=hltTrgSeedMapper.findUniqueSeed(hpath,l1seedexpr)
                                if l1bitname:
                                    l1prescale=trgprescalemap[l1bitname]
                            except KeyError:
                                l1prescale=None
                            if l1prescale and hltprescale:
                                if l1prescale!=0 and hltprescale!=0:
                                    efflumi=recordedlumi/(float(l1prescale)*float(hltprescale))
                                else:
                                    efflumi=0.0
                            efflumidict[hpath]=[l1bitname,l1prescale,hltprescale,efflumi]
            bxdata=None
            if withBXInfo:
                bxvalueblob=lumidata[8]
                bxerrblob=lumidata[9]
                bxidxlist=[]
                bxvaluelist=[]
                bxerrorlist=[]
                if bxvalueblob is not None and bxerrblob is not None:
                    bxvaluearray=CommonUtil.unpackBlobtoArray(bxvalueblob,'f')
                    bxerrorarray=CommonUtil.unpackBlobtoArray(bxerrblob,'f')
                for idx,bxval in enumerate(bxvaluearray):
                    if bxval*perbunchnormval>xingMinLum:
                        bxidxlist.append(idx)
                        bxvaluelist.append(bxval*perbunchnormval)
                        bxerrorlist.append(bxerrorarray[idx]*perbunchnormval)
                bxdata=(bxidxlist,bxvaluelist,bxerrorlist)
            beamdata=None
            if withBeamIntensity:
                bxindexblob=lumidata[10]
                beam1intensityblob=lumidata[11]
                beam2intensityblob=lumidata[12]
                bxindexlist=[]
                b1intensitylist=[]
                b2intensitylist=[]
                if bxindexblob is not None and beam1intensity is not None and beam2intensity is not None:
                    bxindexlist=CommonUtil.unpackBlobtoArray(bxindexblob,'h').tolist()
                    beam1intensitylist=CommonUtil.unpackBlobtoArray(beam1intensityblob,'f').tolist()
                    beam2intensitylist=CommonUtil.unpackBlobtoArray(beam2intensityblob,'f').tolist()
                    beamdata=(bxindexlist,b1intensitylist,b2intensitylist)
            perrunresult.append([lumilsnum,triggeredls,timestamp,bstatus,begev,deliveredlumi,recordedlumi,calibratedlumierror,efflumidict,bxdata,beamdata])
        result[run]=perrunresult
        #print result
    return result
##===printers
    


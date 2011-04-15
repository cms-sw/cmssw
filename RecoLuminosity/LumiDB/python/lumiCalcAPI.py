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
            result [run,{cmslsnum:[deadtimecount,bitzero_count,bitzero_prescale,deadfraction,{bitname:[prescale,counts]}]}]
    '''
    pass

def instLumiForRange(schema,inputRange,beamstatusfilter=None,withBXInfo=False,bxAlgo='OCC1',withBeamIntensity=False,datatag=None):
    '''
    input:
           inputRange  {lumidataid:[cmsls]} (required)
           beamstatus: LS filter on beamstatus (optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
           datatag: data version
    output:
           result {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),instlumi(5),instlumierr(6),startorbit(7),numorbit(8),(bxvalues,bxerrs)(9),(bxidx,b1intensities,b2intensities)(10)]}}
           lumi unit: HZ/ub
    '''
    result={}
    for run in inputRange.keys():
        lslist=inputRange[run]
        if lslist is not None and len(lslist)==0:
            result[run]={}#if no LS is selected for a run
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
            if lslist is not None and lumilsnum not in inputRange[run]:
                continue
            numorbit=perlsdata[6]
            startorbit=perlsdata[7]
            orbittime=c.OrbitToTime(runstarttimeStr,startorbit,0)
            instlumi=perlsdata[1]
            beamstatus=perlsdata[4]
            beamenergy=perlsdata[5]
            instlumierr=perlsdata[2]
            bxinfo=perlsdata[8]
            beaminfo=perlsdata[9]
            if bxinfo:
                bxvaluesArray=None
                bxerrorsArray=None
                bxvaluesBlob=bxinfo[0]
                bxerrorsBlob=bxinfo[1]
                if bxvaluesBlob:
                    bxvaluesArray=CommonUtil.unpackBlobtoArray(bxvaluesBlob,'f')
                if bxerrorsBlob:
                    bxerrorsArray=CommonUtil.unpackBlobtoArray(bxerrorsBlob,'f')
                bxinfo=(bxvaluesArray,bxerrorsArray)
            if beaminfo:
                bxindexArray=None
                beam1intensityArray=None
                beam2intensityArray=None
                bxindexBlob=beaminfo[0]
                beam1intensityBlob=beaminfo[1]
                beam2intensityBlob=beaminfo[2]
                if bxindexBlob:   
                    bxindexArray=CommonUtil.unpackBlobtoArray(bxindexBlob,'h')
                if beam1intensityBlob: 
                    beam1intensityArray=CommonUtil.unpackBlobtoArray(beam1intensityBlob,'f')
                if beam2intensityBlob:
                    beam2intensityArray=CommonUtil.unpackBlobtoArray(beam2intensityBlob,'f')
                beaminfo=(bxindexArray,beam1intensityArray,beam2intensityArray)
            lsresult.append([lumilsnum,cmslsnum,orbittime,beamstatus,beamenergy,instlumi,instlumierr,startorbit,numorbit,bxinfo,beaminfo])         
            del perlsdata
        result[run]=lsresult
    return result

def instCalibratedLumiForRange(schema,inputRange,beamstatus=None,amodetag=None,egev=None,withBXInfo=False,bxAlgo='OCC1',withBeamInfo=False,norm=None,datatag=None):
    '''
    Inst luminosity after calibration
    input:
           inputRange  {run:[cmsls]} (required)
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','IONPHYS']
           beamstatus: LS filter on beamstatus (optional)
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
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
    instresult=instLumiForRange(schema,inputRange,beamstatus,withBXInfo,bxAlgo,withBeamIntensity,datatag)
    for run,perrundata in instresult.items():
        if perrundata is None:
            result[run]=None
            continue
        result[run]={}
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        for (lumilsnum,cmslsnum),perlsdata in perrundata.items():
            timestamp=perlsdata[0]
            bs=perlsdata[1]
            beamenergy=perlsdata[2]
            calibratedlumi=perlsdata[3]*normval             
            calibratedlumierr=perlsdata[4]*normval
            startorbit=perlsdata[5]
            numorbit=perlsdata[6]
            bxdata=perlsdata[7]
            calibfatedbxdata=None
            if bxdata:
                calibratedbxdata=([x*perbunchnormval for x in bxdata[0]],[x*perbunchnormval for x in bxdata[1]])
            intensitydata=perlsdata[8]             
            result[run].append([lumilsnum,cmslsnum,timestamp,bs,beamenergy,calibratedlumi,calibratedlumierr,startorbit,numorbit,calibfatedbxdata,intensitydata])
    return result
         
def deliveredLumiForRange(schema,inputRange,beamstatus=None,amodetag=None,egev=None,withBXInfo=False,bxAlgo='OCC1',xingMinLum=None,withBeamIntensity=False,norm=None,datatag=None):
    '''
    input:
           inputRange  {run:[lsnum]} (required) [lsnum]==None means all ; [lsnum]==[] means selected ls 
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','IONPHYS']
           beamstatus: LS filter on beamstatus (optional)
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
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
        normval=_decidenormFromContex(schema,amodetag,egev)
        perbunchnormval=float(normval)/float(1000)
    instresult=instLumiForRange(schema,inputRange,beamstatus,withBXInfo,bxAlgo,withBeamIntensity,datatag)
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
            beamstatus=perlsdata[3]
            beamenergy=perlsdata[4]
            calibratedlumi=perlsdata[5]*normval
            calibratedlumierr=perlsdata[6]*normval
            numorbit=perlsdata[8]
            numbx=3564
            lslen=lslengthsec(numorbit,numbx)
            deliveredlumi=calibratedlumi*lslen
            bxdata=perlsdata[9]
            calibratedbxdata=None
            if bxdata:
                calibratedbxdata=([x*perbunchnormval for x in bxdata[0]],[x*perbunchnormval for x in bxdata[1]])
            intensitydata=perlsdata[10]             
            result[run].append([lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierr,calibratedbxdata,intensitydata])
    return result
                       
def lumiForRange(schema,inputRange,beamstatus=None,amodetag=None,egev=None,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamInfo=False,norm=None,datatag=None):
    '''
    input:
           inputRange  {run:[cmsls]} (required)
           beamstatus: LS filter on beamstatus (optional)
           amodetag: amodetag for  picking norm(optional)
           egev: beamenergy for picking norm(optional)
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
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
    for run in sorted(inputRange):#loop over run
        lslist=inputRange[run]
        if lslist is not None and len(lslist)==0:#no selected ls, do nothing for this run
            result[run]=[]
            continue
        cmsrunsummary=dataDML.runsummary(schema,run)
        startTimeStr=cmsrunsummary[6]
        lumidataid=None
        trgdataid=None
        hltdataid=None
        (lumidataid,trgdataid,hltdataid)=dataDML.guessAllDataIdByRun(schema,run)
        if lumidataid is None or trgdataid is None or hltdataid is None:
            result[run]=None
            continue
        (lumirunnum,lumidata)=dataDML.lumiLSById(schema,lumidataid,beamstatus=beamstatus,withBXInfo=withBXInfo,bxAlgo=bxAlgo,withBeamIntensity=withBeamInfo)
        (trgrunnum,trgdata)=dataDML.trgLSById(schema,trgdataid,withblobdata=False)
        
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        
        perrunresult=[]
        for lumilsnum,perlsdata in lumidata.items():
            cmslsnum=perlsdata[0]
            triggeredls=cmslsnum
            if lslist is not None and cmslsnum not in lslist:
                continue
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
                    triggeredls=0 #if no trigger, set back to non-cms-active ls
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
            if withBeamInfo:
                bxindexblob=lumidata[10]
                beam1intensityblob=lumidata[11]
                beam2intensityblob=lumidata[12]
                bxindexlist=[]
                b1intensitylist=[]
                b2intensitylist=[]
                if bxindexblob is not None and beam1intensity is not None and beam2intensity is not None:
                    bxindexlist=CommonUtil.unpackBlobtoArray(bxindexblob,'h').tolist()
                    b1intensitylist=CommonUtil.unpackBlobtoArray(beam1intensityblob,'f').tolist()
                    b2intensitylist=CommonUtil.unpackBlobtoArray(beam2intensityblob,'f').tolist()
                beamdata=(bxindexlist,b1intensitylist,b2intensitylist)
            perrunresult.append([lumilsnum,triggeredls,timestamp,bstatus,begev,deliveredlumi,recordedlumi,calibratedlumierror,bxdata,beamdata])
        result[run]=perrunresult
    return result
       
def effectiveLumiForRange(schema,inputRange,hltpathname=None,hltpathpattern=None,amodetag=None,beamstatus=None,egev=None,withBXInfo=False,xingMinLum=1.0e-4,bxAlgo='OCC1',withBeamInfo=False,norm=None,datatag=None):
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
           withBeamInfo: get beam intensity info (optional)
           normname: norm factor name to use (optional)
           datatag: data version
    output:
    result {run:{(lumilsnum,cmslsnum):[timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,{hltpath:[l1name,l1prescale,hltprescale,efflumi]},bxdata,beamdata]}}
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
            result[run]={}
            continue
        cmsrunsummary=dataDML.runsummary(schema,run)
        startTimeStr=cmsrunsummary[6]
        lumidataid=None
        trgdataid=None
        hltdataid=None
        (lumidataid,trgdataid,hltdataid)=dataDML.guessDataIdByRun(schema,run)
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
        perrunresult={}
        if not normval:#if norm cannot be decided , look for it according to context per run
            normval=_decidenormForRun(schema,run)
            perbunchnormval=float(normval)/float(1000)
        if not normval:#still not found? resort to global default (should never come here)
            normval=6370
            perbunchnormval=6.37
            print '[Warning] using default normalization '+str(normval)
        for lumilsnum,perlsdata in lumidata.items():
            cmslsnum=perlsdata[0]
            if lslist is not None and cmslsnum not in lslist:
                continue
            instlumi=perlsdata[1]
            instlumierror=perlsdata[2]
            calibratedlumi=instlumi*normval
            calibratedlumierror=instlumierror*normval
            bstatus=perlsdata[4]
            begev=perlsdata[5]
            numorbit=perlsdata[6]
            startorbit=perlsdata[7]
            timestamp=c.OrbitToUTCTimestamp(startTimeStr,numorbit)
            numbx=3564
            lslen=lslengthsec(numorbit,numbx)
            deliveredlumi=calibratedlumi*lslen
            recordedlumi=0.0
            
            if cmslsnum==0: continue # skip lumils
            deadcount=trgdata[cmslsnum][0] ##subject to change !!
            bitzerocount=trgdata[cmslsnum][1]
            bitzeroprescale=trgdata[cmslsnum][2]
            deadfrac=float(deadcount)/(float(bitzerocount)*float(bitzeroprescale))
            if deadfrac>1.0:
                deadfrac=1.0 
            recordedlumi=deliveredlumi*(1.0-deadfrac)
            efflumidict={}
            l1prescaleblob=trgdata[cmslsnum][4]
            l1prescalearray=CommonUtil.unpackBlobtoArray(l1prescaleblob,'h')
            hltprescaleblob=hltdata[cmslsnum][0]
            hltprescalearray=CommonUtil.unpackBlobtoArray(hltprescaleblob,'h')
            trgprescalemap={} #build trg prescale map {bitname:l1prescale}
            for bitnum,bitprescale in enumerate(l1prescalearray):
                thisbitname=bitnames[bitnum]
                trgprescalemap[thisbitname]=bitprescale
                       
            if hltpathname is None and hltpathpattern is None: #get all paths                       
                for hpath,l1seedexpr in hlttrgmap.items():
                    hltprescale=None
                    l1prescale=None
                    efflumi=None
                for pathidx,nm in enumerate(hltpathnames):
                    if nm==hpath:
                        hltprescale=hltprescalearray[pathidx]
                        break
                try:
                    l1bitname=hltTrgSeedMapper.findUniqueSeed(hpath,l1seedexpr)
                    if l1bitname:
                        l1prescale=trgprescalemap[l1bitname]
                except KeyError:
                    l1prescale=None
                if l1prescale and hltprescale:
                    efflumi=recordedlumi*l1prescale*hltprescale                           
                    efflumidict[hpath]=[l1bitname,l1prescale,hltprescale,efflumi]                       
                elif hltpathname is not None:  #get one path
                    hltprescale=None
                    l1prescale=None
                    efflumi=None
                for pathidx,nm in enumerate(hltpathnames):
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
                    efflumi=recordedlumi*l1prescale*hltprescale                           
                    efflumidict[hpath]=[l1bitname,l1prescale,hltprescale,efflumi]                           
                elif hltpathpattern is not None: #get paths matching pattern                       
                    for hpath,l1seexexpr in hlttrgmap.items():
                        hltprescale=None
                        l1prescale=None
                        efflumi=None
                        if fnmatch.fnmatch(hpath,hltpathpattern):#use fnmatch rules
                            for pathidx,nm in enumerate(hltpathnames):
                                if nm==hpath:
                                    hltprescale=hltprescalearray[pathidx]
                                    break
                        try:
                            l1bitname=hltTrgSeedMapper.findUniqueSeed(hpath,l1seedexpr)
                            if l1bitname:
                                l1prescale=trgprescalemap[l1bitname]
                        except KeyError:
                            l1prescale=None
                        if l1prescale and hltprescale:
                            efflumi=recordedlumi*l1prescale*hltprescale                           
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
            if withBeamInfo:
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
            perrunresult[(lumilsnum,cmslsnum)]=[timestamp,bstatus,begev,deliveredlumi,recordedlumi,calibratedlumierror,efflumidict,bxdata,beamdata]
            lumidata.clear() #clean lumi memory    
            trgdata.clear()
            hltdata.clear()
            result[run]=perrunresult
    return result
##===printers
    


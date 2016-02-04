import os,coral,datetime,fnmatch
from RecoLuminosity.LumiDB import nameDealer,revisionDML,dataDML,lumiTime,CommonUtil,selectionParser,hltTrgSeedMapper

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

def instLumiForRange(schema,inputRange,beamstatus=None,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamIntensity=False,datatag=None):
    '''
    input:
           inputRange  {run:[cmsls]} (required)
           beamstatus: LS filter on beamstatus (optional)
           beamenergy: LS filter on beamenergy (optional)  beamenergy+-beamenergyFluc
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
           datatag: data version
    output:
           result {run:{(lumilsnum,cmslsnum):[timestamp,beamstatus,beamenergy,instlumi,instlumierr,startorbit,numorbit,(bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}}
           lumi unit: HZ/ub
    '''
    pass

def instCalibratedLumiForRange(schema,inputRange,amodetag='PROTPHYS',nominalegev=3500,beamstatus=None,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamInfo=False,normname=None,datatag=None):
    '''
    Inst luminosity after calibration
    input:
           inputRange  {run:[cmsls]} (required)
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','HIPHYS']
           beamstatus: LS filter on beamstatus (optional)
           beamenergy: LS filter on beamenergy (optional)  beamenergy+-beamenergyFluc
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
           normname: norm factor name to use (optional)
           datatag: data version
    output:
           result {run:{(lumilsnum,cmslsnum):[timestamp,beamstatus,beamenergy,calibratedlumi,calibratedlumierr,startorbit,numorbit,(bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}}
           lumi unit: HZ/ub
    '''
    result = {}
    normid=None
    if not normname:
        normid=dataDML.guessnormIdByContext(schema,amodetag,nominalegev)
    if not normid:
        raise ValueError('cannot find a normalization factor for the combined condition '+amodetag+' '+nominalegev)
    normval=dataDML.luminormById(schema,normid)[2]
    perbunchnormval=float(normval)/float(1000)
    instresult=instLumiForRange(schema,inputRange,beamstatus,withBXInfo,bxAlgo,withBeamIntensity,datatag)
    for run,lsdict in instresult.items():
        result[run]={}
        for (lumilsnum,cmslsnum),perlsdata in lsdict.items():
            timestamp=perlsdata[0]
            beamstatus=perlsdata[1]
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
            result[run][(lumilsnum,cmslsnum)]=[timestamp,beamstatus,beamenergy,calibratedlumi,calibratedlumierr,startorbit,numorbit,calibfatedbxdata,intensitydata]
    return result
         
def deliveredLumiForRange(schema,inputRange,amodetag='PROTPHYS',nominalegev=3500,beamstatus=None,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamIntensity=False,normname=None,datatag=None):
    '''
    input:
           inputRange  {run:[lsnum]} (required) [lsnum]==None means all ; [lsnum]==[] means selected ls 
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','HIPHYS']
           beamstatus: LS filter on beamstatus (optional)
           beamenergy: LS filter on beamenergy (optional)  beamenergy+-beamenergyFluc
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
           normname: norm factor name to use (optional)
           datatag: data version
    output:
           result {run:{(lumilsnum,cmslsnum):[timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierr,(bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}}
           avg lumi unit: 1/ub
    '''
    result = {}
    normid=None
    if not normname:
        normid=dataDML.guessnormIdByContext(schema,amodetag,nominalegev)
    if not normid:
        raise ValueError('cannot find a normalization factor for the combined condition '+amodetag+' '+nominalegev)
    normval=dataDML.luminormById(schema,normid)[2]
    perbunchnormval=float(normval)/float(1000)
    instresult=instLumiForRange(schema,inputRange,beamstatus,withBXInfo,bxAlgo,withBeamIntensity,datatag)
    for run,lslist in inputRange.items():
        result[run]={}
        for (lumilsnum,cmslsnum),perlsdata in lsdict.items():
            timestamp=perlsdata[0]
            beamstatus=perlsdata[1]
            beamenergy=perlsdata[2]
            calibratedlumi=perlsdata[3]*normval
            calibratedlumierr=perlsdata[4]*normval
            numorbit=perlsdata[6]
            numbx=3564
            lslen=lslengthsec(numorbit,numbx)
            deliveredlumi=calibratedlumi*lslen
            bxdata=perlsdata[7]
            calibratedbxdata=None
            if bxdata:
                calibratedbxdata=([x*perbunchnormval for x in bxdata[0]],[x*perbunchnormval for x in bxdata[1]])
            intensitydata=perlsdata[8]             
            result[run][(lumilsnum,cmslsnum)]=[timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierr,calibratedbxdata,intensitydata]
    return result
                       
def lumiForRange(schema,inputRange,amodetag='PROTPHYS',beamstatus=None,beamenergy=None,beamenergyFluc=0.2,withBXInfo=False,bxAlgo='OCC1',xingMinLum=1.0e-4,withBeamInfo=False,normname=None,datatag=None):
    '''
    input:
           inputRange  {run:[cmsls]} (required)
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','HIPHYS']
           beamstatus: LS filter on beamstatus (optional)
           beamenergy: LS filter on beamenergy (optional)  beamenergy+-beamenergyFluc
           withBXInfo: get per bunch info (optional)
           bxAlgo: algoname for bx values (optional) ['OCC1','OCC2','ET']
           xingMinLum: cut on bx lumi value (optional)
           withBeamInfo: get beam intensity info (optional)
           normname: norm factor name to use (optional)
           datatag: data version
    output:
           result {run:{(lumilsnum,cmslsnum):[timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,((bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}}
           lumi unit: 1/ub
    '''
           result = {}
           normid=None
           if not normname:
               normid=dataDML.guessnormIdByContext(schema,amodetag,nominalegev)
           if not normid:
               raise ValueError('cannot find a normalization factor for the combined condition '+amodetag+' '+nominalegev)
           normval=dataDML.luminormById(schema,normid)[2]
           perbunchnormval=float(normval)/float(1000)
           c=lumiTime.lumiTime()
           for run,lslist in inputRange.items():
               if lslist is not None and len(lslist)==0:#no selected ls, do nothing for this run
                   result[run]={}
                   continue
               cmsrunsummary=dataDML.runsummary(schema,run)
               startTimeStr=cmsrunsummary[6]
               lumidataid=None
               trgdataid=None
               hltdataid=None
               (lumidataid,trgdataid,hltdataid)=dataDML.guessDataIdByRun(schema,run)
               (lumirunnum,lumidata)=dataDML.lumiLSById(schema,lumidataid,beamstatus,beamenergy,beamenergyFluc,withBXInfo,bxAlgo,withBeamInfo)
               (trgrunnum,trgdata)=dataDML.trgLSById(schema,trgdataid,withblobdata=False)
               perrunresult={}
               for lumilsnum,perlsdata in lumidata.items():
                   cmslsnum=perlsdata[0]
                   if lslist is not None and cmslsnum not in lslist:
                       continue
                   instlumi=perlsdata[1]
                   instlumierror=perlsdata[2]
                   calibratedlumi=instlumi*normval
                   calibratedlumierror=instlumierror*normval
                   beamstatus=perlsdata[4]
                   beamenergy=perlsdata[5]
                   numorbit=perlsdata[6]
                   startorbit=perlsdata[7]
                   timestamp=c.OrbitToUTCTimestamp(startTimeStr,numorbit,startorbit+numorbit,0)
                   numbx=3564
                   lslen=lslengthsec(numorbit,numbx)
                   deliveredlumi=calibratedlumi*lslen
                   recordedlumi=0.0
                   if cmslsnum!=0:                       
                       deadcount=trgdata[cmslsnum][0] ##subject to change !!
                       bitzerocount=trgdata[cmslsnum][1]
                       bitzeroprescale=trgdata[cmslsnum][2]
                       deadfrac=float(deadcount)/(float(bitzerocount)*float(bitzeroprescale))
                       if deadfrac>1.0:
                           deadfrac=0.0  #artificial correction in case of trigger wrong prescale
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
                           beam1intensitylist=CommonUtil.unpackBlobtoArray(beam1intensityblob,'f').tolist()
                           beam2intensitylist=CommonUtil.unpackBlobtoArray(beam2intensityblob,'f').tolist()
                       beamdata=(bxindexlist,b1intensitylist,b2intensitylist)
                   perrunresult[(lumilsnum,cmslsnum)]=[timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,bxdata,beamdata]
               lumidata.clear() #clean lumi memory    
               trgdata.clear()
               result[run]=perrunresult
           return result
       
def effectiveLumiForRange(schema,inputRange,hltpathname=None,hltpathpattern=None,amodetag='PROTPHYS',beamstatus=None,beamenergy=None,beamenergyFluc=0.2,withBXInfo=False,xingMinLum=1.0e-4,bxAlgo='OCC1',withBeamInfo=False,normname=None,datatag=None):
    '''
    input:
           inputRange  {run:[cmsls]} (required)
           hltpathname: selected hltpathname
           hltpathpattern: regex select hltpaths
           amodetag : accelerator mode for all the runs (optional) ['PROTPHYS','HIPHYS']
           beamstatus: LS filter on beamstatus (optional)
           beamenergy: LS filter on beamenergy (optional)  beamenergy+-beamenergyFluc
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
           normid=None
           if not normname:
               normid=dataDML.guessnormIdByContext(schema,amodetag,nominalegev)
           if not normid:
               raise ValueError('cannot find a normalization factor for the combined condition '+amodetag+' '+nominalegev)
           normval=dataDML.luminormById(schema,normid)[2]
           perbunchnormval=float(normval)/float(1000)
           c=lumiTime.lumiTime()
           for run,lslist in inputRange.items():
               if lslist is not None and len(lslist)==0:#no selected ls, do nothing for this run
                   result[run]={}
                   continue
               cmsrunsummary=dataDML.runsummary(schema,run)
               startTimeStr=cmsrunsummary[6]
               lumidataid=None
               trgdataid=None
               hltdataid=None
               (lumidataid,trgdataid,hltdataid)=dataDML.guessDataIdByRun(schema,run)
               (lumirunnum,lumidata)=dataDML.lumiLSById(schema,lumidataid,beamstatus,beamenergy,beamenergyFluc,withBXInfo,bxAlgo,withBeamInfo)
               (trgrunnum,trgdata)=dataDML.trgLSById(schema,trgdataid,withblobdata=True)
               (hltrunnum,hltdata)=dataDML.hltLSById(schema,hltdataid)
               trgrundata=dataDML.trgRunById(schema,trgdataid)
               hltrundata=dataDML.hltRunById(schema,hltdataid)
               bitnames=trgrundata[3].split(',')
               hlttrgmap=dataDML.hlttrgMappingByrun(schema,run)
               pathnames=hltrundata[3].split(',')
               perrunresult={}
               for lumilsnum,perlsdata in lumidata.items():
                   cmslsnum=perlsdata[0]
                   if lslist is not None and cmslsnum not in lslist:
                       continue
                   instlumi=perlsdata[1]
                   instlumierror=perlsdata[2]
                   calibratedlumi=instlumi*normval
                   calibratedlumierror=instlumierror*normval
                   beamstatus=perlsdata[4]
                   beamenergy=perlsdata[5]
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
                       deadfrac=0.0  #artificial correction in case of trigger wrong prescale
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
                   perrunresult[(lumilsnum,cmslsnum)]=[timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,efflumidict,bxdata,beamdata]
               lumidata.clear() #clean lumi memory    
               trgdata.clear()
               hltdata.clear()
               result[run]=perrunresult
           return result
##===printers
    


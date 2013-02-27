import os,coral,fnmatch,time
from RecoLuminosity.LumiDB import nameDealer,dbUtil,revisionDML,lumiTime,CommonUtil,lumiCorrections
import array

########################################################################
# LumiDB DML                           API                             #
#                                                                      #
# Author:      Zhen Xie                                                #
########################################################################

#==============================
# SELECT
#==============================
def guesscorrIdByName(schema,tagname=None):
    '''
    select data_id from lumicorrectionss [where entry_name=:tagname]
    result lumicorrectionsdataid
    
    '''
    lumicorrectionids=[]
    result=None
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.lumicorrectionsTableName() )
        qHandle.addToOutputList('DATA_ID')
        if tagname:
            qConditionStr='ENTRY_NAME=:tagname '
            qCondition=coral.AttributeList()
            qCondition.extend('tagname','string')
            qCondition['tagname'].setData(tagname)
        qResult=coral.AttributeList()
        qResult.extend('DATA_ID','unsigned long long')
        qHandle.defineOutput(qResult)
        if tagname:
            qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            dataid=cursor.currentRow()['DATA_ID'].data()
            lumicorrectionids.append(dataid)
    except :
        del qHandle
        raise
    del qHandle
    if len(lumicorrectionids) !=0:return max(lumicorrectionids)
    return result

def lumicorrById(schema,correctiondataid):
    '''
    select entry_name,a1,a2,drift from lumicorrections where DATA_ID=:dataid
    output: {tagname:(data_id(0),a1(1),a2(2),driftcoeff(3))}
    '''
    result=None
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumicorrectionsTableName())
        qHandle.addToOutputList('ENTRY_NAME')
        qHandle.addToOutputList('A1')
        qHandle.addToOutputList('A2')
        qHandle.addToOutputList('DRIFT')
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(correctiondataid)
        qResult=coral.AttributeList()
        qResult.extend('ENTRY_NAME','string')
        qResult.extend('A1','float')
        qResult.extend('A2','float')
        qResult.extend('DRIFT','float')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            tagname=cursor.currentRow()['ENTRY_NAME'].data()
            a1=cursor.currentRow()['A1'].data()
            a2=0.0
            if cursor.currentRow()['A2'].data():
                a2=cursor.currentRow()['A2'].data()
            drift=0.0
            if cursor.currentRow()['DRIFT'].data():
                drift=cursor.currentRow()['DRIFT'].data()
            result={tagname:(correctiondataid,a1,a2,drift)}
    except :
        del qHandle
        raise
    del qHandle
    return result

def fillInRange(schema,fillmin,fillmax,amodetag,startT,stopT):
    '''
    select fillnum,runnum,starttime from cmsrunsummary where [where fillnum>=:fillmin and fillnum<=:fillmax and amodetag=:amodetag]
    output: [fill]
    '''
    result=[]
    tmpresult={}
    qHandle=schema.newQuery()
    r=nameDealer.cmsrunsummaryTableName()
    lute=lumiTime.lumiTime()
    try:
        qHandle.addToTableList(r)
        qConditionPieces=[]
        qConditionStr=''
        qCondition=coral.AttributeList()
        if fillmin:
            qConditionPieces.append('FILLNUM>=:fillmin')
            qCondition.extend('fillmin','unsigned int')
            qCondition['fillmin'].setData(int(fillmin))
        if fillmax:
            qConditionPieces.append('FILLNUM<=:fillmax')
            qCondition.extend('fillmax','unsigned int')
            qCondition['fillmax'].setData(int(fillmax))
        if amodetag:
            qConditionPieces.append('AMODETAG=:amodetag')
            qCondition.extend('amodetag','string')
            qCondition['amodetag'].setData(amodetag)
        if len(qConditionPieces)!=0:
            qConditionStr=(' AND ').join(qConditionPieces)
        qResult=coral.AttributeList()
        qResult.extend('fillnum','unsigned int')
        qResult.extend('runnum','unsigned int')
        qResult.extend('starttime','string')
        qHandle.defineOutput(qResult)
        if len(qConditionStr)!=0:
            qHandle.setCondition(qConditionStr,qCondition)
        qHandle.addToOutputList('FILLNUM','fillnum')
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('TO_CHAR('+r+'.STARTTIME,\'MM/DD/YY HH24:MI:SS\')','starttime')
        cursor=qHandle.execute()
        while cursor.next():
            currentfill=cursor.currentRow()['fillnum'].data()
            runnum=cursor.currentRow()['runnum'].data()
            starttimeStr=cursor.currentRow()['starttime'].data()
            runTime=lute.StrToDatetime(starttimeStr,customfm='%m/%d/%y %H:%M:%S')
            minTime=None
            maxTime=None
            if startT and stopT:
                minTime=lute.StrToDatetime(startT,customfm='%m/%d/%y %H:%M:%S')
                maxTime=lute.StrToDatetime(stopT,customfm='%m/%d/%y %H:%M:%S')                
                if runTime>=minTime and runTime<=maxTime:
                    tmpresult.setdefault(currentfill,[]).append(runnum)
            elif startT is not None:
                minTime=lute.StrToDatetime(startT,customfm='%m/%d/%y %H:%M:%S')
                if runTime>=minTime:
                    tmpresult.setdefault(currentfill,[]).append(runnum)
            elif stopT is not None:
                maxTime=lute.StrToDatetime(stopT,customfm='%m/%d/%y %H:%M:%S')
                if runTime<=maxTime:
                    tmpresult.setdefault(currentfill,[]).append(runnum)
            else:                
                tmpresult.setdefault(currentfill,[]).append(runnum)
        #print tmpresult
        for f in sorted(tmpresult):
            if tmpresult[f]:
                result.append(f)
    except :
        del qHandle
        raise
    del qHandle
    return result    
def fillrunMap(schema,fillnum=None,runmin=None,runmax=None,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=None):
    '''
    select fillnum,runnum,starttime from cmsrunsummary [where fillnum=:fillnum and runnum>=runmin and runnum<=runmax and amodetag=:amodetag ]
    output: {fill:[runnum,...]}
    '''
    result={}
    timelesslist=[]
    qHandle=schema.newQuery()
    r=nameDealer.cmsrunsummaryTableName()
    lute=lumiTime.lumiTime()
    try:
        qHandle.addToTableList(r)
        qConditionPieces=[]
        qConditionStr=''
        qCondition=coral.AttributeList()        
        if fillnum:
            qConditionPieces.append('FILLNUM=:fillnum')
            qCondition.extend('fillnum','unsigned int')
            qCondition['fillnum'].setData(int(fillnum))
        if runmin:
            qConditionPieces.append('RUNNUM>=:runmin')
            qCondition.extend('runmin','unsigned int')
            qCondition['runmin'].setData(runmin)
        if runmax:
            qConditionPieces.append('RUNNUM<=:runmax')
            qCondition.extend('runmax','unsigned int')
            qCondition['runmax'].setData(runmax)
        if amodetag:
            qConditionPieces.append('AMODETAG=:amodetag')
            qCondition.extend('amodetag','string')
            qCondition['amodetag'].setData(amodetag)
        if l1keyPattern:
            qConditionPieces.append('regexp_like(L1KEY,:l1keypattern)')
            qCondition.extend('l1keypattern','string')
            qCondition['l1keypattern'].setData(l1keyPattern)
        if hltkeyPattern:
            qConditionPieces.append('regexp_like(HLTKEY,:hltkeypattern)')
            qCondition.extend('hltkeypattern','string')
            qCondition['hltkeypattern'].setData(hltkeyPattern)
        if len(qConditionPieces)!=0:
            qConditionStr=(' AND ').join(qConditionPieces)        
        qResult=coral.AttributeList()
        qResult.extend('fillnum','unsigned int')
        qResult.extend('runnum','unsigned int')
        qResult.extend('starttime','string')
        qHandle.defineOutput(qResult)
        if len(qConditionStr) !=0:
            qHandle.setCondition(qConditionStr,qCondition)
        qHandle.addToOutputList('FILLNUM','fillnum')    
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('TO_CHAR('+r+'.STARTTIME,\'MM/DD/YY HH24:MI:SS\')','starttime')
        cursor=qHandle.execute()        
        while cursor.next():
            currentfill=cursor.currentRow()['fillnum'].data()
            starttimeStr=cursor.currentRow()['starttime'].data()
            runnum=cursor.currentRow()['runnum'].data()
            runTime=lute.StrToDatetime(starttimeStr,customfm='%m/%d/%y %H:%M:%S')
            minTime=None
            maxTime=None
            if startT and stopT:
                minTime=lute.StrToDatetime(startT,customfm='%m/%d/%y %H:%M:%S')
                maxTime=lute.StrToDatetime(stopT,customfm='%m/%d/%y %H:%M:%S')                
                if runTime>=minTime and runTime<=maxTime:
                    result.setdefault(currentfill,[]).append(runnum)
            elif startT is not None:
                minTime=lute.StrToDatetime(startT,customfm='%m/%d/%y %H:%M:%S')
                if runTime>=minTime:
                    result.setdefault(currentfill,[]).append(runnum)
            elif stopT is not None:
                maxTime=lute.StrToDatetime(stopT,customfm='%m/%d/%y %H:%M:%S')
                if runTime<=maxTime:
                    result.setdefault(currentfill,[]).append(runnum)
            else:                
                result.setdefault(currentfill,[]).append(runnum)
    except :
        del qHandle
        raise
    del qHandle
    return result
    
def runList(schema,fillnum=None,runmin=None,runmax=None,fillmin=None,fillmax=None,startT=None,stopT=None,l1keyPattern=None,hltkeyPattern=None,amodetag=None,nominalEnergy=None,energyFlut=0.2,requiretrg=True,requirehlt=True,lumitype=None):
    '''
    select runnum,starttime from cmsrunsummary r,lumidata l,trgdata t,hltdata h where r.runnum=l.runnum and l.runnum=t.runnum and t.runnum=h.runnum and r.fillnum=:fillnum and r.runnum>:runmin and r.runnum<:runmax and r.amodetag=:amodetag and regexp_like(r.l1key,:l1keypattern) and regexp_like(hltkey,:hltkeypattern) and l.nominalEnergy>=:nominalEnergy*(1-energyFlut) and l.nominalEnergy<=:nominalEnergy*(1+energyFlut)
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    lumitableName=''
    if lumitype=='HF':
        lumitableName=nameDealer.lumidataTableName()
    elif lumitype == 'PIXEL':
        lumitableName = nameDealer.pixellumidataTableName()
    else:
        assert False, "ERROR Unknown lumitype '%s'" % lumitype
    result=[]
    timelesslist=[]
    qHandle=schema.newQuery()
    r=nameDealer.cmsrunsummaryTableName()
    l=lumitableName
    t=nameDealer.trgdataTableName()
    h=nameDealer.hltdataTableName()
    lute=lumiTime.lumiTime()
    try:
        qHandle.addToTableList(r)
        qHandle.addToTableList(l)
        qConditionStr=r+'.RUNNUM='+l+'.RUNNUM'
        if requiretrg:
            qHandle.addToTableList(t)
            qConditionStr+=' and '+l+'.RUNNUM='+t+'.RUNNUM'
        if requirehlt:
            qHandle.addToTableList(h)
            qConditionStr+=' and '+l+'.RUNNUM='+h+'.RUNNUM'
        qCondition=coral.AttributeList()        
        if fillnum:
            qConditionStr+=' and '+r+'.FILLNUM=:fillnum'
            qCondition.extend('fillnum','unsigned int')
            qCondition['fillnum'].setData(int(fillnum))
        if runmin:
            qConditionStr+=' and '+r+'.RUNNUM>=:runmin'
            qCondition.extend('runmin','unsigned int')
            qCondition['runmin'].setData(runmin)
        if runmax:
            qConditionStr+=' and '+r+'.RUNNUM<=:runmax'
            qCondition.extend('runmax','unsigned int')
            qCondition['runmax'].setData(runmax)
        if fillmin:
            qConditionStr+=' and '+r+'.FILLNUM>=:fillmin'
            qCondition.extend('fillmin','unsigned int')
            qCondition['fillmin'].setData(fillmin)
        if fillmax:
            qConditionStr+=' and '+r+'.FILLNUM<=:fillmax'
            qCondition.extend('fillmax','unsigned int')
            qCondition['fillmax'].setData(fillmax)
        if amodetag:
            qConditionStr+=' and '+r+'.AMODETAG=:amodetag'
            qCondition.extend('amodetag','string')
            qCondition['amodetag'].setData(amodetag)
        if l1keyPattern:
            qConditionStr+=' and regexp_like('+r+'.L1KEY,:l1keypattern)'
            qCondition.extend('l1keypattern','string')
            qCondition['l1keypattern'].setData(l1keyPattern)
        if hltkeyPattern:
            qConditionStr+=' and regexp_like('+r+'.HLTKEY,:hltkeypattern)'
            qCondition.extend('hltkeypattern','string')
            qCondition['hltkeypattern'].setData(hltkeyPattern)
        if nominalEnergy:
            emin=nominalEnergy*(1.0-energyFlut)
            emax=nominalEnergy*(1.0+energyFlut)
            qConditionStr+=' and '+l+'.NOMINALEGEV>=:emin and '+l+'.NOMINALEGEV<=:emax'
            qCondition.extend('emin','float')
            qCondition.extend('emax','float')
            qCondition['emin'].setData(emin)
            qCondition['emax'].setData(emax)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('starttime','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        qHandle.addToOutputList(r+'.RUNNUM','runnum')
        qHandle.addToOutputList('TO_CHAR('+r+'.STARTTIME,\'MM/DD/YY HH24:MI:SS\')','starttime')
        cursor=qHandle.execute()
        
        while cursor.next():
            starttimeStr=cursor.currentRow()['starttime'].data()
            runnum=cursor.currentRow()['runnum'].data()
            minTime=None
            maxTime=None
            if startT and stopT:
                minTime=lute.StrToDatetime(startT,customfm='%m/%d/%y %H:%M:%S')
                maxTime=lute.StrToDatetime(stopT,customfm='%m/%d/%y %H:%M:%S')
                runTime=lute.StrToDatetime(starttimeStr,customfm='%m/%d/%y %H:%M:%S')
                if runTime>=minTime and runTime<=maxTime and runnum not in result:
                    result.append(runnum)
            elif startT is not None:
                minTime=lute.StrToDatetime(startT,customfm='%m/%d/%y %H:%M:%S')
                runTime=lute.StrToDatetime(starttimeStr,customfm='%m/%d/%y %H:%M:%S')
                if runTime>=minTime and runnum not in result:
                    result.append(runnum)
            elif stopT is not None:
                maxTime=lute.StrToDatetime(stopT,customfm='%m/%d/%y %H:%M:%S')
                runTime=lute.StrToDatetime(starttimeStr,customfm='%m/%d/%y %H:%M:%S')
                if runTime<=maxTime and runnum not in result:
                    result.append(runnum)
            else:
                if runnum not in result:
                    result.append(runnum)
    except :
        del qHandle
        raise
    del qHandle
    return result

def runsummary(schema,runnum,sessionflavor=''):
    '''
    select l1key,amodetag,egev,hltkey,fillnum,fillscheme,to_char(starttime),to_char(stoptime) from cmsrunsummary where runnum=:runnum
    output: [l1key(0),amodetag(1),egev(2),hltkey(3),fillnum(4),fillscheme(5),starttime(6),stoptime(7)]
    '''
    result=[]
    qHandle=schema.newQuery()
    t=lumiTime.lumiTime()
    try:
        qHandle.addToTableList(nameDealer.cmsrunsummaryTableName())
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qHandle.addToOutputList('L1KEY','l1key')
        qHandle.addToOutputList('AMODETAG','amodetag')
        qHandle.addToOutputList('EGEV','egev')
        qHandle.addToOutputList('HLTKEY','hltkey')
        qHandle.addToOutputList('FILLNUM','fillnum')
        qHandle.addToOutputList('FILLSCHEME','fillscheme')
        if sessionflavor=='SQLite':
            qHandle.addToOutputList('STARTTIME','starttime')
            qHandle.addToOutputList('STOPTIME','stoptime')
        else:
            qHandle.addToOutputList('to_char(STARTTIME,\''+t.coraltimefm+'\')','starttime')
            qHandle.addToOutputList('to_char(STOPTIME,\''+t.coraltimefm+'\')','stoptime')
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qResult=coral.AttributeList()
        qResult.extend('l1key','string')
        qResult.extend('amodetag','string')
        qResult.extend('egev','unsigned int')
        qResult.extend('hltkey','string')
        qResult.extend('fillnum','unsigned int')
        qResult.extend('fillscheme','string')
        qResult.extend('starttime','string')
        qResult.extend('stoptime','string')
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            result.append(cursor.currentRow()['l1key'].data())
            result.append(cursor.currentRow()['amodetag'].data())
            result.append(cursor.currentRow()['egev'].data())
            result.append(cursor.currentRow()['hltkey'].data())
            result.append(cursor.currentRow()['fillnum'].data())
            fillscheme=''
            if not cursor.currentRow()['fillscheme'].isNull():
                fillscheme=cursor.currentRow()['fillscheme'].data()
            result.append(fillscheme)
            result.append(cursor.currentRow()['starttime'].data())
            result.append(cursor.currentRow()['stoptime'].data())
    except :
        del qHandle
        raise
    del qHandle
    return result

def mostRecentLuminorms(schema,branchfilter):
    '''
    this overview query should be only for norm
    select e.name,n.data_id,r.revision_id,n.amodetag,n.norm_1,n.egev_1,n.norm_occ2,n.norm_et,n.norm_pu,n.constfactor from luminorms_entries e,luminorms_rev r,luminorms n where n.entry_id=e.entry_id and n.data_id=r.data_id and r.revision_id>=min(branchfilter) and r.revision_id<=max(branchfilter);
    output {norm_name:(amodetag(0),norm_1(1),egev_1(2),norm_occ2(3),norm_et(4),norm_pu(5),constfactor(6))}
    '''
    #print branchfilter
    result={}
    entry2datamap={}
    branchmin=0
    branchmax=0
    if branchfilter and len(branchfilter)!=0:
        branchmin=min(branchfilter)
        branchmax=max(branchfilter)
    else:
        return result
    #print branchmin,branchmax
    qHandle=schema.newQuery()
    normdict={}
    try:
        qHandle.addToTableList(nameDealer.entryTableName(nameDealer.luminormTableName()),'e')
        qHandle.addToTableList(nameDealer.luminormTableName(),'n')
        qHandle.addToTableList(nameDealer.revmapTableName(nameDealer.luminormTableName()),'r')
        qHandle.addToOutputList('e.NAME','normname')
        qHandle.addToOutputList('r.DATA_ID','data_id')
        qHandle.addToOutputList('r.REVISION_ID','revision_id')
        qHandle.addToOutputList('n.AMODETAG','amodetag')
        qHandle.addToOutputList('n.NORM_1','norm_1')
        qHandle.addToOutputList('n.EGEV_1','energy_1')
        qHandle.addToOutputList('n.NORM_OCC2','norm_occ2')
        qHandle.addToOutputList('n.NORM_ET','norm_et')
        qHandle.addToOutputList('n.NORM_PU','norm_pu')
        qHandle.addToOutputList('n.CONSTFACTOR','constfactor')
        qCondition=coral.AttributeList()
        qCondition.extend('branchmin','unsigned long long')
        qCondition.extend('branchmax','unsigned long long')
        qCondition['branchmin'].setData(branchmin)
        qCondition['branchmax'].setData(branchmax)
        qResult=coral.AttributeList()
        qResult.extend('normname','string')
        qResult.extend('data_id','unsigned long long')
        qResult.extend('revision_id','unsigned long long')
        qResult.extend('amodetag','string')
        qResult.extend('norm_1','float')
        qResult.extend('energy_1','unsigned int')
        qResult.extend('norm_occ2','float')
        qResult.extend('norm_et','float')
        qResult.extend('norm_pu','float')
        qResult.extend('constfactor','float')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('n.ENTRY_ID=e.ENTRY_ID and n.DATA_ID=r.DATA_ID AND n.DATA_ID=r.DATA_ID AND r.REVISION_ID>=:branchmin AND r.REVISION_ID<=:branchmax',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            data_id=cursor.currentRow()['data_id'].data()
            normname=cursor.currentRow()['normname'].data()
            if not normdict.has_key(normname):
                normdict[normname]=0
            if data_id>normdict[normname]:
                normdict[normname]=data_id
                amodetag=cursor.currentRow()['amodetag'].data()
                norm_1=cursor.currentRow()['norm_1'].data()
                energy_1=cursor.currentRow()['energy_1'].data()
                norm_occ2=1.0
                if not cursor.currentRow()['norm_occ2'].isNull():
                    norm_occ2=cursor.currentRow()['norm_occ2'].data()
                norm_et=1.0
                if not cursor.currentRow()['norm_et'].isNull():
                    norm_et=cursor.currentRow()['norm_et'].data()
                norm_pu=1.0
                if not cursor.currentRow()['norm_pu'].isNull():
                    norm_pu=cursor.currentRow()['norm_pu'].data()
                constfactor=1.0
                if not cursor.currentRow()['constfactor'].isNull():
                    constfactor=cursor.currentRow()['constfactor'].data()
                result[normname]=(amodetag,norm_1,energy_1,norm_occ2,norm_et,norm_pu,constfactor)
    except:
        raise
    return result
def luminormById(schema,dataid):
    '''
    select entry_name,amodetag,norm_1,egev_1,norm_2,egev_2 from luminorms where DATA_ID=:dataid
    output: {norm_name:(amodetag(0),norm_1(1),egev_1(2),norm_occ2(3),norm_et(4),norm_pu(5),constfactor(6))}
    '''
    result=None
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.luminormTableName())
        qHandle.addToOutputList('ENTRY_NAME','normname')
        qHandle.addToOutputList('AMODETAG','amodetag')
        qHandle.addToOutputList('NORM_1','norm_1')
        qHandle.addToOutputList('EGEV_1','energy_1')
        qHandle.addToOutputList('NORM_OCC2','norm_occ2')
        qHandle.addToOutputList('NORM_ET','norm_et')
        qHandle.addToOutputList('NORM_PU','norm_pu')
        qHandle.addToOutputList('CONSTFACTOR','constfactor')        
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('normname','string')
        qResult.extend('amodetag','string')
        qResult.extend('norm_1','float')
        qResult.extend('energy_1','unsigned int')
        qResult.extend('norm_occ2','float')
        qResult.extend('norm_et','float')
        qResult.extend('norm_pu','float')
        qResult.extend('constfactor','float')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normname=cursor.currentRow()['normname'].data()
            amodetag=cursor.currentRow()['amodetag'].data()
            norm_1=cursor.currentRow()['norm_1'].data()
            energy_1=cursor.currentRow()['energy_1'].data()
            norm_occ2=1.0
            if cursor.currentRow()['norm_occ2'].data():
                norm_occ2=cursor.currentRow()['norm_occ2'].data()
            norm_et=1.0
            if cursor.currentRow()['norm_et'].data():
                norm_et=cursor.currentRow()['norm_et'].data()
            norm_pu=1.0
            if cursor.currentRow()['norm_pu'].data():
                norm_pu=cursor.currentRow()['norm_pu'].data()
            constfactor=1.0
            if cursor.currentRow()['constfactor'].data():
                constfactor=cursor.currentRow()['constfactor'].data()
            result={normname:(amodetag,norm_1,energy_1,norm_occ2,norm_et,norm_pu,constfactor)}
    except :
        del qHandle
        raise
    del qHandle
    return result

def mostRecentLumicorrs(schema,branchfilter):
    '''
    this overview query should be only for corr
    select e.name,n.data_id,r.revision_id , n.a1,n.a2,n.drift from lumicorrections_entries e,lumicorrections_rev r,lumicorrections n where n.entry_id=e.entry_id and n.data_id=r.data_id and r.revision_id>=min(branchfilter) and r.revision_id<=max(branchfilter) group by e.entry_name,r.revision_id,n.a1,n.a2,n.drift;
    output {corrname:(data_id,a1,a2,drift)}
    '''
    #print branchfilter
    result={}
    entry2datamap={}
    branchmin=0
    branchmax=0
    if branchfilter and len(branchfilter)!=0:
        branchmin=min(branchfilter)
        branchmax=max(branchfilter)
    else:
        return result
    qHandle=schema.newQuery()
    corrdict={}
    try:
        qHandle.addToTableList(nameDealer.entryTableName(nameDealer.lumicorrectionsTableName()),'e')
        qHandle.addToTableList(nameDealer.lumicorrectionsTableName(),'n')
        qHandle.addToTableList(nameDealer.revmapTableName(nameDealer.lumicorrectionsTableName()),'r')
        qHandle.addToOutputList('e.NAME','corrname')
        qHandle.addToOutputList('r.DATA_ID','data_id')
        qHandle.addToOutputList('r.REVISION_ID','revision_id')
        qHandle.addToOutputList('n.A1','a1')
        qHandle.addToOutputList('n.A2','a2')
        qHandle.addToOutputList('n.DRIFT','drift')
        qCondition=coral.AttributeList()
        qCondition.extend('branchmin','unsigned long long')
        qCondition.extend('branchmax','unsigned long long')
        qCondition['branchmin'].setData(branchmin)
        qCondition['branchmax'].setData(branchmax)
        qResult=coral.AttributeList()
        qResult.extend('corrname','string')
        qResult.extend('data_id','unsigned long long')
        qResult.extend('revision_id','unsigned long long')
        qResult.extend('a1','float')
        qResult.extend('a2','float')
        qResult.extend('drift','float')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('n.ENTRY_ID=e.ENTRY_ID and n.DATA_ID=r.DATA_ID AND n.DATA_ID=r.DATA_ID AND r.REVISION_ID>=:branchmin AND r.REVISION_ID<=:branchmax',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            corrname=cursor.currentRow()['corrname'].data()
            data_id=cursor.currentRow()['data_id'].data()
            if not corrdict.has_key(corrname):
                corrdict[corrname]=0
            if data_id>corrdict[corrname]:
                corrdict[corrname]=data_id
                a1=cursor.currentRow()['a1'].data() #required
                a2=0.0
                if not cursor.currentRow()['a2'].isNull():
                    a2=cursor.currentRow()['a2'].data()
                drift=0.0
                if not cursor.currentRow()['drift'].isNull():
                    drift=cursor.currentRow()['drift'].data()
                result[corrname]=(data_id,a1,a2,drift)
    except:
        raise
    return result

def luminormById(schema,dataid):
    '''
    select entry_name,amodetag,norm_1,egev_1,norm_2,egev_2 from luminorms where DATA_ID=:dataid
    result (normname(0),amodetag(1),egev(2),norm(3),norm_occ2(4),norm_et(5),norm_pu(6),constfactor(7))
    '''
    result=None
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.luminormTableName())
        qHandle.addToOutputList('ENTRY_NAME','normname')
        qHandle.addToOutputList('AMODETAG','amodetag')
        qHandle.addToOutputList('NORM_1','norm_1')
        qHandle.addToOutputList('EGEV_1','energy_1')
        qHandle.addToOutputList('NORM_OCC2','norm_occ2')
        qHandle.addToOutputList('NORM_ET','norm_et')
        qHandle.addToOutputList('NORM_PU','norm_pu')
        qHandle.addToOutputList('CONSTFACTOR','constfactor')
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('normname','string')
        qResult.extend('amodetag','string')
        qResult.extend('norm_1','float')
        qResult.extend('energy_1','unsigned int')
        qResult.extend('norm_occ2','float')
        qResult.extend('norm_et','float')
        qResult.extend('norm_pu','float')
        qResult.extend('constfactor','float')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normname=cursor.currentRow()['normname'].data()
            amodetag=cursor.currentRow()['amodetag'].data()
            norm_1=cursor.currentRow()['norm_1'].data()
            energy_1=cursor.currentRow()['energy_1'].data()
            norm_occ2=1.0
            if cursor.currentRow()['norm_occ2'].data():
                norm_occ2=cursor.currentRow()['norm_occ2'].data()
            norm_et=1.0
            if cursor.currentRow()['norm_et'].data():
                norm_et=cursor.currentRow()['norm_et'].data()
            norm_pu=1.0
            if cursor.currentRow()['norm_pu'].data():
                norm_pu=cursor.currentRow()['norm_pu'].data()
            constfactor=1.0
            if cursor.currentRow()['constfactor'].data():
                constfactor=cursor.currentRow()['constfactor'].data()
            result={normname:(amodetag,norm_1,energy_1,norm_occ2,norm_et,norm_pu,constfactor)}
    except :
        del qHandle
        raise
    del qHandle
    return result

def trgRunById(schema,dataid,trgbitname=None,trgbitnamepattern=None):
    '''
    query: select RUNNUM,SOURCE,BITZERONAME,BITNAMECLOB from trgdata where DATA_ID=:dataid
    
    output: [runnum(0),datasource(1),bitzeroname(2),bitnamedict(3)]
             -- runnumber
             -- original source database name
             -- deadtime norm bitname
             -- bitnamedict [(bitidx,bitname),...]
    '''
    result=[]
    qHandle=schema.newQuery()
    runnum=None
    datasource=None
    bitzeroname=None
    bitnamedict=[]
    try:
        qHandle.addToTableList(nameDealer.trgdataTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('SOURCE','source')
        qHandle.addToOutputList('BITZERONAME','bitzeroname')
        qHandle.addToOutputList('BITNAMECLOB','bitnameclob')
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('source','string')
        qResult.extend('bitzeroname','string')
        qResult.extend('bitnameclob','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('DATA_ID=:dataid',qCondition)        
        cursor=qHandle.execute()
        bitnameclob=None
        bitnames=[]
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            source=cursor.currentRow()['source'].data()
            bitzeroname=cursor.currentRow()['bitzeroname'].data()
            bitnameclob=cursor.currentRow()['bitnameclob'].data()
        if bitnameclob:
            bitnames=bitnameclob.split(',')
            for trgnameidx,trgname in enumerate(bitnames):
                if trgbitname :
                    if trgname==trgbitname:
                        bitnamedict.append((trgnameidx,trgname))
                        break
                elif trgbitnamepattern:
                    if fnmatch.fnmatch(trgname,trgbitnamepattern):
                        bitnamedict.append((trgnameidx,trgname))
                else:
                    bitnamedict.append((trgnameidx,trgname))
        result=[runnum,source,bitzeroname,bitnamedict]
    except :
        del qHandle
        raise 
    del qHandle
    return result

def trgLSById(schema,dataid,trgbitname=None,trgbitnamepattern=None,withL1Count=False,withPrescale=False):
    '''
    output: (runnum,{cmslsnum:[deadtimecount(0),bitzerocount(1),bitzeroprescale(2),deadfrac(3),[(bitname,trgcount,prescale)](4)]})
    '''
    runnum=0
    result={}
    trgnamedict=[]
    if  trgbitname or trgbitnamepattern or withPrescale or withL1Count:
        trgrundata=trgRunById(schema,dataid,trgbitname=trgbitname,trgbitnamepattern=trgbitnamepattern)
        trgnamedict=trgrundata[3]

    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lstrgTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('DEADTIMECOUNT','deadtimecount')
        #qHandle.addToOutputList('BITZEROCOUNT','bitzerocount')
        #qHandle.addToOutputList('BITZEROPRESCALE','bitzeroprescale')
        qHandle.addToOutputList('DEADFRAC','deadfrac')
        if withPrescale:
            qHandle.addToOutputList('PRESCALEBLOB','prescalesblob')
        if withL1Count:
            qHandle.addToOutputList('TRGCOUNTBLOB','trgcountblob')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('deadtimecount','unsigned long long')
        #qResult.extend('bitzerocount','unsigned int')
        #qResult.extend('bitzeroprescale','unsigned int')
        qResult.extend('deadfrac','float')
        if withPrescale:
            qResult.extend('prescalesblob','blob')
        if withL1Count:
            qResult.extend('trgcountblob','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            deadtimecount=cursor.currentRow()['deadtimecount'].data()
            #bitzerocount=cursor.currentRow()['bitzerocount'].data()
            #bitzeroprescale=cursor.currentRow()['bitzeroprescale'].data()
            bitzerocount=0
            bitzeroprescale=0
            deadfrac=cursor.currentRow()['deadfrac'].data()
            if not result.has_key(cmslsnum):
                result[cmslsnum]=[]
            result[cmslsnum].append(deadtimecount)
            result[cmslsnum].append(bitzerocount)
            result[cmslsnum].append(bitzeroprescale)
            result[cmslsnum].append(deadfrac)
            prescalesblob=None
            trgcountblob=None
            if withPrescale:
                prescalesblob=cursor.currentRow()['prescalesblob'].data()
            if withL1Count:
                trgcountblob=cursor.currentRow()['trgcountblob'].data()
            prescales=[]
            trgcounts=[]
            if prescalesblob:
                if runnum <150008: ###WORKAROUND PATCH!! because the 2010 blobs were packed as type l ###
                    prescales=CommonUtil.unpackBlobtoArray(prescalesblob,'l')
                else:
                    prescales=CommonUtil.unpackBlobtoArray(prescalesblob,'I')
            if trgcountblob:
                if runnum <150008: ###WORKAROUND PATCH!! because the 2010 blobs were packed as type l ###
                    trgcounts=CommonUtil.unpackBlobtoArray(trgcountblob,'l')
                else:
                    trgcounts=CommonUtil.unpackBlobtoArray(trgcountblob,'I')
                    
            bitinfo=[]
            for (bitidx,thisbitname) in trgnamedict:
                thispresc=None
                thistrgcount=None
                if prescales:
                    thispresc=prescales[bitidx]
                if trgcounts:
                    thistrgcount=trgcounts[bitidx]
                thisbitinfo=(thisbitname,thistrgcount,thispresc)
                bitinfo.append(thisbitinfo)
            result[cmslsnum].append(bitinfo)
    except:
        del qHandle
        raise 
    del qHandle
#    t1=time.time()
#    print 'trgLSById time ',t1-t0
    return (runnum,result)

def lumiRunByIds(schema,dataidMap,lumitype='HF'):
    '''
    input dataidMap : {run:lumidataid}
    result {runnum: (datasource(0),nominalegev(1),ncollidingbunches(2)}
    '''
    result={}
    if not dataidMap:
        return result
    inputRange=dataidMap.keys()
    for r in inputRange:
        lumidataid=dataidMap[r][0]
        if lumidataid:
            perrundata=lumiRunById(schema,lumidataid,lumitype=lumitype)
            result[r]=(perrundata[1],perrundata[2],perrundata[3])
    return result

def lumiRunById(schema,lumidataid,lumitype='HF'):
    '''
    input: lumidataid
    output: (runnum(0),datasource(1),nominalegev(2),ncollidingbunches(3))
    '''
    if lumitype not in ['HF','PIXEL']:
        raise ValueError('unknown lumitype '+lumitype)
    lumitableName=''
    if lumitype=='HF':
        lumitableName = nameDealer.lumidataTableName()
    else:
        lumitableName = nameDealer.pixellumidataTableName()
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(lumitableName)
        qHandle.addToOutputList('RUNNUM')
        qHandle.addToOutputList('SOURCE')
        qHandle.addToOutputList('NOMINALEGEV')
        qHandle.addToOutputList('NCOLLIDINGBUNCHES')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(lumidataid)
        qResult=coral.AttributeList()
        qResult.extend('RUNNUM','unsigned int')
        qResult.extend('SOURCE','string')
        qResult.extend('NOMINALEGEV','float')
        qResult.extend('NCOLLIDINGBUNCHES','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['RUNNUM'].data()
            datasource=cursor.currentRow()['SOURCE'].data()
            nominalegev=0
            if not cursor.currentRow()['NOMINALEGEV'].isNull():
                nominalegev=cursor.currentRow()['NOMINALEGEV'].data()
            ncollidingbunches=0
            if not cursor.currentRow()['NCOLLIDINGBUNCHES'].isNull():
                ncollidingbunches=cursor.currentRow()['NCOLLIDINGBUNCHES'].data()
            result=(runnum,datasource,nominalegev,ncollidingbunches)
    except :
        del qHandle
        raise
    del qHandle
    return result

def correctionByName(schema,correctiontagname=None):
    '''
    get correction coefficients by name
    input: correctiontagname if None,get current default
    output: [tagname,a1,a2,drift]
    if not correctiontagname
    select entry_name,data_id,a1,a2,drift from lumicorrections where 
    else:
    select entry_name,data_id,a1,a2,drift from lumicorrections where entry_name=:correctiontagname
    '''

    
def fillschemeByRun(schema,runnum):
    fillscheme=''
    ncollidingbunches=0
    r=nameDealer.cmsrunsummaryTableName()
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(r)
        qHandle.addToOutputList('FILLSCHEME')
        qHandle.addToOutputList('NCOLLIDINGBUNCHES')
        qResult=coral.AttributeList()
        qResult.extend('FILLSCHEME','string')
        qResult.extend('NCOLLIDINGBUNCHES','unsigned int')
        qConditionStr='RUNNUM=:runnum'
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnum))
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next(): 
            if not cursor.currentRow()['NCOLLIDINGBUNCHES'].isNull():
                ncollidingbunches=cursor.currentRow()['NCOLLIDINGBUNCHES'].data()
            if not cursor.currentRow()['FILLSCHEME'].isNull():
                fillscheme=cursor.currentRow()['FILLSCHEME'].data()
    except :
        del qHandle
        raise
    del qHandle
    return (fillscheme,ncollidingbunches)
def allfillschemes(schema):
    afterglows=[]
    s=nameDealer.fillschemeTableName()
    try:
        qHandle.addToTableList(s)
        qResult=coral.AttributeList()
        qResult.extend('FILLSCHEMEPATTERN','string')
        qResult.extend('CORRECTIONFACTOR','float')
        qHandle.defineOutput(qResult)
        qHandle.addToOutputList('FILLSCHEMEPATTERN')
        qHandle.addToOutputList('CORRECTIONFACTOR')
        cursor=qHandle.execute()
        while cursor.next():
            fillschemePattern=cursor.currentRow()['FILLSCHEMEPATTERN'].data()
            afterglowfac=cursor.currentRow()['CORRECTIONFACTOR'].data()
            afterglows.append((fillschemePattern,afterglowfac))
    except :
        del qHandle
        raise
    del qHandle
    return afterglows
    
def lumiLSById(schema,dataid,beamstatus=None,withBXInfo=False,bxAlgo='OCC1',withBeamIntensity=False,tableName=None):
    '''
    input:
       beamstatus: filter on beam status flag
    output:
    result (runnum,{lumilsnum,[cmslsnum(0),instlumi(1),instlumierr(2),instlumiqlty(3),beamstatus(4),beamenergy(5),numorbit(6),startorbit(7),(bxvalueArray,bxerrArray)(8),(bxindexArray,beam1intensityArray,beam2intensityArray)(9)]})
    '''
    runnum=0
    result={}
    qHandle=schema.newQuery()
    if withBXInfo and bxAlgo not in ['OCC1','OCC2','ET']:
        raise ValueError('unknown lumi algo '+bxAlgo)
    if beamstatus and beamstatus not in ['STABLE BEAMS',]:
        raise ValueError('unknown beam status '+beamstatus)
    try:
        if tableName is None:
            lls=nameDealer.lumisummaryv2TableName()
        else:
            lls=tableName
        qHandle.addToTableList(lls)
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('INSTLUMI','instlumi')
        qHandle.addToOutputList('INSTLUMIERROR','instlumierr')
        qHandle.addToOutputList('INSTLUMIQUALITY','instlumiqlty')
        qHandle.addToOutputList('BEAMSTATUS','beamstatus')
        qHandle.addToOutputList('BEAMENERGY','beamenergy')
        qHandle.addToOutputList('NUMORBIT','numorbit')
        qHandle.addToOutputList('STARTORBIT','startorbit')
        if withBXInfo:
            qHandle.addToOutputList('BXLUMIVALUE_'+bxAlgo,'bxvalue')
            qHandle.addToOutputList('BXLUMIERROR_'+bxAlgo,'bxerror')
        if withBeamIntensity:
            qHandle.addToOutputList('CMSBXINDEXBLOB','bxindexblob')
            qHandle.addToOutputList('BEAMINTENSITYBLOB_1','beam1intensity')
            qHandle.addToOutputList('BEAMINTENSITYBLOB_2','beam2intensity')
        
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(int(dataid))
        if beamstatus:
            qConditionStr+=' and BEAMSTATUS=:beamstatus'
            qCondition.extend('beamstatus','string')
            qCondition['beamstatus'].setData(beamstatus)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('instlumi','float')
        qResult.extend('instlumierr','float')
        qResult.extend('instlumiqlty','short')
        qResult.extend('beamstatus','string')
        qResult.extend('beamenergy','float')
        qResult.extend('numorbit','unsigned int')
        qResult.extend('startorbit','unsigned int')
        if withBXInfo:
            qResult.extend('bxvalue','blob')
            qResult.extend('bxerror','blob')          
        if withBeamIntensity:
            qResult.extend('bxindexblob','blob')
            qResult.extend('beam1intensity','blob')
            qResult.extend('beam2intensity','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            instlumi=cursor.currentRow()['instlumi'].data()
            instlumierr=cursor.currentRow()['instlumierr'].data()
            instlumiqlty=cursor.currentRow()['instlumiqlty'].data()
            bs=cursor.currentRow()['beamstatus'].data()
            begev=cursor.currentRow()['beamenergy'].data()
            numorbit=cursor.currentRow()['numorbit'].data()
            startorbit=cursor.currentRow()['startorbit'].data()
            bxinfo=None
            bxvalueblob=None
            bxerrblob=None
            if withBXInfo:
                bxvalueblob=cursor.currentRow()['bxvalue'].data()
                bxerrblob=cursor.currentRow()['bxerror'].data()
                if bxvalueblob and bxerrblob:
                    bxvaluesArray=CommonUtil.unpackBlobtoArray(bxvalueblob,'f')
                    bxerrArray=CommonUtil.unpackBlobtoArray(bxerrblob,'f')
                    bxinfo=(bxvaluesArray,bxerrArray)
            bxindexblob=None
            beam1intensity=None
            beam2intensity=None
            beaminfo=None
            if withBeamIntensity:
                bxindexblob=cursor.currentRow()['bxindexblob'].data()
                beam1intensity=cursor.currentRow()['beam1intensity'].data()
                beam2intensity=cursor.currentRow()['beam2intensity'].data()
                if bxindexblob :
                    bxindexArray=CommonUtil.unpackBlobtoArray(bxindexblob,'h')
                    beam1intensityArray=CommonUtil.unpackBlobtoArray(beam1intensity,'f')
                    beam2intensityArray=CommonUtil.unpackBlobtoArray(beam2intensity,'f')
                    beaminfo=(bxindexArray,beam1intensityArray,beam2intensityArray)
            result[lumilsnum]=[cmslsnum,instlumi,instlumierr,instlumiqlty,bs,begev,numorbit,startorbit,bxinfo,beaminfo]
    except :
        del qHandle
        raise 
    del qHandle
    return (runnum,result)
def beamInfoById(schema,dataid,withBeamIntensity=False,minIntensity=0.1):
    '''
    result (runnum,[(lumilsnum(0),cmslsnum(1),beamstatus(2),beamenergy(3),ncollidingbunches(4),beaminfolist(5),..])
         beaminfolist=[(bxidx,beam1intensity,beam2intensity)]
    '''
    runnum=0
    result=[]
    ncollidingbunches=0
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumidataTableName())
        qHandle.addToOutputList('NCOLLIDINGBUNCHES')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('NCOLLIDINGBUNCHES','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            ncollidingbunches=cursor.currentRow()['NCOLLIDINGBUNCHES'].data()
    except :
        del qHandle
        raise
    del qHandle
    qHandle=schema.newQuery()
    try:
       qHandle.addToTableList(nameDealer.lumisummaryv2TableName())
       qHandle.addToOutputList('RUNNUM')
       qHandle.addToOutputList('CMSLSNUM')
       qHandle.addToOutputList('LUMILSNUM')
       qHandle.addToOutputList('BEAMSTATUS')
       qHandle.addToOutputList('BEAMENERGY')
       if withBeamIntensity:
           qHandle.addToOutputList('CMSBXINDEXBLOB')
           qHandle.addToOutputList('BEAMINTENSITYBLOB_1')
           qHandle.addToOutputList('BEAMINTENSITYBLOB_2')
       qConditionStr='DATA_ID=:dataid'
       qCondition=coral.AttributeList()
       qCondition.extend('dataid','unsigned long long')
       qCondition['dataid'].setData(dataid)
       qResult=coral.AttributeList()
       qResult.extend('RUNNUM','unsigned int')
       qResult.extend('CMSLSNUM','unsigned int')
       qResult.extend('LUMILSNUM','unsigned int')
       qResult.extend('BEAMSTATUS','string')
       qResult.extend('BEAMENERGY','float')
       if withBeamIntensity:
           qResult.extend('BXINDEXBLOB','blob')
           qResult.extend('BEAM1INTENSITY','blob')
           qResult.extend('BEAM2INTENSITY','blob')
       qHandle.defineOutput(qResult)
       qHandle.setCondition(qConditionStr,qCondition)
       cursor=qHandle.execute()
       while cursor.next():
           runnum=cursor.currentRow()['RUNNUM'].data()
           cmslsnum=cursor.currentRow()['CMSLSNUM'].data()
           lumilsnum=cursor.currentRow()['LUMILSNUM'].data()
           beamstatus=cursor.currentRow()['BEAMSTATUS'].data()
           beamenergy=cursor.currentRow()['BEAMENERGY'].data()
           bxindexblob=None
           beaminfotupleList=[]
           if withBeamIntensity:
               bxindexblob=cursor.currentRow()['BXINDEXBLOB'].data()
               beam1intensityblob=cursor.currentRow()['BEAM1INTENSITY'].data()
               beam2intensityblob=cursor.currentRow()['BEAM2INTENSITY'].data()
               bxindexArray=None
               beam1intensityArray=None
               beam2intensityArray=None
               if bxindexblob:
                   bxindexArray=CommonUtil.unpackBlobtoArray(bxindexblob,'h')
               if beam1intensityblob:
                   beam1intensityArray=CommonUtil.unpackBlobtoArray(beam1intensityblob,'f')
               if beam2intensityblob:
                   beam2intensityArray=CommonUtil.unpackBlobtoArray(beam2intensityblob,'f')
               if bxindexArray and beam1intensityArray and beam2intensityArray:
                   for idx,bxindex in enumerate(bxindexArray):
                       if (beam1intensityArray[idx] and beam1intensityArray[idx]>minIntensity) or (beam2intensityArray[idx] and beam2intensityArray[idx]>minIntensity):
                           beaminfotuple=(bxindex,beam1intensityArray[idx],beam2intensityArray[idx])                   
                           beaminfotupleList.append(beaminfotuple)
                   del bxindexArray[:]
                   del beam1intensityArray[:]
                   del beam2intensityArray[:]           
           result.append((lumilsnum,cmslsnum,beamstatus,beamenergy,ncollidingbunches,beaminfotupleList))
    except:
       del qHandle
       raise
    del qHandle
    return (runnum,result)
def lumiBXByAlgo(schema,dataid,algoname):
    '''
    result {lumilsnum:[cmslsnum,numorbit,startorbit,bxlumivalue,bxlumierr,bxlumiqlty]}
    '''
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lumisummaryv2TableName())
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('LUMILSNUM','lumilsnum')
        #qHandle.addToOutputList('ALGONAME','algoname')
        qHandle.addToOutputList('NUMORBIT','numorbit')
        qHandle.addToOutputList('STARTORBIT','startorbit')
        qHandle.addToOutputList('BXLUMIVALUE_'+algoname,'bxlumivalue')
        qHandle.addToOutputList('BXLUMIERROR_'+algoname,'bxlumierr')
        qHandle.addToOutputList('BXLUMIQUALITY_'+algoname,'bxlumiqlty')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('lumilsnum','unsigned int')
        qResult.extend('numorbit','unsigned int')
        qResult.extend('startorbit','unsigned int')
        qResult.extend('bxlumivalue','blob')
        qResult.extend('bxlumierr','blob')
        qResult.extend('bxlumiqlty','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            lumilsnum=cursor.currentRow()['lumilsnum'].data()
            numorbit=cursor.currentRow()['numorbit'].data()
            startorbit=cursor.currentRow()['startorbit'].data()
            bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
            bxlumierr=cursor.currentRow()['bxlumierr'].data()
            bxlumiqlty=cursor.currentRow()['bxlumiqlty'].data()
            if not result.has_key(algoname):
                result[algoname]={}
            if not result[algoname].has_key(lumilsnum):
                result[algoname][lumilsnum]=[]
            result[algoname][lumilsnum].extend([cmslsnum,numorbit,startorbit,bxlumivalue,bxlumierr,bxlumiqlty])
    except :
        del qHandle
        raise RuntimeError(' dataDML.lumiBXById: '+str(e)) 
    del qHandle
    return result

def hltRunById(schema,dataid,hltpathname=None,hltpathpattern=None):
    '''
    result [runnum(0),datasource(1),npath(2),hltnamedict(3)]
    output :
         npath : total number of hltpath in DB
         hltnamedict : list of all selected paths [(hltpathidx,hltname),(hltpathidx,hltname)]
    '''
    result=[]    
    qHandle=schema.newQuery()
    runnum=None
    datasource=None
    npath=None
    hltnamedict=[]
    try:
        qHandle.addToTableList(nameDealer.hltdataTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('SOURCE','datasource')
        qHandle.addToOutputList('NPATH','npath')
        qHandle.addToOutputList('PATHNAMECLOB','pathnameclob')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('datasource','string')
        qResult.extend('npath','unsigned int')
        qResult.extend('pathnameclob','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        pathnameclob=None
        pathnames=[]
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            datasource=cursor.currentRow()['datasource'].data()
            npath=cursor.currentRow()['npath'].data()
            pathnameclob=cursor.currentRow()['pathnameclob'].data()
        if pathnameclob:
            pathnames=pathnameclob.split(',')
            for pathnameidx,hltname in enumerate(pathnames):
                if hltpathname:
                    if hltpathname==hltname:
                        hltnamedict.append((pathnameidx,hltname))
                        break
                elif hltpathpattern:
                    if fnmatch.fnmatch(hltname,hltpathpattern):
                        hltnamedict.append((pathnameidx,hltname))
                #else:
                    #hltnamedict.append((pathnameidx,hltname))
        result=[runnum,datasource,npath,hltnamedict]
    except :
        del qHandle
        raise 
    del qHandle
    return result

def hlttrgMappingByrun(schema,runnum,hltpathname=None,hltpathpattern=None):
    '''
    select m.hltpathname,m.l1seed from cmsrunsummary r,trghltmap m where r.runnum=:runnum and m.hltkey=r.hltkey and [m.hltpathname=:hltpathname] 
    output: {hltpath:l1seed}
    '''
    result={}
    queryHandle=schema.newQuery()
    r=nameDealer.cmsrunsummaryTableName()
    m=nameDealer.trghltMapTableName()
    if hltpathpattern and hltpathpattern in ['*','all','All','ALL']:
        hltpathpattern=None
    try:
        queryHandle.addToTableList(r)
        queryHandle.addToTableList(m)
        queryCondition=coral.AttributeList()
        queryCondition.extend('runnum','unsigned int')
        queryCondition['runnum'].setData(int(runnum))
        #queryHandle.addToOutputList(m+'.HLTKEY','hltkey')
        queryHandle.addToOutputList(m+'.HLTPATHNAME','hltpathname')
        queryHandle.addToOutputList(m+'.L1SEED','l1seed')
        conditionStr=r+'.RUNNUM=:runnum and '+m+'.HLTKEY='+r+'.HLTKEY'
        if hltpathname:
            hltpathpattern=None
            conditionStr+=' AND '+m+'.HLTPATHNAME=:hltpathname'
            queryCondition.extend('hltpathname','string')
            queryCondition['hltpathname'].setData(hltpathname)
        queryHandle.setCondition(conditionStr,queryCondition)
        queryResult=coral.AttributeList()
        queryResult.extend('pname','string')
        queryResult.extend('l1seed','string')
        queryHandle.defineOutput(queryResult)
        cursor=queryHandle.execute()
        while cursor.next():
            pname=cursor.currentRow()['pname'].data()
            l1seed=cursor.currentRow()['l1seed'].data()
            if not result.has_key(hltpathname):
                if hltpathpattern:
                    if fnmatch.fnmatch(pname,hltpathpattern):
                        result[pname]=l1seed
                else:
                    result[pname]=l1seed
    except :
        del queryHandle
        raise
    del queryHandle
    return result

def hltLSById(schema,dataid,hltpathname=None,hltpathpattern=None,withL1Pass=False,withHLTAccept=False):
    '''
    result (runnum, {cmslsnum:[(pathname,prescale,1lpass,hltaccept)](0)]} 
    '''
    #print 'entering hltLSById '
    #t0=time.time()
    result={}
    hltrundata=hltRunById(schema,dataid,hltpathname=hltpathname,hltpathpattern=hltpathpattern)
    if not hltrundata:
        return result        
    hltnamedict=hltrundata[3]
    if not hltnamedict:
        return (hltrundata[0],{})
    #tt1=time.time()
    #print '\thltrunbyid time ',tt1-t0
    #tt0=time.time()
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.lshltTableName())
        qHandle.addToOutputList('RUNNUM','runnum')
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        if len(hltnamedict)!=0:
            qHandle.addToOutputList('PRESCALEBLOB','prescaleblob')
        if withL1Pass:
            qHandle.addToOutputList('HLTCOUNTBLOB','hltcountblob')
        if withHLTAccept:
            qHandle.addToOutputList('HLTACCEPTBLOB','hltacceptblob')
        qConditionStr='DATA_ID=:dataid'
        qCondition=coral.AttributeList()
        qCondition.extend('dataid','unsigned long long')
        qCondition['dataid'].setData(dataid)
        qResult=coral.AttributeList()
        qResult.extend('runnum','unsigned int')
        qResult.extend('cmslsnum','unsigned int')
        if len(hltnamedict)!=0:
            qResult.extend('prescaleblob','blob')
        if withL1Pass:
            qResult.extend('hltcountblob','blob')
        if withHLTAccept:
            qResult.extend('hltacceptblob','blob')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['runnum'].data()
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            prescaleblob=None
            hltcountblob=None
            hltacceptblob=None
            if len(hltnamedict)!=0:
                prescaleblob=cursor.currentRow()['prescaleblob'].data()
            if withL1Pass:
                hltcountblob=cursor.currentRow()['hltcountblob'].data()
            if withHLTAccept:
                hltacceptblob=cursor.currentRow()['hltacceptblob'].data()
            if not result.has_key(cmslsnum):
                result[cmslsnum]=[]
            pathinfo=[]
            prescales=None
            hltcounts=None
            hltaccepts=None
            if prescaleblob:
                if runnum <150008: ###WORKAROUND PATCH!! because the 2010 blobs were packed as type l ###
                    prescales=CommonUtil.unpackBlobtoArray(prescaleblob,'l')
                else:
                    prescales=CommonUtil.unpackBlobtoArray(prescaleblob,'I')
            if hltcountblob:
                if runnum <150008: ###WORKAROUND PATCH!! because the 2010 blobs were packed as type l ###
                    hltcounts=CommonUtil.unpackBlobtoArray(hltcountblob,'l')
                else:
                    hltcounts=CommonUtil.unpackBlobtoArray(hltcountblob,'I')
            if hltacceptblob:
                if runnum <150008: ###WORKAROUND PATCH!! because the 2010 blobs were packed as type l ###
                    hltaccepts=CommonUtil.unpackBlobtoArray(hltacceptblob,'l')
                else:
                    hltaccepts=CommonUtil.unpackBlobtoArray(hltacceptblob,'I')
            for (hltpathidx,thispathname) in hltnamedict:#loop over selected paths
                thispresc=0
                thishltcount=0
                thisaccept=0
                if prescales:
                    thispresc=prescales[hltpathidx]
                if hltcounts:
                    thishltcount=hltcounts[hltpathidx]
                if hltaccepts:
                    thisaccept=hltaccepts[hltpathidx]
                thispathinfo=(thispathname,thispresc,thishltcount,thisaccept)
                pathinfo.append(thispathinfo)
            result[cmslsnum]=pathinfo
    except :
        del qHandle
        raise
    del qHandle
    #tt1=time.time()
    #print '\tdb stuff time ',tt1-tt0
    #t1=time.time()
    #print 'tot hltLSById time ',t1-t0
    return (runnum,result)

def intglumiForRange(schema,runlist):
    '''
    output: {run:intglumi_in_fb}
    '''
    result={}
    if not runlist:
        return result
    minrun=min(runlist)
    maxrun=max(runlist)
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.intglumiv2TableName())
        qResult=coral.AttributeList()
        qResult.extend('RUNNUM','unsigned int')
        qResult.extend('INTGLUMI','float')
        qConditionStr='RUNNUM>=:minrun AND RUNNUM<=:maxrun'
        qCondition=coral.AttributeList()
        qCondition.extend('minrun','unsigned int')
        qCondition.extend('maxrun','unsigned int')
        qCondition['minrun'].setData(minrun)
        qCondition['maxrun'].setData(maxrun)
        qHandle.addToOutputList('RUNNUM')
        qHandle.addToOutputList('INTGLUMI')
        qHandle.setCondition(qConditionStr,qCondition)
        qHandle.defineOutput(qResult)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['RUNNUM'].data()
            intglumi=cursor.currentRow()['INTGLUMI'].data()
            result[runnum]=intglumi
    except :
        del qHandle
        raise
    del qHandle
    return result

def fillschemePatternMap(schema,lumitype):
    '''
    output:(patternStr:correctionFac)
    '''
    if lumitype not in ['PIXEL','HF']:
        raise ValueError('[ERROR] unsupported lumitype '+lumitype)
    correctorField='CORRECTIONFACTOR'
    if lumitype=='PIXEL':
        correctorField='PIXELCORRECTIONFACTOR'
    result={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.fillschemeTableName())
        qResult=coral.AttributeList()
        qResult.extend('FILLSCHEMEPATTERN','string')
        qResult.extend('CORRECTIONFACTOR','float')
        qHandle.defineOutput(qResult)
        qHandle.addToOutputList('FILLSCHEMEPATTERN')
        qHandle.addToOutputList(correctorField)
        cursor=qHandle.execute()
        while cursor.next():
            fillschemePattern=cursor.currentRow()['FILLSCHEMEPATTERN'].data()
            afterglowfac=cursor.currentRow()['CORRECTIONFACTOR'].data()
            result[fillschemePattern]=afterglowfac
    except :
        del qHandle
        raise
    del qHandle
    return result

def guessLumiDataIdByRunInBranch(schema,runnum,tablename,branchName):
    revlist=revisionDML.revisionsInBranchName(schema,branchName)
    lumientry_id=revisionDML.entryInBranch(schema,tablename,str(runnum),branchName)
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,tablename,lumientry_id,revlist)
    return latestrevision
        
def guessTrgDataIdByRunInBranch(schema,runnum,tablename,branchName):    
    revlist=revisionDML.revisionsInBranchName(schema,branchName)
    trgentry_id=revisionDML.entryInBranch(schema,tablename,str(runnum),branchName)
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,tablename,trgentry_id,revlist)
    return latestrevision

def guessHltDataIdByRunInBranch(schema,runnum,tablename,branchName):    
    revlist=revisionDML.revisionsInBranchName(schema,branchName)
    hltentry_id=revisionDML.entryInBranch(schema,tablename,str(runnum),branchName)
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,tablename,hltentry_id,revlist)
    return latestrevision

def guessDataIdByRun(schema,runnum,tablename,revfilter=None):
    '''
    select max data_id of the given run. In current design, it's the most recent data of the run
    '''
    result=None
    ids=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(tablename)
        qHandle.addToOutputList('DATA_ID')
        qConditionStr='RUNNUM=:runnum '
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(runnum)
        qResult=coral.AttributeList()
        qResult.extend('DATA_ID','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            dataid=cursor.currentRow()['DATA_ID'].data()
            ids.append(dataid)
    except :
        del qHandle
        raise 
    del qHandle
    if len(ids)>0 :
        return max(ids)
    else:
        return result
        
def guessDataIdForRange(schema,inputRange,tablename):
    '''
    input: inputRange [run]
    output: {run:lumiid}
    select data_id,runnum from hltdata where runnum<=runmax and runnum>=:runmin 
    '''
    result={}
    if not inputRange : return result
    if len(inputRange)==1:
        trgid=guessDataIdByRun(schema,inputRange[0],tablename)
        result[inputRange[0]]=trgid
        return result
    rmin=min(inputRange)
    rmax=max(inputRange)
    result=dict.fromkeys(inputRange,None)
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(tablename)
        qHandle.addToOutputList('DATA_ID')
        qHandle.addToOutputList('RUNNUM')
        qConditionStr='RUNNUM>=:rmin'
        qCondition=coral.AttributeList()
        qCondition.extend('rmin','unsigned int')
        qCondition['rmin'].setData(rmin)
        if rmin!=rmax:
            qConditionStr+=' AND RUNNUM<=:rmax'
            qCondition.extend('rmax','unsigned int')
            qCondition['rmax'].setData(rmax)
        qResult=coral.AttributeList()
        qResult.extend('DATA_ID','unsigned long long')
        qResult.extend('RUNNUM','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            dataid=cursor.currentRow()['DATA_ID'].data()
            runnum=cursor.currentRow()['RUNNUM'].data()
            if result.has_key(runnum):
                if dataid>result[runnum]:
                    result[runnum]=dataid
    except :
        del qHandle
        raise 
    del qHandle
    return result
#def guessAllDataIdByRun(schema,runnum):
#    '''
#    get dataids by runnumber, if there are duplicates, pick max(dataid).Bypass full version lookups
#    result (lumidataid(0),trgdataid(1),hltdataid(2)) 
#    '''
#    lumiids=[]
#    trgids=[]
#    hltids=[]
#    qHandle=schema.newQuery()
#    try:
#        qHandle.addToTableList(nameDealer.lumidataTableName(),'l')
#        qHandle.addToTableList(nameDealer.trgdataTableName(),'t')
#        qHandle.addToTableList(nameDealer.hltdataTableName(),'h')
#        qHandle.addToOutputList('l.DATA_ID','lumidataid')
#        qHandle.addToOutputList('t.DATA_ID','trgdataid')
#        qHandle.addToOutputList('h.DATA_ID','hltdataid')
#        qConditionStr='l.RUNNUM=t.RUNNUM and t.RUNNUM=h.RUNNUM and l.RUNNUM=:runnum '
#        qCondition=coral.AttributeList()
#        qCondition.extend('runnum','unsigned int')
#        qCondition['runnum'].setData(runnum)
#        qResult=coral.AttributeList()
#        qResult.extend('lumidataid','unsigned long long')
#        qResult.extend('trgdataid','unsigned long long')
#        qResult.extend('hltdataid','unsigned long long')
#        qHandle.defineOutput(qResult)
#        qHandle.setCondition(qConditionStr,qCondition)
#        cursor=qHandle.execute()
#        while cursor.next():
#            lumidataid=cursor.currentRow()['lumidataid'].data()
#            trgdataid=cursor.currentRow()['trgdataid'].data()
#            hltdataid=cursor.currentRow()['hltdataid'].data()
#            lumiids.append(lumidataid)
#            trgids.append(trgdataid)
#            hltids.append(hltdataid)
#    except :
#        del qHandle
#        raise 
#    del qHandle
#    if len(lumiids)>0 and len(trgids)>0 and len(hltids)>0:
#        return (max(lumiids),max(trgids),max(hltids))
#    else:
#        return (None,None,None)

def guessnormIdByContext(schema,amodetag,egev1):
    '''
    get norm dataids by amodetag, egev if there are duplicates, pick max(dataid).Bypass full version lookups
    select data_id from luminorm where amodetag=:amodetag and egev_1=:egev1   
    '''
    luminormids=[]
    qHandle=schema.newQuery()
    egevmin=egev1*0.95
    egevmax=egev1*1.05
    try:
        qHandle.addToTableList( nameDealer.luminormTableName() )
        qHandle.addToOutputList('DATA_ID','normdataid')
        qConditionStr='AMODETAG=:amodetag AND EGEV_1>=:egevmin AND  EGEV_1<=:egevmax'
        qCondition=coral.AttributeList()
        qCondition.extend('amodetag','string')
        qCondition.extend('egevmin','unsigned int')
        qCondition.extend('egevmax','unsigned int')
        qCondition['amodetag'].setData(amodetag)
        qCondition['egevmin'].setData(int(egevmin))
        qCondition['egevmax'].setData(int(egevmax))
        qResult=coral.AttributeList()
        qResult.extend('normdataid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normdataid=cursor.currentRow()['normdataid'].data()
            luminormids.append(normdataid)
    except :
        del qHandle
        raise
    del qHandle
    if len(luminormids) !=0:return max(luminormids)
    return None

def guessnormIdByName(schema,normname):
    '''
    get norm dataids by name, if there are duplicates, pick max(dataid).Bypass full version lookups
    select data_id from luminorms where entry_name=:normname
    result luminormdataid
    '''   
    luminormids=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.luminormTableName() )
        qHandle.addToOutputList('DATA_ID','normdataid')
        qConditionStr='ENTRY_NAME=:normname '
        qCondition=coral.AttributeList()
        qCondition.extend('normname','string')
        qCondition['normname'].setData(normname)
        qResult=coral.AttributeList()
        qResult.extend('normdataid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            normdataid=cursor.currentRow()['normdataid'].data()
            luminormids.append(normdataid)
    except :
        del qHandle
        raise
    del qHandle
    if len(luminormids) !=0:return max(luminormids)
    return None

########
########
def dataentryIdByRun(schema,runnum,branchfilter):
    '''
    select el.entry_id,et.entry_id,eh.entry_id,el.revision_id,et.revision_id,eh.revision_id from lumidataentiries el,trgdataentries et,hltdataentries eh where el.name=et.name and et.name=eh.name and el.name=:entryname;
    check on entryrev
   
    return [lumientryid,trgentryid,hltentryid]
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(nameDealer.entryTableName( lumidataTableName() ))
        qHandle.addToTableList(nameDealer.entryTableName( trgdataTableName() ))
        qHandle.addToTableList(nameDealer.entryTableName( hltdataTableName() ))
        qHandle.addToOutputList(lumidataTableName()+'.ENTRY_ID','lumientryid')
        qHandle.addToOutputList(trgdataTableName()+'.ENTRY_ID','trgentryid')
        qHandle.addToOutputList(hltdataTableName()+'.ENTRY_ID','hltentryid')
        qConditionStr=lumidataTableName()+'.NAME='+trgdataTableName()+'.NAME AND '+trgdataTableName()+'.NAME='+hltdataTableName()+'.NAME AND '+lumidataTableName()+'.NAME=:runnumstr'
        qCondition=coral.AttributeList()
        qCondition.extend('runnumstr','string')
        qCondition['runnumstr'].setData(str(runnum))
        qResult=coral.AttributeList()
        qResult.extend('lumientryid','unsigned long long')
        qResult.extend('trgentryid','unsigned long long')
        qResult.extend('hltentryid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            lumientryid=cursor.currentRow()['lumientryid'].data()
            trgentryid=cursor.currentRow()['trgentryid'].data()
            hltentryid=cursor.currentRow()['hltentryid'].data()
            if lumientryid in branchfilter and trgentryid in branchfilter and hltentryid in branchfilter:
                result.extend([lumientryid,trgentryid,hltentryid])
    except:
        del qHandle
        raise 
    del qHandle
    return result

def latestdataIdByEntry(schema,entryid,datatype,branchfilter):
    '''
    select l.data_id,rl.revision_id from lumidatatable l,lumirevisions rl where  l.data_id=rl.data_id and l.entry_id=:entryid
    check revision_id is in branch
    '''
    dataids=[]
    datatablename=''
    revmaptablename=''
    if datatype=='lumi':
        datatablename=nameDealer.lumidataTableName()
    elif datatype=='trg':
        datatablename=nameDealer.trgdataTableName()
    elif dataytpe=='hlt':
        tablename=nameDealer.hltdataTableName()
    else:
        raise RunTimeError('datatype '+datatype+' is not supported')
    revmaptablename=nameDealer.revmapTableName(datatablename)
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList(revmaptablename)
        qHandle.addToTableList(datatablename)
        qHandle.addToOutputList('l.DATA_ID','dataid')
        qHandle.addToOutputList(revmaptablename+'.REVISION_ID','revisionid')
        qConditionStr=datatablename+'.DATA_ID='+revmaptablename+'.DATA_ID AND '+datatablename+'.ENTRY_ID=:entryid'
        qCondition=coral.AttributeList()
        qCondition.extend('entryid','unsigned long long')
        qResult=coral.AttributeList()
        qResult.extend('dataid','unsigned long long')
        qResult.extend('revisionid','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition(qConditionStr,qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            dataid=cursor.currentRow()['dataid'].data()
            revisionid=cursor.currentRow()['revisionid'].data()
            if revisionid in branchfilter:
                dataids.append(dataid)
    except:
        del qHandle
        raise
    del qHandle
    if len(dataids)!=0:return max(dataids)
    return None


#=======================================================
#   INSERT requires in update transaction
#=======================================================
def addNormToBranch(schema,normname,amodetag,norm1,egev1,optionalnormdata,branchinfo):
    '''
    input:
       branchinfo(normrevisionid,branchname)
       optionalnormdata {'norm_occ2':norm_occ2,'norm_et':norm_et,'norm_pu':norm_pu,'constfactor':constfactor}
    output:
       (revision_id,entry_id,data_id)
    '''
    #print 'branchinfo ',branchinfo
    norm_occ2=1.0
    if optionalnormdata.has_key('normOcc2'):
        norm_occ2=optionalnormdata['norm_occ2']
    norm_et=1.0
    if optionalnormdata.has_key('norm_et'):
        norm_et=optionalnormdata['norm_et']
    norm_pu=1.0
    if optionalnormdata.has_key('norm_pu'):
        norm_pu=optionalnormdata['norm_pu']
    constfactor=1.0
    if optionalnormdata.has_key('constfactor'):
        constfactor=optionalnormdata['constfactor']
    try:
        entry_id=revisionDML.entryInBranch(schema,nameDealer.luminormTableName(),normname,branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.luminormTableName())
            entryinfo=(revision_id,entry_id,normname,data_id)
            revisionDML.addEntry(schema,nameDealer.luminormTableName(),entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.luminormTableName() )
            revisionDML.addRevision(schema,nameDealer.luminormTableName(),(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','AMODETAG':'string','NORM_1':'float','EGEV_1':'unsigned int','NORM_OCC2':'float','NORM_ET':'float','NORM_PU':'float','CONSTFACTOR':'float'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':normname,'AMODETAG':amodetag,'NORM_1':norm1,'EGEV_1':egev1,'NORM_OCC2':norm_occ2,'NORM_ET':norm_et,'NORM_PU':norm_pu,'CONSTFACTOR':constfactor}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.luminormTableName(),tabrowDefDict,tabrowValueDict)
        return (revision_id,entry_id,data_id)
    except :
        raise
    
def addCorrToBranch(schema,corrname,a1,optionalcorrdata,branchinfo):
    '''
    input:
       branchinfo(corrrevisionid,branchname)
       optionalcorrdata {'a2':a2,'drift':drif}
    output:
       (revision_id,entry_id,data_id)
    '''
    a2=1.0
    if optionalcorrdata.has_key('a2'):
        a2=optionalcorrdata['a2']
    drift=1.0
    if optionalcorrdata.has_key('drift'):
        drift=optionalcorrdata['drift']
    try:
        entry_id=revisionDML.entryInBranch(schema,nameDealer.lumicorrectionsTableName(),corrname,branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.lumicorrectionsTableName())
            entryinfo=(revision_id,entry_id,corrname,data_id)
            revisionDML.addEntry(schema,nameDealer.lumicorrectionsTableName(),entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.lumicorrectionsTableName() )
            revisionDML.addRevision(schema,nameDealer.lumicorrectionsTableName(),(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','A1':'float','A2':'float','DRIFT':'float'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':corrname,'A1':a1,'A2':a2,'DRIFT':drift}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.lumicorrectionsTableName(),tabrowDefDict,tabrowValueDict)
        return (revision_id,entry_id,data_id)
    except :
        raise

def addLumiRunDataToBranch(schema,runnumber,lumirundata,branchinfo,tableName):
    '''
    input:
          lumirundata [datasource,nominalenergy]
          branchinfo (branch_id,branch_name)
          tableName lumiruntablename
    output:
          (revision_id,entry_id,data_id)
    '''
    try:
        datasource=lumirundata[0]
        nominalegev=3500.0
        if len(lumirundata)>1:
            nominalenergy=lumirundata[1]
        entry_id=revisionDML.entryInBranch(schema,tableName,str(runnumber),branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,tableName)
            entryinfo=(revision_id,entry_id,str(runnumber),data_id)
            revisionDML.addEntry(schema,tableName,entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision(schema,tableName)
            #print 'revision_id,data_id ',revision_id,data_id
            revisionDML.addRevision(schema,tableName,(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','RUNNUM':'unsigned int','SOURCE':'string','NOMINALEGEV':'float'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':str(runnumber),'RUNNUM':int(runnumber),'SOURCE':datasource,'NOMINALEGEV':nominalegev}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(tableName,tabrowDefDict,tabrowValueDict)
        return (revision_id,entry_id,data_id)
    except :
        raise

def addTrgRunDataToBranch(schema,runnumber,trgrundata,branchinfo):
    '''
    input:
       trgrundata [datasource(0),bitzeroname(1),bitnameclob(2)]
       bitnames clob, bitnames separated by ','
    output:
       (revision_id,entry_id,data_id)
    '''
    try:   #fixme: need to consider revision only case
        datasource=trgrundata[0]
        bitzeroname=trgrundata[1]
        bitnames=trgrundata[2]
        entry_id=revisionDML.entryInBranch(schema,nameDealer.trgdataTableName(),str(runnumber),branchinfo[1])
        if entry_id is None:
            (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.trgdataTableName())
            entryinfo=(revision_id,entry_id,str(runnumber),data_id)
            revisionDML.addEntry(schema,nameDealer.trgdataTableName(),entryinfo,branchinfo)
        else:
            (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.trgdataTableName() )
            revisionDML.addRevision(schema,nameDealer.trgdataTableName(),(revision_id,data_id),branchinfo)
        tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','SOURCE':'string','RUNNUM':'unsigned int','BITZERONAME':'string','BITNAMECLOB':'string'}
        tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':str(runnumber),'SOURCE':datasource,'RUNNUM':int(runnumber),'BITZERONAME':bitzeroname,'BITNAMECLOB':bitnames}
        db=dbUtil.dbUtil(schema)
        db.insertOneRow(nameDealer.trgdataTableName(),tabrowDefDict,tabrowValueDict)
        return (revision_id,entry_id,data_id)
    except :
        raise
def addHLTRunDataToBranch(schema,runnumber,hltrundata,branchinfo):
    '''
    input:
        hltrundata [pathnameclob(0),datasource(1)]
    output:
        (revision_id,entry_id,data_id)
    '''
    try:
         pathnames=hltrundata[0]
         datasource=hltrundata[1]
         npath=len(pathnames.split(','))
         entry_id=revisionDML.entryInBranch(schema,nameDealer.hltdataTableName(),str(runnumber),branchinfo[1])
         if entry_id is None:
             (revision_id,entry_id,data_id)=revisionDML.bookNewEntry(schema,nameDealer.hltdataTableName())
             entryinfo=(revision_id,entry_id,str(runnumber),data_id)
             revisionDML.addEntry(schema,nameDealer.hltdataTableName(),entryinfo,branchinfo)
         else:
             (revision_id,data_id)=revisionDML.bookNewRevision( schema,nameDealer.hltdataTableName() )
             revisionDML.addRevision(schema,nameDealer.hltdataTableName(),(revision_id,data_id),branchinfo)
         tabrowDefDict={'DATA_ID':'unsigned long long','ENTRY_ID':'unsigned long long','ENTRY_NAME':'string','RUNNUM':'unsigned int','SOURCE':'string','NPATH':'unsigned int','PATHNAMECLOB':'string'}
         tabrowValueDict={'DATA_ID':data_id,'ENTRY_ID':entry_id,'ENTRY_NAME':str(runnumber),'RUNNUM':int(runnumber),'SOURCE':datasource,'NPATH':npath,'PATHNAMECLOB':pathnames}
         db=dbUtil.dbUtil(schema)
         db.insertOneRow(nameDealer.hltdataTableName(),tabrowDefDict,tabrowValueDict)
         return (revision_id,entry_id,data_id)
    except :
        raise 

def insertRunSummaryData(schema,runnumber,runsummarydata,complementalOnly=False):
    '''
    input:
        runsummarydata [l1key,amodetag,egev,sequence,hltkey,fillnum,starttime,stoptime]
    output:
    '''
    l1key=runsummarydata[0]
    amodetag=runsummarydata[1]
    egev=runsummarydata[2]
    hltkey=''
    fillnum=0
    sequence=''
    starttime=''
    stoptime=''
    if not complementalOnly:
        sequence=runsummarydata[3]
        hltkey=runsummarydata[4]
        fillnum=runsummarydata[5]
        starttime=runsummarydata[6]
        stoptime=runsummarydata[7]
    try:
        if not complementalOnly:
            tabrowDefDict={'RUNNUM':'unsigned int','L1KEY':'string','AMODETAG':'string','EGEV':'unsigned int','SEQUENCE':'string','HLTKEY':'string','FILLNUM':'unsigned int','STARTTIME':'time stamp','STOPTIME':'time stamp'}
            tabrowValueDict={'RUNNUM':int(runnumber),'L1KEY':l1key,'AMODETAG':amodetag,'EGEV':int(egev),'SEQUENCE':sequence,'HLTKEY':hltkey,'FILLNUM':int(fillnum),'STARTTIME':starttime,'STOPTIME':stoptime}
            db=dbUtil.dbUtil(schema)
            db.insertOneRow(nameDealer.cmsrunsummaryTableName(),tabrowDefDict,tabrowValueDict)
        else:
            setClause='L1KEY=:l1key,AMODETAG=:amodetag,EGEV=:egev'
            updateCondition='RUNNUM=:runnum'
            inputData=coral.AttributeList()
            inputData.extend('l1key','string')
            inputData.extend('amodetag','string')
            inputData.extend('egev','unsigned int')
            inputData.extend('runnum','unsigned int')
            inputData['l1key'].setData(l1key)
            inputData['amodetag'].setData(amodetag)
            inputData['egev'].setData(int(egev))
            inputData['runnum'].setData(int(runnumber))
            db=dbUtil.dbUtil(schema)
            db.singleUpdate(nameDealer.cmsrunsummaryTableName(),setClause,updateCondition,inputData)
    except :
        raise   
def insertTrgHltMap(schema,hltkey,trghltmap):
    '''
    input:
        trghltmap {hltpath:l1seed}
    output:
    '''
    hltkeyExists=False
    nrows=0
    try:
        kQueryBindList=coral.AttributeList()
        kQueryBindList.extend('hltkey','string')
        kQuery=schema.newQuery()
        kQuery.addToTableList(nameDealer.trghltMapTableName())
        kQuery.setCondition('HLTKEY=:hltkey',kQueryBindList)
        kQueryBindList['hltkey'].setData(hltkey)
        kResult=kQuery.execute()
        while kResult.next():
            hltkeyExists=True
        if not hltkeyExists:
            bulkvalues=[]   
            trghltDefDict=[('HLTKEY','string'),('HLTPATHNAME','string'),('L1SEED','string')]
            for hltpath,l1seed in trghltmap.items():
                bulkvalues.append([('HLTKEY',hltkey),('HLTPATHNAME',hltpath),('L1SEED',l1seed)])
            db=dbUtil.dbUtil(schema)
            db.bulkInsert(nameDealer.trghltMapTableName(),trghltDefDict,bulkvalues)
            nrows=len(bulkvalues)
        return nrows
    except :
        print 'error in insertTrgHltMap '
        raise
def bulkInsertTrgLSData(session,runnumber,data_id,trglsdata,bulksize=500):
    '''
    insert trg per-LS data for given run and data_id, this operation can be split in transaction chuncks 
    input:
        trglsdata {cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    result nrows inserted
    if nrows==0, then this insertion failed
    '''
    print 'total number of trg rows ',len(trglsdata)
    lstrgDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('DEADTIMECOUNT','unsigned long long'),('BITZEROCOUNT','unsigned int'),('BITZEROPRESCALE','unsigned int'),('PRESCALEBLOB','blob'),('TRGCOUNTBLOB','blob')]
    committedrows=0
    nrows=0
    bulkvalues=[]
    try:
        for cmslsnum,perlstrg in trglsdata.items():
            deadtimecount=perlstrg[0]           
            bitzerocount=perlstrg[1]
            bitzeroprescale=perlstrg[2]
            trgcountblob=perlstrg[3]
            trgprescaleblob=perlstrg[4]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('DEADTIMECOUNT',deadtimecount),('BITZEROCOUNT',bitzerocount),('BITZEROPRESCALE',bitzeroprescale),('PRESCALEBLOB',trgprescaleblob),('TRGCOUNTBLOB',trgcountblob)])
            nrows+=1
            committedrows+=1
            if nrows==bulksize:
                print 'committing trg in LS chunck ',nrows
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert(nameDealer.lstrgTableName(),lstrgDefDict,bulkvalues)
                session.transaction().commit()
                nrows=0
                bulkvalues=[]
            elif committedrows==len(trglsdata):
                print 'committing trg at the end '
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert(nameDealer.lstrgTableName(),lstrgDefDict,bulkvalues)
                session.transaction().commit()
    except :
        print 'error in bulkInsertTrgLSData'
        raise 
def bulkInsertHltLSData(session,runnumber,data_id,hltlsdata,bulksize=500):
    '''
    input:
    hltlsdata {cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}
    '''
    print 'total number of hlt rows ',len(hltlsdata)
    lshltDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('CMSLSNUM','unsigned int'),('PRESCALEBLOB','blob'),('HLTCOUNTBLOB','blob'),('HLTACCEPTBLOB','blob')]
    committedrows=0
    nrows=0
    bulkvalues=[]   
    try:             
        for cmslsnum,perlshlt in hltlsdata.items():
            inputcountblob=perlshlt[0]
            acceptcountblob=perlshlt[1]
            prescaleblob=perlshlt[2]
            bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('CMSLSNUM',cmslsnum),('PRESCALEBLOB',prescaleblob),('HLTCOUNTBLOB',inputcountblob),('HLTACCEPTBLOB',acceptcountblob)])
            
            nrows+=1
            committedrows+=1
            if nrows==bulksize:
                print 'committing hlt in LS chunck ',nrows
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert(nameDealer.lshltTableName(),lshltDefDict,bulkvalues)
                session.transaction().commit()
                nrows=0
                bulkvalues=[]
            elif committedrows==len(hltlsdata):
                print 'committing hlt at the end '
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert(nameDealer.lshltTableName(),lshltDefDict,bulkvalues)
                session.transaction().commit()
    except  :
        print 'error in bulkInsertHltLSData'
        raise 
    
def bulkInsertLumiLSSummary(session,runnumber,data_id,lumilsdata,tableName,bulksize=500,withDetails=True):
    '''
    input:
          lumilsdata {lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beam1intensity,beam2intensity,bxlumivalue_occ1,bxlumierror_occ1,bxlumiquality_occ1,bxlumivalue_occ2,bxlumierror_occ2,bxlumiquality_occ2,bxlumivalue_et,bxlumierror_et,bxlumiquality_et]}
    '''
    lslumiDefDict=[]
    if withDetails:
        lslumiDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('INSTLUMI','float'),('INSTLUMIERROR','float'),('INSTLUMIQUALITY','short'),('BEAMSTATUS','string'),('BEAMENERGY','float'),('NUMORBIT','unsigned int'),('STARTORBIT','unsigned int'),('CMSBXINDEXBLOB','blob'),('BEAMINTENSITYBLOB_1','blob'),('BEAMINTENSITYBLOB_2','blob'),('BXLUMIVALUE_OCC1','blob'),('BXLUMIERROR_OCC1','blob'),('BXLUMIQUALITY_OCC1','blob'),('BXLUMIVALUE_OCC2','blob'),('BXLUMIERROR_OCC2','blob'),('BXLUMIQUALITY_OCC2','blob'),('BXLUMIVALUE_ET','blob'),('BXLUMIERROR_ET','blob'),('BXLUMIQUALITY_ET','blob')]
    else:
        lslumiDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('INSTLUMI','float'),('INSTLUMIERROR','float'),('INSTLUMIQUALITY','short'),('BEAMSTATUS','string'),('BEAMENERGY','float'),('NUMORBIT','unsigned int'),('STARTORBIT','unsigned int')]
    print 'total number of lumi rows ',len(lumilsdata)
    try:
        committedrows=0
        nrows=0
        bulkvalues=[]
        for lumilsnum,perlslumi in lumilsdata.items():
            cmslsnum=perlslumi[0]
            instlumi=perlslumi[1]
            instlumierror=perlslumi[2]
            instlumiquality=perlslumi[3]
            beamstatus=perlslumi[4]
            beamenergy=perlslumi[5]
            numorbit=perlslumi[6]
            startorbit=perlslumi[7]
            if withDetails:
                cmsbxindexindexblob=perlslumi[8]
                beam1intensity=perlslumi[9]
                beam2intensity=perlslumi[10]
                bxlumivalue_occ1=perlslumi[11]
                bxlumierror_occ1=perlslumi[12]
                bxlumiquality_occ1=perlslumi[13]
                bxlumivalue_occ2=perlslumi[14]
                bxlumierror_occ2=perlslumi[15]
                bxlumiquality_occ2=perlslumi[16]
                bxlumivalue_et=perlslumi[17]
                bxlumierror_et=perlslumi[18]
                bxlumiquality_et=perlslumi[19]
                bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('INSTLUMI',instlumi),('INSTLUMIERROR',instlumierror),('INSTLUMIQUALITY',instlumiquality),('BEAMSTATUS',beamstatus),('BEAMENERGY',beamenergy),('NUMORBIT',numorbit),('STARTORBIT',startorbit),('CMSBXINDEXBLOB',cmsbxindexindexblob),('BEAMINTENSITYBLOB_1',beam1intensity),('BEAMINTENSITYBLOB_2',beam2intensity),('BXLUMIVALUE_OCC1',bxlumivalue_occ1),('BXLUMIERROR_OCC1',bxlumierror_occ1),('BXLUMIQUALITY_OCC1',bxlumiquality_occ1),('BXLUMIVALUE_OCC2',bxlumivalue_occ2),('BXLUMIERROR_OCC2',bxlumierror_occ2),('BXLUMIQUALITY_OCC2',bxlumiquality_occ2),('BXLUMIVALUE_ET',bxlumivalue_et),('BXLUMIERROR_ET',bxlumierror_et),('BXLUMIQUALITY_ET',bxlumiquality_et)])
            else:
                bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('INSTLUMI',instlumi),('INSTLUMIERROR',instlumierror),('INSTLUMIQUALITY',instlumiquality),('BEAMSTATUS',beamstatus),('BEAMENERGY',beamenergy),('NUMORBIT',numorbit),('STARTORBIT',startorbit)])
            nrows+=1
            committedrows+=1
            if nrows==bulksize:
                print 'committing lumi in LS chunck ',nrows
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert(tableName,lslumiDefDict,bulkvalues)
                session.transaction().commit()
                nrows=0
                bulkvalues=[]
            elif committedrows==len(lumilsdata):
                print 'committing lumi at the end '
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert(tableName,lslumiDefDict,bulkvalues)
                session.transaction().commit()
    except :
        raise

#def insertLumiLSDetail(schema,runnumber,data_id,lumibxdata):
#    '''
#    input:
#          lumibxdata [(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}),(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]}),(algoname,{lumilsnum:[cmslsnum,bxlumivalue,bxlumierror,bxlumiquality]})]
#    output:
#          nrows
#    '''
#    try:
#        nrow=0
#        bulkvalues=[]
#        lslumiDefDict=[('DATA_ID','unsigned long long'),('RUNNUM','unsigned int'),('LUMILSNUM','unsigned int'),('CMSLSNUM','unsigned int'),('ALGONAME','string'),('BXLUMIVALUE','blob'),('BXLUMIERROR','blob'),('BXLUMIQUALITY','blob')]
#        for (algoname,peralgobxdata) in lumibxdata:
#            for lumilsnum,bxdata in peralgobxdata.items():
#                cmslsnum=bxdata[0]
#                bxlumivalue=bxdata[1]
#                bxlumierror=bxdata[2]
#                bxlumiquality=bxdata[3]
#                bulkvalues.append([('DATA_ID',data_id),('RUNNUM',runnumber),('LUMILSNUM',lumilsnum),('CMSLSNUM',cmslsnum),('ALGONAME',algoname),('BXLUMIVALUE',bxlumivalue),('BXLUMIERROR',bxlumierror),('BXLUMIQUALITY',bxlumiquality)])
#        db=dbUtil.dbUtil(schema)
#        db.bulkInsert(nameDealer.lumidetailTableName(),lslumiDefDict,bulkvalues)
#        return len(bulkvalues)
#    except:
#        raise 
    
#def completeOldLumiData(schema,runnumber,lsdata,data_id):
#    '''
#    input:
#    lsdata [[lumisummary_id,lumilsnum,cmslsnum]]
#    '''
#    try:
#        #update in lumisummary table
#        #print 'insert in lumisummary table'
#        setClause='DATA_ID=:data_id'
#        updateCondition='RUNNUM=:runnum AND DATA_ID is NULL'
#        updateData=coral.AttributeList()
#        updateData.extend('data_id','unsigned long long')
#        updateData.extend('runnum','unsigned int')
#        updateData['data_id'].setData(data_id)
#        updateData['runnum'].setData(int(runnumber))
#        db=dbUtil.dbUtil(schema)
#        db.singleUpdate(nameDealer.lumisummaryTableName(),setClause,updateCondition,updateData)
#        #updates in lumidetail table
#        updateAction='DATA_ID=:data_id,RUNNUM=:runnum,CMSLSNUM=:cmslsnum,LUMILSNUM=:lumilsnum'
#        updateCondition='LUMISUMMARY_ID=:lumisummary_id'
#        bindvarDef=[]
#        bindvarDef.append(('data_id','unsigned long long'))
#        bindvarDef.append(('runnum','unsigned int'))
#        bindvarDef.append(('cmslsnum','unsigned int'))
#        bindvarDef.append(('lumilsnum','unsigned int'))        
#        inputData=[]
#        for [lumisummary_id,lumilsnum,cmslsnum] in lsdata:
#            inputData.append([('data_id',data_id),('runnum',int(runnumber)),('cmslsnum',cmslsnum),('lumilsnum',lumilsnum)])
#        db.updateRows(nameDealer.lumidetailTableName(),updateAction,updateCondition,bindvarDef,inputData)
#    except:
#        raise
    
#=======================================================
#   DELETE
#=======================================================


#=======================================================
#   Unit Test
#=======================================================
if __name__ == "__main__":
    import sessionManager
    import lumidbDDL,revisionDML,generateDummyData
    #myconstr='sqlite_file:test2.db'
    myconstr='oracle://devdb10/cms_xiezhen_dev'
    svc=sessionManager.sessionManager(myconstr,authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    schema=session.nominalSchema()
    session.transaction().start(False)
    lumidbDDL.dropTables(schema,nameDealer.schemaV2Tables())
    lumidbDDL.dropTables(schema,nameDealer.commonTables())
    tables=lumidbDDL.createTables(schema)
    try:
    #    #lumidbDDL.createUniqueConstraints(schema)
        trunkinfo=revisionDML.createBranch(schema,'TRUNK',None,comment='main')
        print trunkinfo
        datainfo=revisionDML.createBranch(schema,'DATA','TRUNK',comment='hold data')
        print datainfo
        norminfo=revisionDML.createBranch(schema,'NORM','TRUNK',comment='hold normalization factor')
        print norminfo
    except:
        raise
        #print 'branch already exists, do nothing'
    (normbranchid,normbranchparent)=revisionDML.branchInfoByName(schema,'NORM')
    normbranchinfo=(normbranchid,'NORM')
    addNormToBranch(schema,'pp7TeV','PROTPHYS',6370.0,3500,{},normbranchinfo)
    addNormToBranch(schema,'hi7TeV','HIPHYS',2.38,3500,{},normbranchinfo)
    (branchid,branchparent)=revisionDML.branchInfoByName(schema,'DATA')
    branchinfo=(branchid,'DATA')
    for runnum in [1200,1211,1222,1233,1345]:
        runsummarydata=generateDummyData.runsummary(schema,'PROTPHYS',3500)
        insertRunSummaryData(schema,runnum,runsummarydata)
        hlttrgmap=generateDummyData.hlttrgmap(schema)
        insertTrgHltMap(schema,hlttrgmap[0],hlttrgmap[1])
        lumidummydata=generateDummyData.lumiSummary(schema,20)
        lumirundata=[lumidummydata[0]]
        lumilsdata=lumidummydata[1]
        (lumirevid,lumientryid,lumidataid)=addLumiRunDataToBranch(schema,runnum,lumirundata,branchinfo)
        insertLumiLSSummary(schema,runnum,lumidataid,lumilsdata)
        trgdata=generateDummyData.trg(schema,20)        
        trgrundata=[trgdata[0],trgdata[1],trgdata[2]]
        trglsdata=trgdata[3]
        (trgrevid,trgentryid,trgdataid)=addTrgRunDataToBranch(schema,runnum,trgrundata,branchinfo)
        insertTrgLSData(schema,runnum,trgdataid,trglsdata)        
        hltdata=generateDummyData.hlt(schema,20)
        hltrundata=[hltdata[0],hltdata[1]]
        hltlsdata=hltdata[2]
        (hltrevid,hltentryid,hltdataid)=addHLTRunDataToBranch(schema,runnum,hltrundata,branchinfo)
        insertHltLSData(schema,runnum,hltdataid,hltlsdata)
    session.transaction().commit()
    print 'test reading'
    session.transaction().start(True)
    print '===inspecting NORM by name==='
    normrevlist=revisionDML.revisionsInBranchName(schema,'NORM')
    luminormentry_id=revisionDML.entryInBranch(schema,nameDealer.luminormTableName(),'pp7TeV','NORM')
    latestNorms=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.luminormTableName(),luminormentry_id,normrevlist)
    print 'latest norm data_id for pp7TeV ',latestNorms
    
    print '===inspecting DATA branch==='
    print revisionDML.branchType(schema,'DATA')
    revlist=revisionDML.revisionsInBranchName(schema,'DATA')
    print revlist
    lumientry_id=revisionDML.entryInBranch(schema,nameDealer.lumidataTableName(),'1211','DATA')
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.lumidataTableName(),lumientry_id,revlist)
    print 'latest lumi data_id for run 1211 ',latestrevision
    lumientry_id=revisionDML.entryInBranch(schema,nameDealer.lumidataTableName(),'1222','DATA')
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.lumidataTableName(),lumientry_id,revlist)
    print 'latest lumi data_id for run 1222 ',latestrevision
    trgentry_id=revisionDML.entryInBranch(schema,nameDealer.trgdataTableName(),'1222','DATA')
    latestrevision=revisionDML.latestDataRevisionOfEntry(schema,nameDealer.trgdataTableName(),trgentry_id,revlist)
    print 'latest trg data_id for run 1222 ',latestrevision
    session.transaction().commit()
    print 'tagging data so far as data_orig'
    session.transaction().start(False)
    (revisionid,parentid,parentname)=revisionDML.createBranch(schema,'data_orig','DATA',comment='tag of 2010data')
    session.transaction().commit()
    session.transaction().start(True)
    print revisionDML.branchType(schema,'data_orig')
    revlist=revisionDML.revisionsInTag(schema,revisionid,branchinfo[0])
    print revlist
    session.transaction().commit()
    session.transaction().start(False)
    for runnum in [1200,1222]:
        print 'revising lumidata for run ',runnum
        lumidummydata=generateDummyData.lumiSummary(schema,20)
        lumirundata=[lumidummydata[0]]
        lumilsdata=lumidummydata[1]
        (lumirevid,lumientryid,lumidataid)=addLumiRunDataToBranch(schema,runnum,lumirundata,branchinfo)
        insertLumiLSSummary(schema,runnum,lumidataid,lumilsdata)
    revlist=revisionDML.revisionsInTag(schema,revisionid,branchinfo[0])
    print 'revisions in branch DATA',revisionDML.revisionsInBranch(schema,branchinfo[0])
    session.transaction().commit()
    #print 'revisions in tag data_orig ',revlist
    
    print '===test reading==='
    session.transaction().start(True)
    #print 'guess norm by name'
    #normid1=guessnormIdByName(schema,'pp7TeV')
    #print 'normid1 ',normid1
    #normid2=guessnormIdByContext(schema,'PROTPHYS',3500)
    #print 'guess norm of PROTPHYS 3500'
    #print 'normid2 ',normid2
    #normid=normid2
    #(lumidataid,trgdataid,hltdataid)=guessDataIdByRun(schema,1200)
    #print 'normid,lumiid,trgid,hltid ',normid,lumidataid,trgdataid,hltdataid
    #print 'lumi norm'
    #print luminormById(schema,normid)
    #print 'runinfo '
    #print runsummary(schema,runnum,session.properties().flavorName())
    #print 'lumirun '
    #print lumiRunById(schema,lumidataid)
    #print 'lumisummary'
    #print lumiLSById(schema,lumidataid)
    #print 'beam info'
    #print beamInfoById(schema,lumidataid)
    #print 'lumibx by algo OCC1'
    #print lumiBXByAlgo(schema,lumidataid,'OCC1')
    print 'trg run, trgdataid ',trgdataid
    print trgRunById(schema,trgdataid,withblobdata=True)  
    #print 'trg ls'
    #print trgLSById(schema,trgdataid)
    #print 'hlt run'
    #print hltRunById(schema,hltdataid)
    #print 'hlt ls'
    #print hltLSById(schema,hltdataid)
    session.transaction().commit()
    del session

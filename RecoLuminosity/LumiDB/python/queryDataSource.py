import array,coral
from RecoLuminosity.LumiDB import CommonUtil,nameDealer

def trgFromOldLumi(session,runnumber):
    '''
    select bitnum,bitname from trg where runnum=:runnumber and cmslsnum=1 order by bitnum
    select cmslsnum,deadtime,trgcount,prescale from trg where bitnum=:bitnum and runnum=:runnumber 
    input: runnumber
    output: [bitnames,{cmslsnum,[deadtime,bitzerocount,bitzerpoprescale,trgcountBlob,trgprescaleBlob]}]
    '''
    session.transaction().start(True)
    lumischema=session.nominalSchema()
    qHandle=lumischema.newQuery()
    try:
        qHandle=lumischema.newQuery()
        qHandle.addToTableList(nameDealer.trgTableName())
        qHandle.addToOutputList('BITNUM','bitnum')
        qHandle.addToOutputList('BITNAME','bitname')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnumber))
        qCondition.extend('cmslsnum','unsigned int')
        qCondition['cmslsnum'].setData(int(1))
        qResult=coral.AttributeList()
        qResult.extend('bitnum','unsigned int')
        qResult.extend('bitname','string')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum AND CMSLSNUM=:cmslsnum',qCondition)
        qHandle.addToOrderList('BITNUM')
        cursor=qHandle.execute()
        bitnums=[]
        bitnameList=[]
        while cursor.next():
            bitnum=cursor.currentRow()['bitnum'].data()
            bitname=cursor.currentRow()['bitname'].data()
            bitnums.append(bitnum)
            bitnameList.append(bitname)
        del qHandle
        bitnames=','.join(bitnameList)
        databuffer={}
        nbits=len(bitnums)
        qHandle=lumischema.newQuery()
        qHandle.addToTableList(nameDealer.trgTableName())
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('BITNUM','bitnum')
        qHandle.addToOutputList('DEADTIME','deadtime')
        qHandle.addToOutputList('TRGCOUNT','trgcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnumber))
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('bitnum','unsigned int')
        qResult.extend('deadtime','unsigned long long')
        qResult.extend('trgcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('CMSLSNUM')
        qHandle.addToOrderList('BITNUM')
        cursor=qHandle.execute()
        trgcountArray=array.array('l')
        prescaleArray=array.array('l')
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            bitnum=cursor.currentRow()['bitnum'].data()
            deadtime=cursor.currentRow()['deadtime'].data()
            trgcount=cursor.currentRow()['trgcount'].data()
            prescale=cursor.currentRow()['prescale'].data()
            if not databuffer.has_key(cmslsnum):
                databuffer[cmslsnum]=[]
                databuffer[cmslsnum].append(deadtime)
            if bitnum==0:
                databuffer[cmslsnum].append(trgcount)
                databuffer[cmslsnum].append(prescale)
            trgcountArray.append(trgcount)
            prescaleArray.append(prescale)
            if bitnum==nbits-1:
                trgcountBlob=CommonUtil.packArraytoBlob(trgcountArray)
                prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)
                databuffer[cmslsnum].append(trgcountBlob)
                databuffer[cmslsnum].append(prescaleBlob)
                trgcountArray=array.array('l')
                prescaleArray=array.array('l')
        del qHandle            
        session.transaction().commit()
        return [bitnames,databuffer]
    except:
        del qHandle
        raise

def trgFromWBM(session,runnumber):
    '''
    '''
    pass

def trgFromGT(session,runnumber):
    '''
    select counts,lsnr,algobit from cms_gt_mon.gt_mon_trig_algo_view where runnr=:runnumber order by lsnr,algobit
    select counts,lsnr,techbit from cms_gt_mon.gt_mon_trig_tech_view where runnr=:runnumber order by lsnr,techbit
    select counts,lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr
    select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber order by algo_index
    select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber order by techtrig_index
    select prescale_factor_algo_000,prescale_factor_algo_001..._127 from cms_gt.gt_run_presc_algo_view where runnr=:runnumber and prescale_index=0;
    select prescale_factor_tt_000,prescale_factor_tt_001..._63 from cms_gt.gt_run_presc_tech_view where runnr=:runnumber and prescale_index=0;
    '''
    pass

def trgFromOldGT(session,runnumber):
    '''
    input: runnumber
    if complementalOnly is True:
       select deadfrac from 
    else:
    output: [bitnameclob,{cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}]
    select counts,lsnr,algobit from cms_gt_mon.gt_mon_trig_algo_view where runnr=:runnumber order by lsnr,algobit
    select counts,lsnr,techbit from cms_gt_mon.gt_mon_trig_tech_view where runnr=:runnumber order by lsnr,techbit
    select counts,lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr
    select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber order by algo_index
    select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber order by techtrig_index
    select prescale_factor_algo_000,prescale_factor_algo_001..._127 from cms_gt.gt_run_presc_algo_view where runnr=:runnumber and prescale_index=0;
    select prescale_factor_tt_000,prescale_factor_tt_001..._63 from cms_gt.gt_run_presc_tech_view where runnr=:runnumber and prescale_index=0;
    '''
    pass
    #bitnames=''
    #databuffer={} #{cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    #qHandle=schema.newQuery()
    #try:
        
    #except:
    #    del qHandle
    #    raise 

def hltFromRuninfoV2(session,runnumber):
    '''
    input:
    output: [datasource,pathnameclob,{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}]
    select count(distinct PATHNAME) as npath from HLT_SUPERVISOR_LUMISECTIONS_V2 where runnr=:runnumber and lsnumber=1;
    select l.pathname,l.lsnumber,l.l1pass,l.paccept,m.psvalue from hlt_supervisor_lumisections_v2 l,hlt_supervisor_scalar_map m where l.runnr=m.runnr and l.psindex=m.psindex and l.pathname=m.pathname and l.runnr=:runnumber order by l.lsnumber
    
    '''
    pass

def hltFromRuninfoV3(session,runnumber):
    '''
    input:
    output: [datasource,pathnameclob,{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}]
    select count(distinct PATHNAME) as npath from HLT_SUPERVISOR_LUMISECTIONS_V2 where runnr=:runnumber and lsnumber=1;
    select l.pathname,l.lsnumber,l.l1pass,l.paccept,m.psvalue from hlt_supervisor_lumisections_v2 l,hlt_supervisor_scalar_map m where l.runnr=m.runnr and l.psindex=m.psindex and l.pathname=m.pathname and l.runnr=:runnumber order by l.lsnumber
    
    '''
    pass

def hltFromOldLumi(session,runnumber):
    '''
    '''
    pass

def hltconf(schema,hltkey):
    '''
    select paths.pathid,paths.name,stringparamvalues.value from stringparamvalues,paths,parameters,superidparameterassoc,modules,moduletemplates,pathmoduleassoc,configurationpathassoc,configurations where parameters.paramid=stringparamvalues.paramid and  superidparameterassoc.paramid=parameters.paramid and modules.superid=superidparameterassoc.superid and moduletemplates.superid=modules.templateid and pathmoduleassoc.moduleid=modules.superid and paths.pathid=pathmoduleassoc.pathid and configurationpathassoc.pathid=paths.pathid and configurations.configid=configurationpathassoc.configid and moduletemplates.name='HLTLevel1GTSeed' and parameters.name='L1SeedsLogicalExpression' and configurations.configid=1905; 

    '''
    pass

def runsummary(session,runnumber,complementalOnly=False):
    '''
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:SEQ_NAME'
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:HLT_KEY_DESCRIPTION';
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.SCAL:FILLN' and rownum<=1;
    select time from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:START_TIME_T';
    select time from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:STOP_TIME_T';
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='AMODETAG'
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='EGEV'
    input:
    output:[hltkey,l1key,fillnum,sequence,starttime,stoptime,amodetag,egev]
    if complementalOnly:
       
    '''
    pass
    
if __name__ == "__main__":
    from RecoLuminosity.LumiDB import sessionManager
    svc=sessionManager.sessionManager('oracle://cms_orcoff_prod/cms_lumi_prod',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    lsresult=trgFromOldLumi(session,149181)
    print lsresult
    del session

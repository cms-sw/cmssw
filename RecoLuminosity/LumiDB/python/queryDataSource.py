import array,coral
from RecoLuminosity.LumiDB import CommonUtil,nameDealer

def uncalibrateLumi(runnumber,instlumi,instlumierror):
    '''
    input: runnumber,calibrated instlumi,calibrated instlumierror
    output: (uncalibrated instlumi, uncalbrated instlumierror)
    '''
    normpp7tev=float(6370.0)
    normpp900gev=float(16500.0)
    normhi7tev=float(2.383)
    oldnormpp7tev=float(6330)
    if runnumber in [136290,136294,136297,134721,134725]:#900Gev pp runs
        return (float(instlumi)/normpp900gev,float(instlumierror)/normpp900gev)
    if runnumber >=150431 and runnumber<=153368 :#7TeV HI runs
        return (float(instlumi)/normhi7tev,float(instlumierror)/normhi7tev)
    return (float(instlumi)/normpp7tev,float(instlumierror)/normpp7tev)
    
def uncalibratedlumiFromOldLumi(session,runnumber):
    '''
    retrieve old lumi value, divide by norm and restore to raw value
    select lumilsnum,cmslsnum,instlumi,instlumierror,instlumiquality,startorbit,numorbit,beamenergy,beamstatus,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2 from lumisummary where runnum=:runnumber order by lumilsnum
    
    select s.lumilsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality from lumidetail d,lumisummary s where d.lumisummary_id=s.lumisummary_id and s.runnum=:runnumber and d.algoname=:algoname order by s.lumilsnum
    
    output: {lumilsnum:[cmslsnum,uncalibratedinstlumi,uncalibratedinstlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2,bxlumivalue_occ1,bxlumierror_occ1,,bxlumiquality_occ1,bxlumivalue_occ2,bxlumierror_occ2,bxlumiquality_occ2,bxlumivalue_et,bxlumierror_et,bxlumiquality_et]}]}
    dict size ~ 200mb for 1000LS
    '''
    nTotAlgo=3
    try:
        datadict={}
        session.transaction().start(True)
        lumischema=session.nominalSchema()
        summaryQuery=lumischema.newQuery()
        summaryQuery.addToTableList( nameDealer.lumisummaryTableName() )
        summaryQuery.addToOutputList( 'LUMILSNUM','lumilsnum')
        summaryQuery.addToOutputList( 'CMSLSNUM','cmslsnum')
        summaryQuery.addToOutputList( 'INSTLUMI','instlumi')
        summaryQuery.addToOutputList( 'INSTLUMIERROR','instlumierror')
        summaryQuery.addToOutputList( 'INSTLUMIQUALITY','instlumiquality')
        summaryQuery.addToOutputList( 'BEAMSTATUS','beamstatus')
        summaryQuery.addToOutputList( 'BEAMENERGY','beamenergy')        
        summaryQuery.addToOutputList( 'NUMORBIT','numorbit')
        summaryQuery.addToOutputList( 'STARTORBIT','startorbit')
        summaryQuery.addToOutputList( 'CMSBXINDEXBLOB','cmsbxindexblob')
        summaryQuery.addToOutputList( 'BEAMINTENSITYBLOB_1','beamintensityblob_1')
        summaryQuery.addToOutputList( 'BEAMINTENSITYBLOB_2','beamintensityblob_2')
        summaryCondition=coral.AttributeList()
        summaryCondition.extend('runnumber','unsigned int')
        summaryCondition['runnumber'].setData(int(runnumber))
        summaryResult=coral.AttributeList()
        summaryResult.extend('lumilsnum','unsigned int')
        summaryResult.extend('cmslsnum','unsigned int')
        summaryResult.extend('instlumi','float')
        summaryResult.extend('instlumierror','float')
        summaryResult.extend('instlumiquality','short')
        summaryResult.extend('beamstatus','string')
        summaryResult.extend('beamenergy','float')
        summaryResult.extend('numorbit','unsigned int')
        summaryResult.extend('startorbit','unsigned int')
        summaryResult.extend('cmsbxindexblob','blob')
        summaryResult.extend('beamintensityblob_1','blob')
        summaryResult.extend('beamintensityblob_2','blob')
        summaryQuery.defineOutput(summaryResult)
        summaryQuery.addToOrderList('lumilsnum')
        summaryQuery.setCondition('RUNNUM=:runnumber',summaryCondition)
        summarycursor=summaryQuery.execute()
        while summarycursor.next():
            lumilsnum=summarycursor.currentRow()['lumilsnum'].data()
            cmslsnum=summarycursor.currentRow()['cmslsnum'].data()
            instlumi=summarycursor.currentRow()['instlumi'].data()
            instlumierror=summarycursor.currentRow()['instlumierror'].data()
            instlumiquality=summarycursor.currentRow()['instlumiquality'].data()
            (uncalibratedinstlumi,uncalibratedinstlumierror)=uncalibrateLumi(runnumber,instlumi,instlumierror)
            startorbit=summarycursor.currentRow()['startorbit'].data()
            numorbit=summarycursor.currentRow()['numorbit'].data()
            beamenergy=summarycursor.currentRow()['beamenergy'].data()
            beamstatus=summarycursor.currentRow()['beamstatus'].data()
            cmsbxindexblob=None
            if not summarycursor.currentRow()['cmsbxindexblob'].isNull():
                cmsbxindexblob=summarycursor.currentRow()['cmsbxindexblob'].data()
            beamintensityblob_1=None
            if not summarycursor.currentRow()['beamintensityblob_1'].isNull():
                beamintensityblob_1=summarycursor.currentRow()['beamintensityblob_1'].data()
            beamintensityblob_2=None
            if not summarycursor.currentRow()['beamintensityblob_2'].isNull():
                beamintensityblob_2=summarycursor.currentRow()['beamintensityblob_2'].data()
            datadict[lumilsnum]=[cmslsnum,uncalibratedinstlumi,uncalibratedinstlumierror,instlumiquality,beamstatus,beamenergy,startorbit,numorbit,cmsbxindexblob,beamintensityblob_1,beamintensityblob_2]
        del summaryQuery
        #print datadict
        for algoname in ['OCC1','OCC2','ET']:
            detailQuery=lumischema.newQuery()
            detailQuery.addToTableList( nameDealer.lumisummaryTableName(),'s' )
            detailQuery.addToTableList( nameDealer.lumidetailTableName(),'d' )
            detailQuery.addToOutputList('s.LUMILSNUM','lumilsnum' )
            detailQuery.addToOutputList('d.BXLUMIVALUE','bxlumivalue' )
            detailQuery.addToOutputList('d.BXLUMIERROR','bxlumierror' )
            detailQuery.addToOutputList('d.BXLUMIQUALITY','bxlumiquality' )
            detailCondition=coral.AttributeList()
            detailCondition.extend('runnumber','unsigned int')
            detailCondition.extend('algoname','string')
            detailCondition['runnumber'].setData(int(runnumber))
            detailCondition['algoname'].setData(algoname)
            detailResult=coral.AttributeList()
            detailResult.extend('lumilsnum','unsigned int')
            detailResult.extend('bxlumivalue','blob')
            detailResult.extend('bxlumierror','blob')
            detailResult.extend('bxlumiquality','blob')
            detailQuery.defineOutput(detailResult)
            detailQuery.addToOrderList('lumilsnum')
            detailQuery.setCondition('s.RUNNUM=:runnumber AND s.LUMISUMMARY_ID=d.LUMISUMMARY_ID AND d.ALGONAME=:algoname',detailCondition)
            detailcursor=detailQuery.execute()
            while detailcursor.next():
                lumilsnum=detailcursor.currentRow()['lumilsnum'].data()
                bxlumivalue=detailcursor.currentRow()['bxlumivalue'].data()
                bxlumierror=detailcursor.currentRow()['bxlumierror'].data()
                bxlumiquality=detailcursor.currentRow()['bxlumiquality'].data()
                datadict[lumilsnum].extend([bxlumivalue,bxlumierror,bxlumiquality])
            del detailQuery
        session.transaction().commit()
        return datadict
    except :
        raise     
def hltFromOldLumi(session,runnumber):
    '''
    select count(distinct pathname) from hlt where runnum=:runnum
    select cmslsnum,pathname,inputcount,acceptcount,prescale from hlt where runnum=:runnum order by cmslsnum,pathname
    [pathnames,databuffer]
    databuffer: {cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}
    '''
    try:
        databuffer={}
        session.transaction().start(True)
        lumischema=session.nominalSchema()
        npath=0
        qHandle=lumischema.newQuery()
        qHandle.addToTableList( nameDealer.hltTableName() )
        qHandle.addToOutputList('COUNT(DISTINCT PATHNAME)','npath')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnumber))
        qResult=coral.AttributeList()
        qResult.extend('npath','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            npath=cursor.currentRow()['npath'].data()
        del qHandle
        #print 'npath ',npath

        qHandle=lumischema.newQuery()
        qHandle.addToTableList( nameDealer.hltTableName() )
        qHandle.addToOutputList('CMSLSNUM','cmslsnum')
        qHandle.addToOutputList('PATHNAME','pathname')
        qHandle.addToOutputList('INPUTCOUNT','inputcount')
        qHandle.addToOutputList('ACCEPTCOUNT','acceptcount')
        qHandle.addToOutputList('PRESCALE','prescale')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(int(runnumber))
        qResult=coral.AttributeList()
        qResult.extend('cmslsnum','unsigned int')
        qResult.extend('pathname','string')
        qResult.extend('inputcount','unsigned int')
        qResult.extend('acceptcount','unsigned int')
        qResult.extend('prescale','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM=:runnum',qCondition)
        qHandle.addToOrderList('cmslsnum')
        qHandle.addToOrderList('pathname')
        cursor=qHandle.execute()
        pathnameList=[]
        inputcountArray=array.array('l')
        acceptcountArray=array.array('l')
        prescaleArray=array.array('l')
        ipath=0
        pathnHLT_PixelTracksVdMames=''
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            pathname=cursor.currentRow()['pathname'].data()
            ipath+=1
            inputcount=cursor.currentRow()['inputcount'].data()
            acceptcount=cursor.currentRow()['acceptcount'].data()
            prescale=cursor.currentRow()['prescale'].data()
            pathnameList.append(pathname)
            inputcountArray.append(inputcount)
            acceptcountArray.append(acceptcount)
            prescaleArray.append(prescale)
            if ipath==npath:
                if cmslsnum==1:
                    pathnames=','.join(pathnameList)
                inputcountBlob=CommonUtil.packArraytoBlob(inputcountArray)
                acceptcountBlob=CommonUtil.packArraytoBlob(acceptcountArray)
                prescaleBlob=CommonUtil.packArraytoBlob(prescaleArray)
                databuffer[cmslsnum]=[inputcountBlob,acceptcountBlob,prescaleBlob]
                pathnameList=[]
                inputcountArray=array.array('l')
                acceptcountArray=array.array('l')
                prescaleArray=array.array('l')
                ipath=0
        del qHandle
        session.transaction().commit()
        #print 'pathnames ',pathnames
        return [pathnames,databuffer]
    except :
        raise 

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

def trgFromNewGT(session,runnumber):
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
    output: [bitnameclob,{cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}]
    1. select counts,lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr
    2. select counts,lsnr,algobit from cms_gt_mon.gt_mon_trig_algo_view where runnr=:runnumber order by lsnr,algobit
    3. select counts,lsnr,techbit from cms_gt_mon.gt_mon_trig_tech_view where runnr=:runnumber order by lsnr,techbit
    4. select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber order by algo_index
    ## not needed5. select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber order by techtrig_index
    5  select distinct(prescale_index)  from cms_gt_mon.lumi_sections where run_number=:runnumber;
    6. select prescale_factor_algo_000,algo.prescale_factor_algo_001..._127 from cms_gt.gt_run_presc_algo_view where prescale_index=:prescale_index and runnr=:runnumber;
    7. select prescale_factor_tt_000,tech.prescale_factor_tt_001..._63 from cms_gt.gt_run_presc_tech_view where prescale_index=:prescale_index and runnr=:runnumber;
    8. select lumi_section,prescale_index from cms_gt_mon.lumi_sections where run_number=:runnumber order by lumi_section
    
    '''
    result=[]
    deadtimeresult={}#{cmslsnum:deadtime}
    NAlgoBit=128 #0-127
    NTechBit=64  #0-63
    algocount={}#{cmslsnum:[algocounts]}
    techcount={}#{cmslsnum:[techcounts]}
    bitnames=[]
    bitzerocountDict={}
    bitzeroprescaleDict={}
    perlsdict={}#{cmslsnum:[deadtime,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]}
    prescaleDict={} #{prescale_index:[[algo_prescale_factors][tech_prescale_factors]]}
    prescaleResult={}#{lsnumber:[algo_prescale_factors,tech_prescale_factors]}
    try:
        session.transaction().start(True)
        gtmonschema=session.schema('CMS_GT_MON')
        #
        # select counts,lsnr from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter=:countername order by lsnr
        #
        deadviewQuery=gtmonschema.newQuery()
        deadviewQuery.addToTableList('GT_MON_TRIG_DEAD_VIEW')
        deadOutput=coral.AttributeList()
        deadOutput.extend('counts','unsigned int')
        deadOutput.extend('lsnr','unsigned int')
        deadviewQuery.addToOutputList('counts')
        deadviewQuery.addToOutputList('lsnr')
        bindVariablesDead=coral.AttributeList()
        bindVariablesDead.extend('runnumber','int')
        bindVariablesDead.extend('countername','string')
        bindVariablesDead['runnumber'].setData(int(runnumber))
        bindVariablesDead['countername'].setData('DeadtimeBeamActive')
        deadviewQuery.setCondition('RUNNR=:runnumber AND DEADCOUNTER=:countername',bindVariablesDead)
        deadviewQuery.addToOrderList('lsnr')
        deadviewQuery.defineOutput(deadOutput)
        deadcursor=deadviewQuery.execute()
        s=0
        while deadcursor.next():
            row=deadcursor.currentRow()
            s+=1
            lsnr=row['lsnr'].data()
            while s!=lsnr:
                print 'DEADTIME alert: found hole in LS range'
                print '         fill deadtimebeamactive 0 for LS '+str(s)
                deadtimeresult[s]=0
                s+=1
            count=row['counts'].data()
            deadtimeresult[s]=count
        if s==0:
            deadcursor.close()
            del deadviewQuery
            session.transaction().commit()
            raise 'requested run '+str(runnumber )+' does not exist for deadcounts'
        del deadviewQuery
        #print 'deadtimeresult ',deadtimeresult
        mybitcount_algo=[]
        algoviewQuery=gtmonschema.newQuery()
        #
        # select counts,lsnr,algobit from cms_gt_mon.gt_mon_trig_algo_view where runnr=:runnumber order by lsnr,algobit
        #
        algoviewQuery.addToTableList('GT_MON_TRIG_ALGO_VIEW')
        algoOutput=coral.AttributeList()
        algoOutput.extend('counts','unsigned int')
        algoOutput.extend('lsnr','unsigned int')
        algoOutput.extend('algobit','unsigned int')
        algoviewQuery.addToOutputList('counts')
        algoviewQuery.addToOutputList('lsnr')
        algoviewQuery.addToOutputList('algobit')
        algoCondition=coral.AttributeList()
        algoCondition.extend('runnumber','unsigned int')
        algoCondition['runnumber'].setData(int(runnumber))
        algoviewQuery.setCondition('RUNNR=:runnumber',algoCondition)
        algoviewQuery.addToOrderList('lsnr')
        algoviewQuery.addToOrderList('algobit')
        algoviewQuery.defineOutput(algoOutput)
      
        algocursor=algoviewQuery.execute()
        s=0
        while algocursor.next():
            row=algocursor.currentRow()
            lsnr=row['lsnr'].data()
            counts=row['counts'].data()
            algobit=row['algobit'].data()
            mybitcount_algo.append(counts)
            if algobit==0:
                bitzerocountDict[lsnr]=counts
            if algobit==NAlgoBit-1:
                s+=1
                while s!=lsnr:
                    print 'ALGO COUNT alert: found hole in LS range'
                    print '     fill all algocount 0 for LS '+str(s)
                    tmpzero=[0]*NAlgoBit
                    algocount[s]=tmpzero
                    s+=1
                algocount[s]=mybitcount_algo
                mybitcount_algo=[]
        if s==0:
            algocursor.close()
            del algoviewQuery
            session.transaction().commit()
            raise 'requested run '+str(runnumber+' does not exist for algocounts ')
        del algoviewQuery
                
        mybitcount_tech=[]
        techviewQuery=gtmonschema.newQuery()
        techviewQuery.addToTableList('GT_MON_TRIG_TECH_VIEW')
        #
        # select counts,lsnr,techbit from cms_gt_mon.gt_mon_trig_tech_view where runnr=:runnumber order by lsnr,techbit
        #
        techOutput=coral.AttributeList()
        techOutput.extend('counts','unsigned int')
        techOutput.extend('lsnr','unsigned int')
        techOutput.extend('techbit','unsigned int')
        techviewQuery.addToOutputList('COUNTS','counts')
        techviewQuery.addToOutputList('LSNR','lsnr')
        techviewQuery.addToOutputList('TECHBIT','techbit')
        techCondition=coral.AttributeList()
        techCondition.extend('runnumber','unsigned int')
        techCondition['runnumber'].setData(int(runnumber))
        techviewQuery.setCondition('RUNNR=:runnumber',techCondition)
        techviewQuery.addToOrderList('lsnr')
        techviewQuery.addToOrderList('techbit')
        techviewQuery.defineOutput(techOutput)
      
        techcursor=techviewQuery.execute()
        s=0
        while techcursor.next():
            row=techcursor.currentRow()
            lsnr=row['lsnr'].data()
            counts=row['counts'].data()
            techbit=row['techbit'].data()
            mybitcount_tech.append(counts)
            if techbit==NTechBit-1:
                s+=1
                while s!=lsnr:
                    print 'TECH COUNT alert: found hole in LS range'
                    print '     fill all techcount 0 for LS '+str(s)
                    tmpzero=[0]*NTechBit
                    techcount[s]=tmpzero
                    s+=1
                techcount[s]=mybitcount_tech
                mybitcount_tech=[]
        if s==0:
            techcursor.close()
            del techviewQuery
            session.transaction().commit()
            raise 'requested run '+str(runnumber+' does not exist for techcounts ')
        del techviewQuery

        gtschema=session.schema('CMS_GT')
        triggernamemap={}
        namealiasQuery=gtschema.newQuery()
        #
        # select algo_index,alias from cms_gt.gt_run_algo_view where runnumber=:runnumber 
        #
        triggernamemap={}
        namealiasQuery.addToTableList('GT_RUN_ALGO_VIEW')
        algonameOutput=coral.AttributeList()
        algonameOutput.extend('algo_index','unsigned int')
        algonameOutput.extend('alias','string')
        namealiasQuery.addToOutputList('algo_index')
        namealiasQuery.addToOutputList('alias')
        algonameCondition=coral.AttributeList()
        algonameCondition.extend('runnumber','unsigned int')
        algonameCondition['runnumber'].setData(int(runnumber))
        namealiasQuery.setCondition('RUNNUMBER=:runnumber',algonameCondition)
        namealiasQuery.defineOutput(algonameOutput)
        algonamecursor=namealiasQuery.execute()
        while algonamecursor.next():
            row=algonamecursor.currentRow()
            algo_index=row['algo_index'].data()
            algo_name=row['alias'].data()
            triggernamemap[algo_index]=algo_name
        del namealiasQuery

        #techtriggernamemap={}
        #technamealiasQuery=gtschema.newQuery()
        #
        # select techtrig_index,name from cms_gt.gt_run_tech_view where runnumber=:runnumber
        #
        # this queryis unnecessary
        #technamealiasQuery.addToTableList('GT_RUN_TECH_VIEW')
        #technameOutput=coral.AttributeList()
        #technameOutput.extend('techtrig_index','unsigned int')
        #technameOutput.extend('name','string')
        #technamealiasQuery.addToOutputList('techtrig_index')
        #technamealiasQuery.addToOutputList('name')
        #technameCondition=coral.AttributeList()
        #technameCondition.extend('runnumber','unsigned int')
        #technameCondition['runnumber'].setData(int(runnumber))
        #technamealiasQuery.setCondition('RUNNUMBER=:runnumber',technameCondition)
        #technamealiasQuery.defineOutput(technameOutput)
        #technamecursor=technamealiasQuery.execute()
        #while technamecursor.next():
            #row=technamecursor.currentRow()
            #techtrig_index=row['techtrig_index'].data()
            #tech_name=row['name'].data()
            #techtriggernamemap[techtrig_index]=tech_name
        #del technamealiasQuery
        #
        # reprocess Algo name result filling unallocated trigger bit with string "False"
        #
        for algoidx in range(NAlgoBit):
            if algoidx in triggernamemap.keys():
                bitnames.append(triggernamemap[algoidx])
            else:
                bitnames.append('False')
        #
        # reprocess Tech name 
        #
        for techidx in range(NTechBit):
             bitnames.append(str(techidx))
        bitnameclob=','.join(bitnames)     
        #
        # select distinct(prescale_index) from cms_gt_mon.lumi_sections where run_number=:runnumber;
        #
        prescaleidx=[]
        presidxQuery=gtmonschema.newQuery()
        presidxQuery.addToTableList('LUMI_SECTIONS')
        
        presidxBindVariable=coral.AttributeList()
        presidxBindVariable.extend('runnumber','int')
        presidxBindVariable['runnumber'].setData(int(runnumber))

        presidxOutput=coral.AttributeList()
        presidxOutput.extend('prescale_index','int')
        presidxQuery.addToOutputList('distinct(PRESCALE_INDEX)','prescale_index')
        presidxQuery.defineOutput(presidxOutput)
        presidxQuery.setCondition('RUN_NUMBER=:runnumber',presidxBindVariable)
        presidxCursor=presidxQuery.execute()
        while presidxCursor.next():
            presc=presidxCursor.currentRow()['prescale_index'].data()
            prescaleidx.append(presc)
        #print prescaleidx
        del presidxQuery
        #
        # select algo.prescale_factor_algo_000,,algo.prescale_factor_algo_001..._127 from gt_run_presc_algo_view where run_number=:runnumber and prescale_index=:prescale_index;
        #
        for prescaleindex in prescaleidx:
            algoprescQuery=gtschema.newQuery()
            algoprescQuery.addToTableList('GT_RUN_PRESC_ALGO_VIEW')
            algoPrescOutput=coral.AttributeList()
            algoprescBase='PRESCALE_FACTOR_ALGO_'
            for bitidx in range(NAlgoBit):
                algopresc=algoprescBase+str(bitidx).zfill(3)
                algoPrescOutput.extend(algopresc,'unsigned int')
                algoprescQuery.addToOutputList(algopresc)
            PrescbindVariable=coral.AttributeList()
            PrescbindVariable.extend('runnumber','int')
            PrescbindVariable.extend('prescaleindex','int')
            PrescbindVariable['runnumber'].setData(int(runnumber))
            PrescbindVariable['prescaleindex'].setData(prescaleindex)
            algoprescQuery.setCondition('RUNNR=:runnumber AND PRESCALE_INDEX=:prescaleindex',PrescbindVariable)
            algoprescQuery.defineOutput(algoPrescOutput)
            algopresccursor=algoprescQuery.execute()
            while algopresccursor.next():
                row=algopresccursor.currentRow()
                algoprescale=[]
                for bitidx in range(NAlgoBit):
                    algopresc=algoprescBase+str(bitidx).zfill(3)
                    algoprescale.append(row[algopresc].data())
                prescaleDict[prescaleindex]=algoprescale
            del algoprescQuery

        #
        # select prescale_factor_tt_000,prescale_factor_tt_001..._127 from gt_run_presc_tech_view where run_number=:runnumber and prescale_index=:prescale_index;
        #
        for prescaleindex in prescaleidx:
            techprescQuery=gtschema.newQuery()
            techprescQuery.addToTableList('GT_RUN_PRESC_TECH_VIEW')
            techPrescOutput=coral.AttributeList()
            techprescBase='PRESCALE_FACTOR_TT_'
            for bitidx in range(NTechBit):
                techpresc=techprescBase+str(bitidx).zfill(3)
                techPrescOutput.extend(techpresc,'unsigned int')
                techprescQuery.addToOutputList(techpresc)
            PrescbindVariable=coral.AttributeList()
            PrescbindVariable.extend('runnumber','int')
            PrescbindVariable.extend('prescaleindex','int')
            PrescbindVariable['runnumber'].setData(int(runnumber))
            PrescbindVariable['prescaleindex'].setData(prescaleindex)
            techprescQuery.setCondition('RUNNR=:runnumber AND PRESCALE_INDEX=:prescaleindex',PrescbindVariable)
            techprescQuery.defineOutput(techPrescOutput)
            techpresccursor=techprescQuery.execute()
            while techpresccursor.next():
                row=techpresccursor.currentRow()
                techprescale=[]
                for bitidx in range(NTechBit):
                    techpresc=techprescBase+str(bitidx).zfill(3)
                    techprescale.append(row[techpresc].data())
                prescaleDict[prescaleindex]+=techprescale
            del techprescQuery
        #print prescaleDict
        #
        #select lumi_section,prescale_index from cms_gt_mon.lumi_sections where run_number=:runnumber
        #
        lumiprescQuery=gtmonschema.newQuery()
        lumiprescQuery.addToTableList('LUMI_SECTIONS')
        
        lumiprescBindVariable=coral.AttributeList()
        lumiprescBindVariable.extend('runnumber','int')
        lumiprescBindVariable['runnumber'].setData(int(runnumber))

        lumiprescOutput=coral.AttributeList()
        lumiprescOutput.extend('lumisection','int')
        lumiprescOutput.extend('prescale_index','int')
        lumiprescQuery.addToOutputList('LUMI_SECTION')
        lumiprescQuery.addToOutputList('PRESCALE_INDEX')
        lumiprescQuery.defineOutput(lumiprescOutput)
        lumiprescQuery.setCondition('RUN_NUMBER=:runnumber',lumiprescBindVariable)
        lumiprescCursor=lumiprescQuery.execute()
        while lumiprescCursor.next():
            row=lumiprescCursor.currentRow()
            lumisection=row['lumisection'].data()
            psindex=row['prescale_index'].data()
            bitzeroprescale=prescaleDict[psindex][0]
            bitzeroprescaleDict[lumisection]=prescaleDict[psindex][0]
            prescaleResult[lumisection]=prescaleDict[psindex]
        #print prescaleResult
        del lumiprescQuery
        #return result
        session.transaction().commit()
        #
        #assemble result
        #
        for cmslsnum,deadcount in deadtimeresult.items():
            bitzerocount=bitzerocountDict[cmslsnum]
            bitzeroprescale=bitzeroprescaleDict[cmslsnum]
            trgcounts=array.array('l')
            for acounts in algocount[cmslsnum]:
                trgcounts.append(acounts)
            for tcounts in techcount[cmslsnum]:
                trgcounts.append(tcounts)
            trgcountBlob=CommonUtil.packArraytoBlob(trgcounts)
            trgprescale=array.array('l')
            for aprescale in prescaleResult[cmslsnum]:
                trgprescale.append(aprescale)
            trgprescaleBlob=CommonUtil.packArraytoBlob(trgprescale)
            perlsdict[cmslsnum]=[deadcount,bitzerocount,bitzeroprescale,trgcountBlob,trgprescaleBlob]
        result=[bitnameclob,perlsdict]
        return result
    except:
        session.transaction().rollback()
        del session
        raise
    
def hltFromRuninfoV2(session,runnumber):
    '''
    input:
    output: [pathnameclob,{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}]
    select count(distinct PATHNAME) as npath from HLT_SUPERVISOR_LUMISECTIONS_V2 where runnr=:runnumber and lsnumber=1;
    select l.pathname,l.lsnumber,l.l1pass,l.paccept,m.psvalue from hlt_supervisor_lumisections_v2 l,hlt_supervisor_scalar_map m where l.runnr=m.runnr and l.psindex=m.psindex and l.pathname=m.pathname and l.runnr=:runnumber order by l.lsnumber
    '''
    npaths=0
    pathnames=[]
    hltdict={}
    try:
        session.transaction().start(True)
        hltschema=session.schema('CMS_RUNINFO')
        bvar=coral.AttributeList()
        bvar.extend('runnumber','unsigned int')
        bvar.extend('lsnumber','unsigned int')
        bvar['runnumber'].setData(int(runnumber))
        bvar['lsnumber'].setData(1)
        q1=hltschema.newQuery()
        q1.addToTableList('HLT_SUPERVISOR_LUMISECTIONS_V2')
        nls=coral.AttributeList()
        nls.extend('npath','unsigned int')
        q1.addToOutputList('count(distinct PATHNAME)','npath')
        q1.setCondition('RUNNR=:runnumber AND LSNUMBER=:lsnumber',bvar)
        q1.defineOutput(nls)
        c=q1.execute()
        while c.next():
            npath=c.currentRow()['npath'].data()
        del q1
        if npath==0:
            print 'request run is empty, do nothing'
            
        q=hltschema.newQuery()
        bindVar=coral.AttributeList()
        bindVar.extend('runnumber','unsigned int')
        bindVar['runnumber'].setData(int(runnumber))
        q.addToTableList('HLT_SUPERVISOR_LUMISECTIONS_V2','l')
        q.addToTableList('HLT_SUPERVISOR_SCALAR_MAP','m')
        q.addToOutputList('l.LSNUMBER','lsnumber')
        q.addToOutputList('l.PATHNAME','pathname')
        q.addToOutputList('l.L1PASS','hltinput')
        q.addToOutputList('l.PACCEPT','hltaccept')
        q.addToOutputList('m.PSVALUE','prescale')
        q.setCondition('l.RUNNR=m.RUNNR and l.PSINDEX=m.PSINDEX and l.PATHNAME=m.PATHNAME and l.RUNNR=:runnumber',bindVar)
        q.addToOrderList('l.LSNUMBER')
        q.addToOrderList('l.PATHNAME')
        q.setRowCacheSize(10692)
        cursor=q.execute()
        lastLumiSection=1
        currentLumiSection=0
        allpaths=[]
        ipath=0
        hltinputs=array.array('l')
        hltaccepts=array.array('l')
        prescales=array.array('l')
        while cursor.next():
            row=cursor.currentRow()
            cmsluminr=row['lsnumber'].data()
            hltinput=row['hltinput'].data()
            hltaccept=row['hltaccept'].data()
            prescale=row['prescale'].data()
            pathname=row['pathname'].data()
            ipath+=1
            if cmsluminr==1:
                pathnames.append(pathname)
            if not hltdict.has_key(cmsluminr):
                hltdict[cmsluminr]=[]
            hltinputs.append(hltinput)
            hltaccepts.append(hltaccept)
            prescales.append(prescale)
            if ipath==npath:
                #pack
                #print 'packing hltinputs ',hltinputs
                hltinputsBlob=CommonUtil.packArraytoBlob(hltinputs)
                #print 'packing hltaccepts ',hltaccepts
                hltacceptsBlob=CommonUtil.packArraytoBlob(hltaccepts)
                #print 'packing prescales ',prescales
                prescalesBlob=CommonUtil.packArraytoBlob(prescales)
                hltdict[cmsluminr].extend([hltinputsBlob,hltacceptsBlob,hltacceptsBlob])
                ipath=0
                hltinputs=array.array('l')
                hltaccepts=array.array('l')
                prescales=array.array('l')
        pathnameclob=','.join(pathnames)  
        del q
        session.transaction().commit()
        return [pathnameclob,hltdict]
    except:
        raise

def hltFromRuninfoV3(session,runnumber):
    '''
    input:
    output: [datasource,pathnameclob,{cmslsnum:[inputcountBlob,acceptcountBlob,prescaleBlob]}]
    select distinct(pathid) from HLT_SUPERVISOR_TRIGGERPATHS where runnnumber=:runnumber;
    select count(*) from HLT_SUPERVISOR_LUMISECTIONS_V3 where runnumber=:runnumber;//total ls
    select tr.runnumber,tr.lsnumber,tr.pathid,tr.l1pass,tr.paccept,ls.psindex,sm.psvalue from hlt_supervisor_triggerpaths tr,hlt_supervisor_lumisections_v3 ls,hlt_supervisor_scalar_map_v2 sm where tr.runnumber=ls.runnumber and tr.lsnumber=ls.lsnumber and sm.runnumber=tr.runnumber and sm.pathid=tr.pathid and sm.psindex=ls.psindex and tr.runnumber=:runnumber order by tr.lsnumber;
    loop :
          select pathname from cms_hlt.paths where pathid=:pathid
    '''
    pass

def hltconf(session,hltkey):
    '''
    select paths.pathid,paths.name,stringparamvalues.value from stringparamvalues,paths,parameters,superidparameterassoc,modules,moduletemplates,pathmoduleassoc,configurationpathassoc,configurations where parameters.paramid=stringparamvalues.paramid and  superidparameterassoc.paramid=parameters.paramid and modules.superid=superidparameterassoc.superid and moduletemplates.superid=modules.templateid and pathmoduleassoc.moduleid=modules.superid and paths.pathid=pathmoduleassoc.pathid and configurationpathassoc.pathid=paths.pathid and configurations.configid=configurationpathassoc.configid and moduletemplates.name='HLTLevel1GTSeed' and parameters.name='L1SeedsLogicalExpression' and configurations.configdescriptor=:hlt_description;
    select paths.pathid,paths.name,stringparamvalues.value from stringparamvalues,paths,parameters,superidparameterassoc,modules,moduletemplates,pathmoduleassoc,configurationpathassoc,configurations where parameters.paramid=stringparamvalues.paramid and  superidparameterassoc.paramid=parameters.paramid and modules.superid=superidparameterassoc.superid and moduletemplates.superid=modules.templateid and pathmoduleassoc.moduleid=modules.superid and paths.pathid=pathmoduleassoc.pathid and configurationpathassoc.pathid=paths.pathid and configurations.configid=configurationpathassoc.configid and moduletemplates.name='HLTLevel1GTSeed' and parameters.name='L1SeedsLogicalExpression' and configurations.configid=:hlt_numkey;
    ##select paths.pathid from cms_hlt.paths paths,cms_hlt.configurations config where config.configdescriptor=' ' and name=:pathname
    '''
    try:
        session.transaction().start(True)
        hltconfschema=session.nominalSchema()
        hltconfQuery=hltconfschema.newQuery()

        hltconfQuery.addToOutputList('PATHS.NAME','hltpath')
        hltconfQuery.addToOutputList('STRINGPARAMVALUES.VALUE','l1expression')
                
        hltconfQuery.addToTableList('PATHS')
        hltconfQuery.addToTableList('STRINGPARAMVALUES')
        hltconfQuery.addToTableList('PARAMETERS')
        hltconfQuery.addToTableList('SUPERIDPARAMETERASSOC')
        hltconfQuery.addToTableList('MODULES')
        hltconfQuery.addToTableList('MODULETEMPLATES')
        hltconfQuery.addToTableList('PATHMODULEASSOC')
        hltconfQuery.addToTableList('CONFIGURATIONPATHASSOC')
        hltconfQuery.addToTableList('CONFIGURATIONS')

        hltconfBindVar=coral.AttributeList()
        hltconfBindVar.extend('hltseed','string')
        hltconfBindVar.extend('l1seedexpr','string')
        hltconfBindVar.extend('hltkey','string')
        hltconfBindVar['hltseed'].setData('HLTLevel1GTSeed')
        hltconfBindVar['l1seedexpr'].setData('L1SeedsLogicalExpression')
        hltconfBindVar['hltkey'].setData(hltkey)
        hltconfQuery.setCondition('PARAMETERS.PARAMID=STRINGPARAMVALUES.PARAMID AND SUPERIDPARAMETERASSOC.PARAMID=PARAMETERS.PARAMID AND MODULES.SUPERID=SUPERIDPARAMETERASSOC.SUPERID AND MODULETEMPLATES.SUPERID=MODULES.TEMPLATEID AND PATHMODULEASSOC.MODULEID=MODULES.SUPERID AND PATHS.PATHID=PATHMODULEASSOC.PATHID AND CONFIGURATIONPATHASSOC.PATHID=PATHS.PATHID AND CONFIGURATIONS.CONFIGID=CONFIGURATIONPATHASSOC.CONFIGID AND MODULETEMPLATES.NAME=:hltseed AND PARAMETERS.NAME=:l1seedexpr AND CONFIGURATIONS.CONFIGDESCRIPTOR=:hltkey',hltconfBindVar)
        hlt2l1map={}
        cursor=hltconfQuery.execute()
        while cursor.next():
            hltpath=cursor.currentRow()['hltpath'].data()
            print hltpath
            l1expression=cursor.currentRow()['l1expression'].data()
            hlt2l1map[hltpath]=l1expression
        del hltconfQuery
        session.transaction().commit()
        return hlt2l1map
    except:
        raise

def runsummary(session,schemaname,runnumber,complementalOnly=False):
    '''
    x select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.TRG:TSC_KEY';
    x select distinct(string_value) from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.SCAL:AMODEtag'
    x select distinct(string_value),session_id from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.SCAL:EGEV' order by SESSION_ID
    
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:SEQ_NAME'
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:HLT_KEY_DESCRIPTION';
    select string_value from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.SCAL:FILLN' and rownum<=1;
    select time from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:START_TIME_T';
    select time from cms_runinfo.runsession_parameter where runnumber=:runnumber and name='CMS.LVL0:STOP_TIME_T';
    input:
    output:[l1key,amodetag,egev,sequence,hltkey,fillnum,starttime,stoptime]
    if complementalOnly:
       [l1key,amodetag,egev]
    '''
    runsessionparameterTable=''
    result=[]
    l1key=''
    amodetag=''
    egev=''
    hltkey=''
    fillnum=''
    sequence=''
    starttime=''
    stoptime=''
    try:
        session.transaction().start(True)
        runinfoschema=session.schema(schemaname)
        l1keyQuery=runinfoschema.newQuery()
        l1keyQuery.addToTableList('RUNSESSION_PARAMETER')
        l1keyOutput=coral.AttributeList()
        l1keyOutput.extend('l1key','string')
        l1keyCondition=coral.AttributeList()
        l1keyCondition.extend('name','string')
        l1keyCondition.extend('runnumber','unsigned int')
        l1keyCondition['name'].setData('CMS.TRG:TSC_KEY')
        l1keyCondition['runnumber'].setData(int(runnumber))
        l1keyQuery.addToOutputList('STRING_VALUE')
        l1keyQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',l1keyCondition)
        l1keyQuery.defineOutput(l1keyOutput)
        cursor=l1keyQuery.execute()
        while cursor.next():
            l1key=cursor.currentRow()['l1key'].data()
        del l1keyQuery
        result.append(l1key)
        
        amodetagQuery=runinfoschema.newQuery()
        amodetagQuery.addToTableList('RUNSESSION_PARAMETER')
        amodetagOutput=coral.AttributeList()
        amodetagOutput.extend('amodetag','string')
        amodetagCondition=coral.AttributeList()
        amodetagCondition.extend('name','string')
        amodetagCondition.extend('runnumber','unsigned int')
        amodetagCondition['name'].setData('CMS.SCAL:AMODEtag')
        amodetagCondition['runnumber'].setData(int(runnumber))
        amodetagQuery.addToOutputList('distinct(STRING_VALUE)')
        amodetagQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',amodetagCondition)
        amodetagQuery.limitReturnedRows(1)
        amodetagQuery.defineOutput(amodetagOutput)
        cursor=amodetagQuery.execute()
        while cursor.next():
            amodetag=cursor.currentRow()['amodetag'].data()
        del amodetagQuery
        result.append(amodetag)
        
        egevQuery=runinfoschema.newQuery()
        egevQuery.addToTableList('RUNSESSION_PARAMETER')
        egevOutput=coral.AttributeList()
        egevOutput.extend('egev','string')
        egevCondition=coral.AttributeList()
        egevCondition.extend('name','string')
        egevCondition.extend('runnumber','unsigned int')
        egevCondition['name'].setData('CMS.SCAL:EGEV')
        egevCondition['runnumber'].setData(int(runnumber))
        egevQuery.addToOutputList('distinct(STRING_VALUE)')
        egevQuery.addToOutputList('SESSION_ID')
        egevQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',egevCondition)
        egevQuery.defineOutput(egevOutput)
        egevQuery.addToOrderList('SESSION_ID')
        cursor=egevQuery.execute()
        while cursor.next():
            egev=cursor.currentRow()['egev'].data()
        del egevQuery
        result.append(egev)
        
        if not complementalOnly:
            seqQuery=runinfoschema.newQuery()
            seqQuery.addToTableList('RUNSESSION_PARAMETER')
            seqOutput=coral.AttributeList()
            seqOutput.extend('seq','string')
            seqCondition=coral.AttributeList()
            seqCondition.extend('name','string')
            seqCondition.extend('runnumber','unsigned int')
            seqCondition['name'].setData('CMS.LVL0:SEQ_NAME')
            seqCondition['runnumber'].setData(int(runnumber))
            seqQuery.addToOutputList('STRING_VALUE')
            seqQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',seqCondition)
            seqQuery.defineOutput(seqOutput)
            cursor=seqQuery.execute()
            while cursor.next():
                sequence=cursor.currentRow()['seq'].data()
            del seqQuery
            result.append(sequence)

            hltkeyQuery=runinfoschema.newQuery()
            hltkeyQuery.addToTableList('RUNSESSION_PARAMETER')
            hltkeyOutput=coral.AttributeList()
            hltkeyOutput.extend('hltkey','string')
            hltkeyCondition=coral.AttributeList()
            hltkeyCondition.extend('name','string')
            hltkeyCondition.extend('runnumber','unsigned int')
            hltkeyCondition['name'].setData('CMS.LVL0:HLT_KEY_DESCRIPTION')
            hltkeyCondition['runnumber'].setData(int(runnumber))
            hltkeyQuery.addToOutputList('STRING_VALUE')
            hltkeyQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',hltkeyCondition)
            #hltkeyQuery.limitReturnedRows(1)
            hltkeyQuery.defineOutput(hltkeyOutput)
            cursor=hltkeyQuery.execute()
            while cursor.next():
                hltkey=cursor.currentRow()['hltkey'].data()
                del hltkeyQuery
            result.append(hltkey)

            fillnumQuery=runinfoschema.newQuery()
            fillnumQuery.addToTableList('RUNSESSION_PARAMETER')
            fillnumOutput=coral.AttributeList()
            fillnumOutput.extend('fillnum','string')
            fillnumCondition=coral.AttributeList()
            fillnumCondition.extend('name','string')
            fillnumCondition.extend('runnumber','unsigned int')
            fillnumCondition['name'].setData('CMS.SCAL:FILLN')
            fillnumCondition['runnumber'].setData(int(runnumber))
            fillnumQuery.addToOutputList('STRING_VALUE')
            fillnumQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',fillnumCondition)
            fillnumQuery.limitReturnedRows(1)
            fillnumQuery.defineOutput(fillnumOutput)
            cursor=fillnumQuery.execute()
            while cursor.next():
                fillnum=cursor.currentRow()['fillnum'].data()
            del fillnumQuery
            result.append(fillnum)

            starttimeQuery=runinfoschema.newQuery()
            starttimeQuery.addToTableList('RUNSESSION_PARAMETER')
            starttimeOutput=coral.AttributeList()
            starttimeOutput.extend('starttime','time stamp')
            starttimeCondition=coral.AttributeList()
            starttimeCondition.extend('name','string')
            starttimeCondition.extend('runnumber','unsigned int')
            starttimeCondition['name'].setData('CMS.LVL0:START_TIME_T')
            starttimeCondition['runnumber'].setData(int(runnumber))
            starttimeQuery.addToOutputList('TIME')
            starttimeQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',starttimeCondition)
            starttimeQuery.defineOutput(starttimeOutput)
            cursor=starttimeQuery.execute()
            while cursor.next():
                starttime=cursor.currentRow()['starttime'].data()
            del starttimeQuery
            result.append(starttime)

            stoptimeQuery=runinfoschema.newQuery()
            stoptimeQuery.addToTableList('RUNSESSION_PARAMETER')
            stoptimeOutput=coral.AttributeList()
            stoptimeOutput.extend('stoptime','time stamp')
            stoptimeCondition=coral.AttributeList()
            stoptimeCondition.extend('name','string')
            stoptimeCondition.extend('runnumber','unsigned int')
            stoptimeCondition['name'].setData('CMS.LVL0:STOP_TIME_T')
            stoptimeCondition['runnumber'].setData(int(runnumber))
            stoptimeQuery.addToOutputList('TIME')
            stoptimeQuery.setCondition('NAME=:name AND RUNNUMBER=:runnumber',stoptimeCondition)
            stoptimeQuery.defineOutput(stoptimeOutput)
            cursor=stoptimeQuery.execute()
            while cursor.next():
                stoptime=cursor.currentRow()['stoptime'].data()
            del stoptimeQuery
            result.append(stoptime)
            session.transaction().commit()
        else:
            session.transaction().commit()
        return result
    except:
        raise
    
if __name__ == "__main__":
    from RecoLuminosity.LumiDB import sessionManager
    #svc=sessionManager.sessionManager('oracle://cms_orcoff_prep/cms_lumi_dev_offline',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    #session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    #lsresult=trgFromOldLumi(session,135735)
    #print lsresult
    #lshltresult=hltFromOldLumi(session,135735)
    #print lshltresult
    #svc=sessionManager.sessionManager('oracle://cms_orcoff_prod/cms_runinfo',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    #session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    #runsummary(session,'CMS_RUNINFO',135735,complementalOnly=True)
    #svc=sessionManager.sessionManager('oracle://cms_orcoff_prod/cms_gt',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    #session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    #print trgFromOldGT(session,135735)
    #svc=sessionManager.sessionManager('oracle://cms_orcoff_prod/cms_runinfo',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    #session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    #print hltFromRuninfoV2(session,135735)
    svc=sessionManager.sessionManager('oracle://cms_orcoff_prod/cms_lumi_prod',authpath='/afs/cern.ch/user/x/xiezhen',debugON=False)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    print uncalibratedlumiFromOldLumi(session,135735)
    del session

import coral
from RecoLuminosity.LumiDB import sessionManager,dbUtil,nameDealer,dataDML
'''
select tt, algo bit masks and pack them into 3 unsigned long long:
algomask_hi, algomask_lo,ttmask

select gt_rs_key,run_number from cms_gt_mon.global_runs where run_number>=132440;

select tt.finor_tt_*, from cms_gt.gt_partition_finor_tt tt,cms_gt.gt_run_settings r where tt.id=r.finor_tt_fk and r.id=:gt_rs_key ;

select algo.finor_algo_*, from cms_gt.gt_partition_finor_algo algo,cms_gt.gt_run_settings r where algo.id=r.finor_algo_fk and r.id=:gt_rs_key;

output:{run:[algomask_hi,algomask_lo,ttmask]}

algomask_high,      algomask_low
127,126,.....,64,;  63,62,0
ttmask
63,62,...0
'''

def updatedb(schema,runkeymap,keymaskmap):
    '''
    update trgdata set algomask_h=:algomask_h,algomask_l=:algomask_l,techmask=:techmask where runnum=:runnum
    input:
       runkeymap 
       keymaskmap
    '''
    setClause='ALGOMASK_H=:algomask_h,ALGOMASK_L=:algomask_l,TECHMASK=:techmask'
    updateCondition='RUNNUM=:runnum'
    inputData=coral.AttributeList()
    inputData.extend('algomask_h','unsigned long long')
    inputData.extend('algomask_l','unsigned long long')
    inputData.extend('techmask','unsigned long long')
    inputData.extend('runnum','unsigned int')
    db=dbUtil.dbUtil(schema)
    for runnum in runkeymap.keys():
        gt_rs_key=runkeymap[runnum]
        print runnum,gt_rs_key
        [algo_h,algo_l,tech]=keymaskmap[gt_rs_key]
        inputData['algomask_h'].setData(algo_h)
        inputData['algomask_l'].setData(algo_l)
        inputData['techmask'].setData(tech)
        inputData['runnum'].setData(runnum)
        r=db.singleUpdate(nameDealer.trgdataTableName(),setClause,updateCondition,inputData)
        if r>0:
            print 'updated'
if __name__ == '__main__':
    #pth='/afs/cern.ch/user/l/lumipro/'
    pth='/nfshome0/xiezhen/authwriter'
    #sourcestr='oracle://cms_orcon_adg/cms_gt'
    sourcestr='oracle://cms_omds_lb/cms_gt'
    sourcesvc=sessionManager.sessionManager(sourcestr,authpath=pth,debugON=False)
    sourcesession=sourcesvc.openSession(isReadOnly=True,cpp2sqltype=[('short','NUMBER(1)'),('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    
    #deststr='oracle://cms_orcoff_prep/cms_lumi_dev_offline'
    deststr='oracle://cms_orcon_prod/cms_lumi_prod'
    destsvc=sessionManager.sessionManager(deststr,authpath=pth,debugON=False)
    destsession=destsvc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    
    sourcesession.transaction().start(True)
    gtschema=sourcesession.schema('CMS_GT')
    gtmonschema=sourcesession.schema('CMS_GT_MON')
    runkeymap={}#{run:gt_rs_key}
    runkeyquery=gtmonschema.newQuery()
    minrun=211001
    try:
        runkeyquery.addToTableList('GLOBAL_RUNS')
        qCondition=coral.AttributeList()
        qCondition.extend('runnum','unsigned int')
        qCondition['runnum'].setData(minrun)
        runkeyquery.addToOutputList('GT_RS_KEY')
        runkeyquery.addToOutputList('RUN_NUMBER')
        qResult=coral.AttributeList()
        qResult.extend('GT_RS_KEY','string')
        qResult.extend('RUN_NUMBER','unsigned int')
        runkeyquery.defineOutput(qResult)
        runkeyquery.setCondition('RUN_NUMBER>=:runnum',qCondition)
        cursor=runkeyquery.execute()
        while cursor.next():
            gtrskey=cursor.currentRow()['GT_RS_KEY'].data()
            runnum=cursor.currentRow()['RUN_NUMBER'].data()
            runkeymap[runnum]=gtrskey
        del runkeyquery
    except:
        if runkeyquery:del runkeyquery
        raise
    uniquegtkeys=set(runkeymap.values())
    testkey=runkeymap[minrun]
    print 'testkey ',testkey
    keymaskmap={}#{gtkey:[algomask_hi,algomask_lo,ttmask]}
    ttTab='GT_PARTITION_FINOR_TT'
    algoTab='GT_PARTITION_FINOR_ALGO'
    runsetTab='GT_RUN_SETTINGS'        
    try:
        for k in uniquegtkeys:
            algomask_hi=0
            algomask_lo=0
            ttmask=0
            keymaskmap[k]=[algomask_hi,algomask_lo,ttmask]
            
            ttquery=gtschema.newQuery()
            ttquery.addToTableList(ttTab)
            ttquery.addToTableList(runsetTab)
            ttResult=coral.AttributeList()
            for i in range(0,64):
                ttquery.addToOutputList(ttTab+'.FINOR_TT_%03d'%(i),'tt_%03d'%(i))
                ttResult.extend('tt_%03d'%(i),'short')
            ttConditionStr=ttTab+'.ID='+runsetTab+'.FINOR_TT_FK AND '+runsetTab+'.ID=:gt_rs_key'
            ttCondition=coral.AttributeList()
            ttCondition.extend('gt_rs_key','string')
            ttCondition['gt_rs_key'].setData(k)
            ttquery.defineOutput(ttResult)
            ttquery.setCondition(ttConditionStr,ttCondition)
            cursor=ttquery.execute()
            while cursor.next():
                for ttidx in range(0,64):
                    kvalue=cursor.currentRow()['tt_%03d'%(ttidx)].data()
                    if kvalue!=0:
                        ttdefaultval=keymaskmap[k][2]
                        keymaskmap[k][2]=ttdefaultval|1<<ttidx
            del ttquery
            
            algoquery=gtschema.newQuery()
            algoquery.addToTableList(algoTab)
            algoquery.addToTableList(runsetTab)
            algoResult=coral.AttributeList()
            for i in range(0,128):
                algoquery.addToOutputList(algoTab+'.FINOR_ALGO_%03d'%(i),'algo_%03d'%(i))
                algoResult.extend('algo_%03d'%(i),'short')
            algoConditionStr=algoTab+'.ID='+runsetTab+'.FINOR_ALGO_FK AND '+runsetTab+'.ID=:gt_rs_key'
            algoCondition=coral.AttributeList()
            algoCondition.extend('gt_rs_key','string')
            algoCondition['gt_rs_key'].setData(k)
            algoquery.defineOutput(algoResult)
            algoquery.setCondition(algoConditionStr,algoCondition)
            cursor=algoquery.execute()
            while cursor.next():
                for algoidx in range(0,128):
                    kvalue=cursor.currentRow()['algo_%03d'%(algoidx)].data()
                    if kvalue!=0:
                        if algoidx<64:
                            #if k==testkey:
                            #    print algoidx,kvalue
                            algodefaultval=keymaskmap[k][1]#low 63-0
                            keymaskmap[k][1]=algodefaultval|1<<algoidx
                        else:
                            if k==testkey:
                                print algoidx,(algoidx-64),kvalue
                            algodefaultval=keymaskmap[k][0]#high 127-64
                            keymaskmap[k][0]=algodefaultval|1<<(algoidx-64)
            del algoquery
        
        ahi=keymaskmap[testkey][0]
        print 'algo 84 ',ahi>>(84-64)&1
        print 'algo 126 ',ahi>>(126-64)&1
        alo=keymaskmap[testkey][1]
        print 'algo 0 ',alo>>0&1
    except:
        raise
    sourcesession.transaction().commit()
    
    #destsession.transaction().start(False)
    #updatedb(destsession.nominalSchema(),runkeymap,keymaskmap)
    #destsession.transaction().commit()

    destsession.transaction().start(True)
    gt_rsKey=runkeymap[minrun]
    trgrundata=dataDML.trgRunById(destsession.nominalSchema(),2379)
    print trgrundata
    algomask_h=trgrundata[4]
    algomask_l=trgrundata[5]
    print 'dest algo 84 ',algomask_h>>(84-64)&1
    print 'dest algo 126 ',algomask_h>>(126-64)&1
    print 'dest algo 0 ',algomask_l>>0&1
    destsession.transaction().commit()
    

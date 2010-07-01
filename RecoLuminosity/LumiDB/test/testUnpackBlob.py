import coral
import os,binascii
import array
class constants(object):
    def __init__(self):
        self.debug=False
        self.nbx=3564
        self.lumischema='CMS_LUMI_PROD'
        self.lumidb='oracle://cms_orcoff_prod/cms_lumi_prod'
        self.lumisummaryname='LUMISUMMARY'
        self.lumidetailname='LUMIDETAIL'
        
def detailForRun(dbsession,c,runnum,algos=['OCC1']):
    '''select 
    s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname from LUMIDETAIL d,LUMISUMMARY s where s.runnum=133885 and d.algoname='OCC1' and s.lumisummary_id=d.lumisummary_id order by s.cmslsnum
    '''
    try:
        c=constants()
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.lumischema)
        if not schema:
            raise 'cannot connect to schema ',c.lumischema
        detailOutput=coral.AttributeList()
        detailOutput.extend('cmslsnum','unsigned int')
        detailOutput.extend('bxlumivalue','blob')
        detailOutput.extend('bxlumierror','blob')
        detailOutput.extend('bxlumiquality','blob')
        detailOutput.extend('algoname','string')

        detailCondition=coral.AttributeList()
        detailCondition.extend('runnum','unsigned int')
        detailCondition.extend('algoname','string')
        detailCondition['runnum'].setData(runnum)
        detailCondition['algoname'].setData(algos[0])
        query=schema.newQuery()
        query.addToTableList(c.lumisummaryname,'s')
        query.addToTableList(c.lumidetailname,'d')
        query.addToOutputList('s.CMSLSNUM','cmslsnum')
        query.addToOutputList('d.BXLUMIVALUE','bxlumivalue')
        query.addToOutputList('d.BXLUMIERROR','bxlumierror')
        query.addToOutputList('d.BXLUMIQUALITY','bxlumiquality')
        query.addToOutputList('d.ALGONAME','algoname')
        query.setCondition('s.RUNNUM=:runnum and d.ALGONAME=:algoname and s.LUMISUMMARY_ID=d.LUMISUMMARY_ID',detailCondition)
        query.addToOrderList('s.CMSLSNUM')
        query.defineOutput(detailOutput)
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            algoname=cursor.currentRow()['algoname'].data()
            bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
            print 'cmslsnum , algoname ',cmslsnum,algoname
            #print 'bxlumivalue starting address ',bxlumivalue.startingAddress()
            #bxlumivalue float[3564]
            print 'bxlumivalue size ',bxlumivalue.size()
            #
            #convert bxlumivalue to byte string, then unpack??
            #binascii.hexlify(bxlumivalue.readline()) 
            #
            a=array.array('f')
            a.fromstring(bxlumivalue.readline())
            realvalue=a.tolist()
            print len(realvalue)
            #print realvalue
        del query
        dbsession.transaction().commit()
        
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession

def main():
    c=constants()
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/cms/DB/lumi'
    svc=coral.ConnectionService()
    session=svc.connect(c.lumidb,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    runnum=136066
    detailForRun(session,c,runnum)
    
if __name__=='__main__':
    main()

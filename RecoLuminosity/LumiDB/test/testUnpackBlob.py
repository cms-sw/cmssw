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
        
def beamintensityForRun(dbsession,c,runnum):
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.lumischema)
        if not schema:
            raise 'cannot connect to schema ',c.lumischema
        myOutput=coral.AttributeList()
        myOutput.extend('cmslsnum','unsigned int')
        myOutput.extend('bxindexBlob','blob')
        myOutput.extend('beam1intensityBlob','blob')
        myOutput.extend('beam2intensityBlob','blob')
        myCondition=coral.AttributeList()
        myCondition.extend('runnum','unsigned int')
        myCondition['runnum'].setData(runnum)
        
        query=schema.newQuery()
        query.addToTableList(c.lumisummaryname)
        query.addToOutputList('CMSLSNUM','cmslsnum')
        query.addToOutputList('CMSBXINDEXBLOB','bxindexBlob')
        query.addToOutputList('BEAMINTENSITYBLOB_1','beam1intensityBlob')
        query.addToOutputList('BEAMINTENSITYBLOB_2','beam2intensityBlob')

        query.setCondition('RUNNUM=:runnum',myCondition)
        query.addToOrderList('CMSLSNUM')
        query.defineOutput(myOutput)
        cursor=query.execute()
        
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            bxindex=cursor.currentRow()['bxindexBlob'].data()
            beam1intensity=cursor.currentRow()['beam1intensityBlob'].data()
            beam2intensity=cursor.currentRow()['beam2intensityBlob'].data()
            if bxindex is None:
                print 'bxindex is None',cmslsnum
            if bxindex.size() ==0:
                print 'bxindex is 0',cmslsnum
            if beam1intensity is None:
                print 'beam1intensity is None',cmslsnum
            if beam1intensity.size() ==0:
                print 'beam1intensity is 0',cmslsnum
            if beam2intensity is None:
                print 'beam2intensity is None',cmslsnum
            if beam2intensity.size() ==0:
                print 'beam2intensity is 0',cmslsnum
            if cmslsnum!=0 and bxindex.size()!=0 and beam1intensity.size()!=0 and beam2intensity.size()!=0:                
                bxidxarray=array.array('h')
                if bxindex.readline!='':
                    bxidxarray.fromstring(bxindex.readline())
                    beam1intensityarray=array.array('f')
                    beam1intensityarray.fromstring(beam1intensity.readline())
                    beam2intensityarray=array.array('f')
                    beam2intensityarray.fromstring(beam2intensity.readline())
                    print 'cmslsnum,arraypos,bxidx,beam1intensity,beam2intensity'
                    for pos,bxidx in  enumerate(bxidxarray):
                        print '%4d,%4d,%4d,%.3e,%.3e'%(cmslsnum,pos,bxidx,beam1intensityarray[pos],beam2intensityarray[pos])
        del query
        dbsession.transaction().commit()
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession    
def detailForRun(dbsession,c,runnum,algos=['OCC1']):
    '''select 
    s.cmslsnum,d.bxlumivalue,d.bxlumierror,d.bxlumiquality,d.algoname from LUMIDETAIL d,LUMISUMMARY s where s.runnum=133885 and d.algoname='OCC1' and s.lumisummary_id=d.lumisummary_id order by s.startorbit,s.cmslsnum
    '''
    try:
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
        query.addToOrderList('s.STARTORBIT')
        query.addToOrderList('s.CMSLSNUM')
        query.defineOutput(detailOutput)
        cursor=query.execute()
        
        while cursor.next():
            cmslsnum=cursor.currentRow()['cmslsnum'].data()
            algoname=cursor.currentRow()['algoname'].data()
            bxlumivalue=cursor.currentRow()['bxlumivalue'].data()
            print 'cmslsnum , algoname'
            print cmslsnum,algoname
            print '===='
            #print 'bxlumivalue starting address ',bxlumivalue.startingAddress()
            #bxlumivalue float[3564]
            #print 'bxlumivalue size ',bxlumivalue.size()
            #
            #convert bxlumivalue to byte string, then unpack??
            #binascii.hexlify(bxlumivalue.readline()) 
            #
            a=array.array('f')
            a.fromstring(bxlumivalue.readline())
            print '   bxindex, bxlumivalue'
            if cmslsnum!=95:
                continue
            for index,lum in enumerate(a):
                print "  %4d,%.3e"%(index,lum)
            #realvalue=a.tolist()
            #print len(realvalue)
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
<<<<<<< testUnpackBlob.py
    runnum=149294
=======
    runnum=149011
>>>>>>> 1.6
    ##here arg 4 is default to ['OCC1'], if you want to see all the algorithms do
    ##  detailForRun(session,c,runnum,['OCC1','OCC2','ET']) then modify detailForRun adding an outer loop on algos argument. I'm lazy
    #detailForRun(session,c,runnum)
    beamintensityForRun(session,c,runnum)
if __name__=='__main__':
    main()

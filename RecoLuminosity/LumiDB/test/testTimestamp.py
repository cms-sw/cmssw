import os,sys
from datetime import datetime,timedelta
import coral
connectstring='frontier://cmsfrontier.cern.ch:8000/LumiProd/CMS_LUMI_PROD'
#connectstring='oracle://cms_orcoff_prod/cms_lumi_prod'
os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/cms/DB/lumi'
svc=coral.ConnectionService()
dbsession=svc.connect(connectstring,accessMode=coral.access_Update)
#try:
#    dbsession.transaction().start(True)
#    schema=dbsession.nominalSchema()
#    query=schema.tableHandle('CMSRUNSUMMARY').newQuery()
#    query.addToOutputList("STARTTIME","starttime")
#    query.addToOutputList("STOPTIME","stoptime")
#    queryBind=coral.AttributeList()
#    queryBind.extend("runnum","unsigned int")
#    queryBind["runnum"].setData(int(132440))
#    queryresult=coral.AttributeList()
#    queryresult.extend("starttime","time stamp")
#    queryresult.extend("stoptime","time stamp")
#    query.setCondition("RUNNUM=:runnum",queryBind)
#    query.defineOutput(queryresult)
#    cursor=query.execute()
#    while cursor.next():
#        startT=cursor.currentRow()['starttime']
#        print 'startT ',startT
#        stopT=cursor.currentRow()['stoptime']
#        print 'stopT ',stopT
#    del query
#    dbsession.transaction().commit()
#except Exception,e:
#    print 'caught exception ',str(e)
#    dbsession.transaction().rollback()
#lumisection duration in microseconds
lumisectionDelta=3564*2**18*0.025    
try:
    dbsession.transaction().start(True)
    schema=dbsession.nominalSchema()
    query=schema.tableHandle('CMSRUNSUMMARY').newQuery()
    query.addToOutputList("TO_CHAR(STARTTIME,'MM/DD/YY HH24:MI:SS.FF6')","starttime")
    query.addToOutputList("TO_CHAR(STOPTIME,'MM/DD/YY HH24:MI:SS.FF6')","stoptime")
    queryBind=coral.AttributeList()
    queryBind.extend("runnum","unsigned int")
    queryBind["runnum"].setData(int(132440))
    queryresult=coral.AttributeList()
    queryresult.extend("starttime","string")
    queryresult.extend("stoptime","string")
    query.setCondition("RUNNUM=:runnum",queryBind)
    query.defineOutput(queryresult)
    cursor=query.execute()
    while cursor.next():
        startT=cursor.currentRow()['starttime']
        print 'startT ',startT.data()
        s=datetime.strptime(startT.data(),'%m/%d/%y %H:%M:%S.%f')
        print 'startT ',s
        print 'startT month/day/year hour/minute/second.microsecond',s.month,s.day,s.year,s.hour,s.minute,s.second,s.microsecond
        stopT=cursor.currentRow()['stoptime']
        print 'stopT ',stopT.data()
        a=datetime.strptime(stopT.data(),'%m/%d/%y %H:%M:%S.%f')
        print 'stopT month/day/year hour/minute/second.microsecond',a.month,a.day,a.year,a.hour,a.minute,a.second,a.microsecond
        delta=timedelta(microseconds=lumisectionDelta)
        print '2nd LS would be in ',s+delta
        print '3rd LS would be in ',s+2*delta
    del query
    dbsession.transaction().commit()
except Exception,e:
    print 'caught exception ',str(e)
    dbsession.transaction().rollback()
del dbsession
del svc

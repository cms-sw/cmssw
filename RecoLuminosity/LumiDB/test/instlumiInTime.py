import os,sys
import coral,datetime,time
from RecoLuminosity.LumiDB import lumiQueryAPI,lumiTime,csvReporter

def main(*args):
    runnum=0
    try:
        runnum=args[1]
        report=csvReporter.csvReporter('instlumibytime-'+str(runnum)+'.csv')
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Error)
        os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/cms/DB/lumi'
        svc = coral.ConnectionService()
        connectstr='oracle://cms_orcoff_prod/cms_lumi_prod'
        session=svc.connect(connectstr,accessMode=coral.access_ReadOnly)
        session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
        session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
        session.transaction().start(True)
        schema=session.nominalSchema()
        q=schema.newQuery()
        runsummaryOut=lumiQueryAPI.runsummaryByrun(q,runnum)
        del q
        q=schema.newQuery()
        lumisummaryOut=lumiQueryAPI.lumisummaryByrun(q,runnum,'0001')
        del q
        session.transaction().commit()  
        del session
        del svc
        #print runsummaryOut
        starttimestr=runsummaryOut[3]
        t=lumiTime.lumiTime()
        report.writeRow(['cmslsnum','utctime','unixtimestamp','instlumi'])
        for dataperls in lumisummaryOut:
            cmslsnum=dataperls[0]
            instlumi=dataperls[1]
            startorbit=dataperls[3]
            orbittime=t.OrbitToTime(starttimestr,startorbit)
            orbittimestamp=time.mktime(orbittime.timetuple())+orbittime.microsecond/1e6
            report.writeRow([cmslsnum,orbittime,orbittimestamp,instlumi])

    except IndexError:
        print 'runnumber should be provided'
        return 1
    except Exception, er:
        print str(er)
        return 2
    else:
        return 0

if __name__=='__main__':
    sys.exit(main(*sys.argv))

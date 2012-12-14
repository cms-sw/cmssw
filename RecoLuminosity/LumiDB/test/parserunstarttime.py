import datetime,csv
import coral
from RecoLuminosity.LumiDB import sessionManager,lumiTime,dbUtil,nameDealer
def parsetimefile(filename,datefmt):
    result={}#{run:starttime}
    fileobj=open(filename,'rb')
    freader=csv.reader(fileobj,delimiter=',')
    for row in freader:
        if not row: continue
        runstr=row[0]
        runnum=int("".join(runstr.split()))
        datestr=row[1]
        d=datetime.datetime.strptime(datestr,datefmt)
        result[runnum]=d
    fileobj.close()
    return result
def parserunlength(filename):
    '''
    parse file of format
    select runnum,data_id,count(*) from lumisummaryv2 group by (runnum,data_id);
    '''
    result={}#{(run,lumiid):nls}
    nls=0
    fileobj=open(filename,'rb')
    freader=csv.reader(fileobj,delimiter=',')
    for row in freader:
        if not row: continue
        runstr=row[0]
        runnum=int("".join(runstr.split()))
        lumidataidstr=row[1]
        lumidataid=int("".join(lumidataidstr.split()))
        nlsstr=row[2]
        nls=int("".join(nlsstr.split()))
        result[(runnum,lumidataid)]=nls
    fileobj.close()
    return result

def updatedb(schema,runmap,lumitype):
    '''
    update lumidata set starttime=:rstart,stoptime=:rstop,nls=:nls where runnum=:runnum and data_id=:lumidataid
    '''
    lumidatatableName=''
    if lumitype=='HF':
        lumitableName=nameDealer.lumidataTableName()
    elif lumitype == 'PIXEL':
        lumitableName = nameDealer.pixellumidataTableName()
    else:
        assert False, "ERROR Unknown lumitype '%s'" % lumitype
    t=lumiTime.lumiTime()
    setClause='STARTTIME=:runstarttime,STOPTIME=:runstoptime,NLS=:nls'
    updateCondition='RUNNUM=:runnum AND DATA_ID=:lumidataid'
    inputData=coral.AttributeList()
    inputData.extend('runstarttime','time stamp')
    inputData.extend('runstoptime','time stamp')
    inputData.extend('nls','unsigned int')
    inputData.extend('runnum','unsigned int')
    inputData.extend('lumidataid','unsigned long long')
    db=dbUtil.dbUtil(schema)
    for (run,lumidataid) in runmap.keys():
        [runstarttime,runstoptime,nls]=runmap[(run,lumidataid)]
        runstartT=coral.TimeStamp(runstarttime.year,runstarttime.month,runstarttime.day,runstarttime.hour,runstarttime.minute,runstarttime.second,runstarttime.microsecond*1000)
        runstopT=coral.TimeStamp(runstoptime.year,runstoptime.month,runstoptime.day,runstoptime.hour,runstoptime.minute,runstoptime.second,runstoptime.microsecond*1000)
        inputData['runstarttime'].setData(runstartT)
        inputData['runstoptime'].setData(runstopT)
        inputData['nls'].setData(nls)
        inputData['runnum'].setData(int(run))
        inputData['lumidataid'].setData(int(lumidataid))
        db.singleUpdate(lumitableName,setClause,updateCondition,inputData)
        
if __name__ == "__main__" :
    dbstr='oracle://cms_orcon_prod/cms_lumi_prod'
    authpath='/nfshome0/xiezhen/authwriter'
    f='runstarttime.dat'
    runlengthfile='nlsperrun-pixel.dat'
    dfmt='%d-%m-%Y %H:%M:%S'
    t=parsetimefile(f,dfmt)
    timedrunlist=t.keys()
    r=parserunlength(runlengthfile)
    output={}#{(run,lumidataid):[[date,nls]]}
    for (run,lumidataid) in r.keys():
        nls=r[(run,lumidataid)]
        if run in timedrunlist:
            runstarttime=t[run]
            runstoptime=runstarttime+nls*datetime.timedelta(seconds=23.31)
            output[(run,lumidataid)]=[runstarttime,runstoptime,nls]
    try:
        lumisvc=sessionManager.sessionManager(dbstr,authpath=authpath,debugON=False)
        lumisession=lumisvc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        lumisession.transaction().start(False)
        updatedb(lumisession.nominalSchema(),output,'PIXEL')
        lumisession.transaction().commit()
        del lumisession
        del lumisvc
    except:
        raise
    


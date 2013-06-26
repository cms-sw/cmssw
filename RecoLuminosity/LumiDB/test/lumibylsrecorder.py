import sys,os,os.path,csv,coral,commands,glob
from RecoLuminosity.LumiDB import dbUtil,lumiTime,sessionManager,nameDealer,argparse
def getrunsInResult(schema,minrun=132440,maxrun=500000):
    '''
    get runs in result tables in specified range
    output:
         [runnum]
         select distinct runnum from hflumiresult where runnum>=:minrun and runnum<=:maxrun;
    '''
    result=[]
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( 'HFLUMIRESULT' )
        qHandle.addToOutputList('distinct RUNNUM')
        qCondition=coral.AttributeList()
        qCondition.extend('minrun','unsigned int')
        qCondition.extend('maxrun','unsigned int')
        qCondition['minrun'].setData(minrun)
        qCondition['maxrun'].setData(maxrun)
        qResult=coral.AttributeList()
        qResult.extend('RUNNUM','unsigned int')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM>=:minrun AND RUNNUM<=:maxrun',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['RUNNUM'].data()
            result.append(runnum)
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    return result
    
def getrunsInCurrentData(schema,minrun=132440,maxrun=500000):
    '''
    get runs in data tables in specified range
    output:
         [runnum]
         select runnum,tagid from tagruns where runnum>=:minrun and runnum<=:maxrun;
    '''
    tmpresult={}
    qHandle=schema.newQuery()
    try:
        qHandle.addToTableList( nameDealer.tagRunsTableName() )
        qHandle.addToOutputList('RUNNUM')
        qHandle.addToOutputList('TAGID')
        qCondition=coral.AttributeList()
        qCondition.extend('minrun','unsigned int')
        qCondition.extend('maxrun','unsigned int')
        qCondition['minrun'].setData(minrun)
        qCondition['maxrun'].setData(maxrun)
        qResult=coral.AttributeList()
        qResult.extend('RUNNUM','unsigned int')
        qResult.extend('TAGID','unsigned long long')
        qHandle.defineOutput(qResult)
        qHandle.setCondition('RUNNUM>=:minrun AND RUNNUM<=:maxrun',qCondition)
        cursor=qHandle.execute()
        while cursor.next():
            runnum=cursor.currentRow()['RUNNUM'].data()
            tagid=cursor.currentRow()['TAGID'].data()
            if not tmpresult.has_key(runnum):
                tmpresult[runnum]=tagid
            else:
                if tagid>tmpresult[runnum]:
                    tmpresult[runnum]=tagid
        del qHandle
    except:
        if qHandle:del qHandle
        raise
    if tmpresult:return tmpresult.keys()    
    return []

def execCalc(connectStr,authpath,outdir,runnum,minbias):
    '''
    run lumiCalc2.py lumibyls for the run
    '''
    outdatafile=os.path.join(outdir,str(runnum)+'.csv')
    outheaderfile=os.path.join(outdir,str(runnum)+'.txt')
    command = 'lumiCalc2.py lumibyls --without-checkforupdate -c ' +connectStr+' -P '+authpath+' -r '+str(runnum)+' -o '+outdatafile+' --headerfile '+outheaderfile+' --minBiasXsec '+str(minbias)
    (status, output) = commands.getstatusoutput(command)
    if status != 0:
        print 'empty result ',command
        return 0
    print command
    return runnum

class lslumiParser(object):
    def __init__(self,lslumifilename,headerfilename):
        '''
        '''
        self.__filename=lslumifilename
        self.__headername=headerfilename
        self.lumidata=[]#[fill,run,lumils,cmsls,beamstatus,beamenergy,delivered,recorded,avgpu]
        self.datatag=''
        self.normtag=''
    def parse(self):
        '''
        parse ls lumi file
        '''
        hf=None
        try:
            hf=open(self.__headername,'rb')
        except IOError:
            print 'failed to open file ',self.__headername
            return
        for line in hf:
            if "lumitype" in line:
                fields=line.strip().split(',')
                for field in fields:
                    a=field.strip().split(':')
                    if a[0]=='datatag':
                        self.datatag=a[1].strip()
                    if a[0]=='normtag':
                        self.normtag=a[1].strip()
                break
        hf.close()
        f=None
        try:
            f=open(self.__filename,'rb')
        except IOError:
            print 'failed to open file ',self.__filename
            return
        freader=csv.reader(f,delimiter=',')
        idx=0
        for row in freader:
           if idx==0:
               idx=1 # skip header
               continue
           [run,fill]=map(lambda i:int(i),row[0].split(':'))
           [lumils,cmsls]=map(lambda i:int(i),row[1].split(':'))
           chartime=row[2]
           beamstatus=row[3]
           beamenergy=float(row[4])
           beamenergy=int(round(beamenergy))
           delivered=float(row[5])
           recorded=float(row[6])
           avgpu=float(row[7])
           self.lumidata.append([fill,run,lumils,cmsls,chartime,beamstatus,beamenergy,delivered,recorded,avgpu])
        f.close()
    
def updateindb(session,datatag,normtag,lumidata,bulksize):
    '''
    '''
def findmaxrun(dirname):
    '''
    find max runnum from file xxxxxx.csv in specified dir
    output: 0 or runnum
    '''
    maxrunnum=0
    result=maxrunnum
    allcsvfiles=glob.glob(os.path.join(dirname,'??????.csv'))
    if not allcsvfiles:
        return result
    for csvfile in allcsvfiles:
        runnum=int(os.path.basename(csvfile).split('.')[0])
        if runnum>maxrunnum:
            result=runnum
            maxrunnum=runnum
    return result
def addindb(session,datatag,normtag,lumidata,bulksize):
    '''
    input : [fill,run,lumils,cmsls,lstime,beamstatus,beamenergy,delivered,recorded,avgpu]
    '''
    hfresultDefDict=[('RUNNUM','unsigned int'),('LS','unsigned int'),('CMSLS','unsigned int'),('FILLNUM','unsigned int'),('TIME','time stamp'),('BEAM_STATUS','string'),('ENERGY','unsigned int'),('DELIVERED','float'),('RECORDED','float'),('AVG_PU','float'),('DATA_VERSION','string'),('NORM_VERSION','string'),('INSERT_TIME','time stamp')]
    
    committedrows=0
    nrows=0
    bulkvalues=[]
    lute=lumiTime.lumiTime()
    try:
        for datum in lumidata:
            [fillnum,runnum,lumils,cmsls,lstime_char,beamstatus,beamenergy,delivered,recorded,avgpu]=datum
            inserttime=coral.TimeStamp()
            lstime=lute.StrToDatetime(lstime_char,customfm='%m/%d/%y %H:%M:%S')
            corallstime=coral.TimeStamp(lstime.year,lstime.month,lstime.day,lstime.hour,lstime.minute,lstime.second,0)
            bulkvalues.append([('RUNNUM',runnum),('LS',lumils),('CMSLS',cmsls),('FILLNUM',fillnum),('TIME',corallstime),('BEAM_STATUS',beamstatus),('ENERGY',beamenergy),('DELIVERED',delivered),('RECORDED',recorded),('AVG_PU',avgpu),('DATA_VERSION',datatag),('NORM_VERSION',normtag),('INSERT_TIME',inserttime)])
            nrows+=1
            committedrows+=1
            if nrows==bulksize:
                print 'committing trg in LS chunck ',nrows
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert('HFLUMIRESULT',hfresultDefDict,bulkvalues)
                session.transaction().commit()
                nrows=0
                bulkvalues=[]
            elif committedrows==len(lumidata):
                print 'committing at the end '
                db=dbUtil.dbUtil(session.nominalSchema())
                session.transaction().start(False)
                db.bulkInsert('HFLUMIRESULT',hfresultDefDict,bulkvalues)
                session.transaction().commit()
                
    except :
        print 'error in addindb'
        raise 
if __name__ == "__main__" :
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "check/compute lumi for new runs in specified range",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s',dest='sourcestr',action='store',
                        required=False,
                        help='source DB connect string',
                        default='oracle://cms_orcon_prod/cms_lumi_prod')
    parser.add_argument('-d',dest='deststr',action='store',
                        required=False,
                        help='dest DB connect string',
                        default='oracle://cms_orcon_prod/cms_lumi_prod')
    parser.add_argument('-P',dest='pth',action='store',
                        required=False,
                        help='path to authentication file')
    parser.add_argument('-b',dest='begrun',action='store',
                        type=int,
                        required=False,
                        help='begin run number')
    parser.add_argument('-e',dest='endrun',action='store',
                        type=int,
                        required=False,
                        default=500000,
                        help='end run number')
    parser.add_argument('--minBiasXsec',dest='minbiasxsec',action='store',
                        default=69300.0,
                        type=float,
                        required=False,
                        help='minbias cross-section in ub'
                        )
    parser.add_argument('-i',dest='indir',action='store',
                        required=False,
                        default=None,
                        help='input directory'
                        )
    parser.add_argument('-o',dest='outdir',action='store',
                        required=False,
                        default='.',
                        help='output directory'
                        )
    options=parser.parse_args()
    if not os.path.isdir(options.outdir) or not os.path.exists(options.outdir):
        print '[ERROR] non-existing output dir ',options.outdir
        sys.exit(12)
    if not options.begrun and not options.indir:
        print '[ERROR] must specify at least -b or -i'
        sys.exit(13)
    begrun=0
    if options.begrun:
        begrun=options.begrun
    if options.indir and os.path.exists(options.indir) and os.path.isdir(options.indir):
        maxrunindir=findmaxrun(options.indir)
        if maxrunindir>begrun:
            begrun=maxrunindir
    if not begrun:
        print '[ERROR] cannot find the begin run'
        sys.exit(14)
    sourcesvc=sessionManager.sessionManager(options.sourcestr,authpath=options.pth,debugON=False)
    sourcesession=sourcesvc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    sourcesession.transaction().start(True)
    sourcerunlist=getrunsInCurrentData(sourcesession.nominalSchema(),minrun=begrun,maxrun=options.endrun)
    sourcesession.transaction().commit()
    sourcerunlist.sort()
    #print 'source ',len(sourcerunlist),sourcerunlist
    destsvc=sessionManager.sessionManager(options.deststr,authpath=options.pth,debugON=False)
    destsession=destsvc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    destsession.transaction().start(True)
    destrunlist=getrunsInResult(destsession.nominalSchema(),minrun=begrun,maxrun=options.endrun)
    destsession.transaction().commit()
    destrunlist.sort()
    #print 'dest ',len(destrunlist),destrunlist
    processedruns=[]
    for r in sourcerunlist:
        if r not in destrunlist:
            result=execCalc(options.sourcestr,options.pth,options.outdir,r,options.minbiasxsec)
            if result:
                processedruns.append(result)
    if len(processedruns)==0:
        print '[INFO] no new runs found in range:',begrun,options.endrun
    else:
        for pr in processedruns:
            lslumifilename=os.path.join(options.outdir,str(pr)+'.csv')
            lumiheaderfilename=os.path.join(options.outdir,str(pr)+'.txt')
            p=lslumiParser(lslumifilename,lumiheaderfilename)
            p.parse()
            addindb(destsession,p.datatag,p.normtag,p.lumidata,bulksize=500)
    del sourcesession
    del sourcesvc
    del destsession
    del destsvc

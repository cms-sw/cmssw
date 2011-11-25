import sys,os,csv,re,coral,array
from RecoLuminosity.LumiDB import argparse,sessionManager,CommonUtil,dataDML,revisionDML
def patchV2(dbsession,runnum,inputdata):
    '''
    input: {pathname:[(cmslsnum,presc)]}
    unpack hltdata.pathnameclob
    for each inputpath, find data_id from runnum,cmslsnum
    for each inputpath, find its position in pathnameclob
    inputdata:  {pathname:[(cmslsnum,presc)]}
    '''
    #query pathnameclob
    schema=dbsession.nominalSchema()
    oldhltdataid=dataDML.guessHltDataIdByRun(schema,runnum)
    existingrundata=dataDML.hltRunById(schema,oldhltdataid)
    hltnamedict=existingrundata[3]
    existinglsdata=dataDML.hltLSById(schema,oldhltdataid)
    oldlsdata=existinglsdata[1]
    print oldlsdata.keys()
    #print existinglsdata
def parsepresc(inputlistoflist,minlsnum,maxlsnum,lsboundaries):
    '''
    input:
        inputlistoflist : [[path,presc1,presc2],[]...]
        minlsnum: minimum ls
        maxlsnum: max ls
        prescboundaries=[(1,2),(3,67),(68,188),(189,636),(637,1004))]
    output:
        result : {pathname:[(cmslsnum,presc)]}
    '''
    if not maxlsnum:
        maxlsnum=lsboundaries[-1][1]
    else:
        maxlsnum=int(maxlsnum)
    if not minlsnum:
        minlsnum=lsboundaries[0][0]
    else:
        minlsnum=int(minlsnum)
    result={}#{pathname:[(cmslsnum,presc),()...]}
    alllsnum=range(minlsnum,maxlsnum+1)
    prescidxdict={}#{cmsls:prescidx}
    for cmsls in alllsnum:
        for prescidx,(blow,bhigh) in enumerate(lsboundaries):
            if cmsls>=blow and cmsls<=bhigh:
                prescidxdict[cmsls]=prescidx+1
    for pathinfo in inputlistoflist:#loop over path
        pathnamefield=pathinfo[0]
        pathname=pathnamefield.split(' ')[0]
        if not result.has_key(pathname): result[pathname]=[]
        for idx,p in enumerate(pathinfo):#loop over presc possibilities
            if idx==0: continue #this is a pathname field
            presc=int(p)
            for cmsls in alllsnum:
                if idx==prescidxdict[cmsls]:
                    result[pathname].append(presc)
    return result
def parseInfile(filename):
    '''
    input filename
    output:(runnum,[(lsboundaryLow,lsboundaryHigh),...],[[pathname,presc1,presc2,presc3...],[]])
             every path with its possible presc
    '''
    result=[]
    f=open(filename,'rb')
    reader=csv.reader(f,delimiter=',')
    i=0
    runnum=0
    lsboundaries=[]
    pathinfo=[]
    p=re.compile('\d+')
    for row in reader:
        if not row or len(row)==0: continue
        fields=[r.strip() for r in row if r and len(r)!=0]
        if len(fields)==0: continue
        if i==0:
            runnum=row[0].split(' ')[1]
        elif i==1:
            for field in fields:
                [low,high]=p.findall(field)
                lsboundaries.append([int(low),int(high)])
        else:
            pathinfo.append(fields)
        i+=1
    return (int(runnum),lsboundaries,pathinfo)

def  main(*args):
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Patch HLT prescale from text file",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['v2']
    parser.add_argument('action',choices=allowedActions,
                        help='command actions')
    parser.add_argument('-c',dest='connect',action='store',
                        required=True,
                        help='connect string to lumiDB,optional',
                        default=None)
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file')
    parser.add_argument('-i',dest='ifile',action='store',
                        required=True,
                        help='patch data file ')
    parser.add_argument('-lsmin',dest='lsmin',action='store',
                        default=1,
                        required=False,
                        help='minimum ls to patch')
    parser.add_argument('-lsmax',dest='lsmax',action='store',
                        default=None,
                        required=False,
                        help='max ls to patch')
    parser.add_argument('--debug',dest='debug',action='store_true',
                        required=False,
                        help='debug ')
    options=parser.parse_args()
    (runnum,lsboundaries,pathinfo)=parseInfile(options.ifile)
    result=parsepresc(pathinfo,options.lsmin,options.lsmax,lsboundaries)
    #print result
    
    os.environ['CORAL_AUTH_PATH'] = options.authpath      
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    dbsession.transaction().start(True)
    patchV2(dbsession,runnum,result)
    dbsession.transaction().commit()
    del dbsession
    del svc
    
if __name__=='__main__':
    sys.exit(main(*sys.argv))

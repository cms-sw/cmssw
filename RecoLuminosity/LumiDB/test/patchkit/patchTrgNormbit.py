import csv,os,sys,coral,array
from RecoLuminosity.LumiDB import argparse,sessionManager,CommonUtil,idDealer,dbUtil,dataDML,revisionDML
NCOLS=4
def updateLSTrg(dbsession,runnum,perlsrawdata):
    '''
    input: perlsrawdata [(cmslsnum,normbitcount,normbitprescale),(cmslsnum,normbitcount,normbitprescale)...]
    update lstrg set bitzerocount=:normbitcount,bitzeroprescale=:normbitprescale where runnum=:runnum and cmslsnum=:cmslsnum
    '''
    dataDef=[]
    dataDef.append(('CMSLSNUM','unsigned int'))
    dataDef.append(('BITZEROCOUNT','unsigned int'))
    dataDef.append(('BITZEROPRESCALE','unsigned int'))
    dataDef.append(('RUNNUM','unsigned int'))
    dataDef.append(('CMSLSNUM','unsigned int'))
    bulkinput=[]
    dbsession.transaction().start(False)
    db=dbUtil.dbUtil(dbsession.nominalSchema())
    updateAction='BITZEROCOUNT=:bitzerocount,BITZEROPRESCALE=:bitzeroprescale'
    updateCondition='RUNNUM=:runnum AND CMSLSNUM=:cmslsnum'
    bindvarDef=[('bitzerocount','unsigned int'),('bitzeroprescale','unsigned int'),('runnum','unsigned int'),('cmslsnum','unsigned int')]
    for (cmslsnum,normbitcount,normbitprescale) in perlsrawdata:
        bulkinput.append([('bitzerocount',normbitcount),('bitzeroprescale',normbitprescale),('runnum',runnum),('cmslsnum',cmslsnum)])
    db.updateRows('LSTRG',updateAction,updateCondition,bindvarDef,bulkinput)
    #dbsession.transaction().rollback()
    dbsession.transaction().commit()

def parseInputFile(ifilename):
    perlsdata=[]
    try:
        csvfile=open(ifilename,'rb')
        reader=csv.reader(csvfile,delimiter=',',skipinitialspace=True)
        for row in reader:
            if not row: continue
            if len(row)!=NCOLS: continue
            runnumStr=row[0].strip()
            cmslsnumStr=row[1].strip()
            normbitCountStr=row[2].strip()
            normbitPrescStr=row[3].strip()
            perlsdata.append( (int(cmslsnumStr),int(normbitCountStr),int(normbitPrescStr)) )
    except Exception,e:
        raise RuntimeError(str(e))
    return perlsdata
def main(*args):
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi fake",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    parser.add_argument('-r',dest='runnumber',action='store',
                        type=int,
                        required=True,
                        help='run number')
    parser.add_argument('-i',dest='ifile',action='store',
                        required=True,
                        help='patch data file ')
    parser.add_argument('--debug',dest='debug',action='store_true',
                        required=False,
                        help='debug ')
    options=parser.parse_args()
    os.environ['CORAL_AUTH_PATH'] = options.authpath      
    perlsrawdata=parseInputFile(options.ifile)
    print perlsrawdata
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    dbsession=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    if options.action=='v2':
        updateLSTrg(dbsession,options.runnumber,perlsrawdata)
    #elif options.action=='v1' :
    #    summaryidlsmap=insertLumiSummarydata(dbsession,options.runnumber,perlsrawdata,deliveredonly=options.deliveredonly)
    #    if perbunchrawdata:
    #        insertLumiDetaildata(dbsession,perlsrawdata,perbunchrawdata,summaryidlsmap)
    del dbsession
    del svc

if __name__=='__main__':
    sys.exit(main(*sys.argv))

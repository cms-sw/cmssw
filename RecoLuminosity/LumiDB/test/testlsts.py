import os,sys,datetime,time,calendar
import coral
from RecoLuminosity.LumiDB import lumiCalcAPI,argparse,sessionManager

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "get lumi lstimestamp",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',
                        action='store',
                        required=False,
                        help='connect string to lumiDB,optional',
                        default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',
                        action='store',
                        help='path to authentication file,optional')
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug')
    options=parser.parse_args()
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    session.transaction().start(True)
    schema=session.nominalSchema()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    irunlsdict={181530:[1,2,4],182133:None}#to decide by yourself!
    lumidata=lumiCalcAPI.instLumiForRange(schema,irunlsdict)
    orderedrunlist=sorted(lumidata)
    print 'run,lumilsnum,unixtimestamp'
    for run in orderedrunlist:
        perrundata=lumidata[run]
        for perlsdata in perrundata:
            lumilsnum=perlsdata[0]
            tsdatetime=perlsdata[2]
            ts=calendar.timegm(tsdatetime.utctimetuple())
            print run,lumilsnum,ts
    session.transaction().commit()
    #above method is limited to data in the lumiDB, not all runs/ls are available in lumiDB
    #in case of reading a datetime string from somewhere other than lumiDB:
    #1. convert date timestring to python datetime object,utility lumiTime.StrToDatetime can be used for this
    #2. calendar.timegm(tsdatetime.utctimetuple())

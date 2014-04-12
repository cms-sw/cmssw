#!/usr/bin/env python
import sys,os
from RecoLuminosity.LumiDB import sessionManager,argparse,nameDealer,revisionDML,dataDML

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "pixel lumi reader",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',
                        required=True,
                        help='connect string to lumiDB (required)',
                        )
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file (required)'
                        )
    parser.add_argument('-r',dest='runnum',action='store',
                        type=int,
                        required=True,
                        help='run number'
                        )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug'
                        )
    
    options=parser.parse_args()
    svc=sessionManager.sessionManager(options.connect,
                                      authpath=options.authpath,
                                      debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    session.transaction().start(True)
    lumiIdInDATA=dataDML.guessLumiDataIdByRunInBranch(session.nominalSchema(),options.runnum,nameDealer.lumidataTableName(),branchName='DATA')
    print lumiIdInDATA
    lumiIdInPIXELLUMI=dataDML.guessLumiDataIdByRunInBranch(session.nominalSchema(),options.runnum,nameDealer.lumidataTableName(),branchName='PIXELLUMI')
    print lumiIdInPIXELLUMI
    session.transaction().commit()
    del session
    del svc

#!/usr/bin/env python
###################################################################
# Command to insert pixel lumi data in lumiDB                     #
#                                                                 #
# Author:      Zhen Xie                                           #
###################################################################

import os,sys,time,json
from RecoLuminosity.LumiDB import sessionManager,argparse,nameDealer,revisionDML,CommonUtil,dataDML,lumiCalcAPI,lumiParameters

def getdistribution(lsnum,bxindices,bxvalue):
    '''
    input: lsnum,bxindices[],bxvalue[]
    output:[(bxindex,bxratio)]
    '''
    result=[]
    totlslumi=sum(bxvalue)
    for arrayidx,bxindex in enumerate(bxindices):
        bxratio=0.
        if totlslumi!=0:
            bxval=bxvalue[arrayidx]
            bxratio=bxval/totlslumi
        result.append((bxindex,bxratio))
    return result

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "pixel lumi loader",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',
                        required=True,
                        help='connect string to lumiDB (required)',
                        )
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file (required)'
                        )
    parser.add_argument('-r',dest='runnumber',action='store',
                        type=int,
                        required=False,
                        help='run number')
    parser.add_argument('-i',dest='inputfile',action='store',
                        required=False,
                        help='lumi range selection file')
    parser.add_argument('--comment',action='store',
                        required=False,
                        help='patch comment'
                       )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug'
                        )
    options=parser.parse_args()
    svc=sessionManager.sessionManager(options.connect,
                                      authpath=options.authpath,
                                      debugON=options.debug)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    runlist=[]
    irunlsdict={}
    if options.runnumber: # if runnumber specified, do not go through other run selection criteria
        irunlsdict[options.runnumber]=None
        runlist=irunlsdict.keys()
    else:
        if options.inputfile:
            (irunlsdict,iresults)=parseInputFiles(options.inputfile)
            runlist=irunlsdict.keys()
    session.transaction().start(True)
    (hf_tagid,hf_tagname)=revisionDML.currentDataTag(session.nominalSchema(),lumitype='HF')
    hfdataidmap=revisionDML.dataIdsByTagId(session.nominalSchema(),hf_tagid,runlist,withcomment=False,lumitype='HF')
    lumirundata=dataDML.lumiRunByIds(session.nominalSchema(),hfdataidmap,lumitype='HF')
    #{runnum: (datasource(1),nominalegev(2),ncollidingbunches(3)}
    GrunsummaryData=lumiCalcAPI.runsummaryMap(session.nominalSchema(),irunlsdict)
    session.transaction().commit()
    for runnum in lumirundata.keys():
        session.transaction().start(True)
        hfdataidinfo=hfdataidmap[runnum]
        hflumidata=lumiCalcAPI.instLumiForIds(session.nominalSchema(),irunlsdict,hfdataidmap,GrunsummaryData,beamstatusfilter=None,withBXInfo=True,bxAlgo='OCC1')
        for perlsdata in hflumidata[runnum]:
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            bxinfo=perlsdata[9]
            bxdistro=getdistribution(cmslsnum,bxinfo[0],bxinfo[1])
            print lumilsnum,cmslsnum,CommonUtil.flatten(bxdistro)
        session.transaction().commit()

    del session
    del svc

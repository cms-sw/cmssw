#!/usr/bin/env python
import os,sys,time,json
from RecoLuminosity.LumiDB import sessionManager,argparse,nameDealer,revisionDML,dataDML,generateDummyData

def generateLumiRundata(filename,nominalegev,runlist):
    '''
    output: {runnum:[datasource,nominalegev]}
    '''
    result={}
    for run in runlist:
        result[run]=[filename,nominalegev]
    return result

def generateLumiLSdataForRun(lsdata):
    '''
    input [(cmslsnum,instlumi),...]
    output:
    i.e. bulkInsertLumiLSSummary expected input: {lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit]}
    '''
    result={}
    beamstatus='STABLE BEAMS'
    beamenergy=3.5e03
    numorbit=262144
    startorbit=0
    for (cmslsnum,instlumi) in lsdata:
        lumilsnum=cmslsnum
        instlumierror=0.0
        instlumiquality=0
        startorbit=(cmslsnum-1)*numorbit
        result[lumilsnum]=[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit]
    return result

def inversem2toinverseub(i):
    '''
    input: number in m-2
    output: number in /ub
    '''
    return float(i)*(10e-34)
def toinstlumi(i):
    '''
    input: luminosity integrated in ls
    output: avg instlumi in ls Hz/ub
    '''
    lslength=23.357
    return float(i)/lslength
def parseInputFile(filename):
    '''
    input:pixel lumi json file
    output:{runnumber,[(cmslsnum,instlumi)]}
    '''
    result={}
    json_data=open(filename)
    strresult=json.load(json_data)
    json_data.close()
    for runnum,perrundata in strresult.items():
        allls=map(int,perrundata.keys())        
        for cmsls in range(1,max(allls)+1):
            instlumi=0.0
            if cmsls in allls:
               intglumiinub=inversem2toinverseub(perrundata[str(cmsls)][0])
               instlumi=toinstlumi(intglumiinub)#unit Hz/ub
            result.setdefault(int(runnum),[]).append((cmsls,instlumi))
    return result
def createBranch(schema):
    try:
        pixellumiinfo=revisionDML.createBranch(schema,'PIXELLUMI','TRUNK',comment='pixel lumi data')
        print 'branch PIXELLUMI created: ',pixellumiinfo
    except:
        print 'branch already exists, do nothing'
##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "pixel lumi loader",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['createbranch','load']
    parser.add_argument('action',choices=allowedActions,
                        help='command actions')
    parser.add_argument('-c',dest='connect',action='store',
                        required=True,
                        help='connect string to lumiDB (required)',
                        )
    parser.add_argument('-P',dest='authpath',action='store',
                        required=True,
                        help='path to authentication file (required)'
                        )
    parser.add_argument('-i',dest='inputfile',action='store',
                        required=True,
                        help='pixel lumi file'
                        )
    parser.add_argument('--debug',dest='debug',action='store_true',
                        help='debug'
                        )
    options=parser.parse_args()
    svc=sessionManager.sessionManager(options.connect,
                                      authpath=options.authpath,
                                      debugON=options.debug)
    session=svc.openSession(isReadOnly=False,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    inputfilename=os.path.abspath(options.inputfile)
    if options.action=='createbranch':
        session.transaction().start(False)
        createBranch()
        session.transaction().commit()
    
    if options.action=='load':
        session.transaction().start(True)
        (pixellumibranchid,pixellumibranchparent)=revisionDML.branchInfoByName(session.nominalSchema(),'PIXELLUMI')
        pixellumibranchinfo=(pixellumibranchid,'PIXELLUMI')
        session.transaction().commit()
        print 'PIXELLUMI branch info ',pixellumibranchinfo
        print 'data source ',inputfilename
        beamenergy=3500.0
        parseresult=parseInputFile(inputfilename)
        alllumirundata=generateLumiRundata(inputfilename,beamenergy,parseresult.keys())
        alllumilsdata={}
        for runnum,perrundata in parseresult.items():
            alllumilsdata[runnum]=generateLumiLSdataForRun(perrundata)
        #print 'runlevel data ',alllumirundata
        #print 'lslevel data ',alllumilsdata        
        for runnum,pixellumirundata in alllumirundata.items():
            print runnum
            session.transaction().start(False)
            (pixellumirevid,pixellumientryid,pixellumidataid)=dataDML.addLumiRunDataToBranch(session.nominalSchema(),runnum,pixellumirundata,pixellumibranchinfo)
            pixellumilsdata=alllumilsdata[runnum]
            dataDML.bulkInsertLumiLSSummary(session,runnum,pixellumidataid,pixellumilsdata,withDetails=False)
            session.transaction().commit()
    del session
    del svc

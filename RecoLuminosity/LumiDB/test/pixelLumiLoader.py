#!/usr/bin/env python
###################################################################
# Command to insert pixel lumi data in lumiDB                     #
#                                                                 #
# Author:      Zhen Xie                                           #
###################################################################

import os,sys,time,json,coral
from datetime import datetime
from RecoLuminosity.LumiDB import sessionManager,argparse,nameDealer,revisionDML,dataDML,lumiParameters,lumiTime

def generateLumiRundata(filename,runsummaryData,runlist):
    '''
    input: runsummaryData {runnum: (datasource(0),nominalegev(1),ncollidingbunches(2),starttime(3),stoptime(4)}
    output: {runnum: [nominalenergy,ncollidingbunches,starttime,stoptime,nls] }
    '''
    result={}
    t=lumiTime.lumiTime()
    for run in runlist:
        summary=runsummaryData[run]
        start=datetime.strptime(summary[3],'%m/%d/%y %H:%M:%S')
        stop=datetime.strptime(summary[4],'%m/%d/%y %H:%M:%S')
        starttime=coral.TimeStamp(start.year,start.month,start.day,start.hour,start.minute,start.second,0)
        stoptime=coral.TimeStamp(stop.year,stop.month,stop.day,stop.hour,stop.minute,stop.second,0)
        result[run]=[filename,summary[1],summary[2],starttime,stoptime,0]
    return result

def generateLumiLSdataForRun(lsdata,lumirundata,beamsta):
    '''
    input:
      lsdata: [(cmslsnum,instlumi),...]
      lumirundata: [datasource,nominalegev,ncollidingbunches,ncollidingbunches,starttime,stoptime,o]
      beamstatus {cmslsnum:beamstatus}
    output:
    i.e. bulkInsertLumiLSSummary expected input: {lumilsnum:[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit]}
    '''
    lumip=lumiParameters.ParametersObject()
    result={}
    beamstatus='STABLE BEAMS'
    beamenergy=lumirundata[1]
    numorbit=lumip.numorbit
    startorbit=0
    for (cmslsnum,instlumi) in lsdata:
        lumilsnum=cmslsnum
        instlumierror=0.0
        instlumiquality=0
        startorbit=(cmslsnum-1)*numorbit
        if beamsta and cmslsnum in beamsta:
            beamstatus=beamsta[cmslsnum]
        result[lumilsnum]=[cmslsnum,instlumi,instlumierror,instlumiquality,beamstatus,beamenergy,numorbit,startorbit]
    return result

def inversem2toinverseub(i):
    '''
    input: number in m-2
    output: number in /ub
    '''
    return float(i)*(1.0e-34)
def toinstlumi(i):
    '''
    input: luminosity integrated in ls
    output: avg instlumi in ls Hz/ub
    '''
    lumip=lumiParameters.ParametersObject()
    lslength=lumip.lslengthsec()
    return float(i)/lslength
def parseInputFile(filename,singlerun=None):
    '''
    input:pixel lumi json file
    output:{runnumber,[(cmslsnum,instlumi)]}
    '''
    result={}
    json_data=open(filename)
    strresult=json.load(json_data)
    json_data.close()
    strruns=strresult.keys()
    rs=[int(x) for x in strruns]
    rs.sort()
    print rs
    for runnum,perrundata in strresult.items():
        if singlerun:
            if int(runnum)!=int(singlerun):
                print 'skip '+str(runnum)+' , is not single run ',singlerun
                continue
        allls=map(int,perrundata.keys())        
        for cmsls in range(1,max(allls)+1):
            instlumi=0.0
            if cmsls in allls:
               intglumiinub=inversem2toinverseub(perrundata[str(cmsls)])
               instlumi=toinstlumi(intglumiinub)#unit Hz/ub
            result.setdefault(int(runnum),[]).append((cmsls,instlumi))
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
    parser.add_argument('-i',dest='inputfile',action='store',
                        required=True,
                        help='pixel lumi file'
                        )
    parser.add_argument('--comment',action='store',
                        required=False,
                        help='patch comment'
                       )
    parser.add_argument('--singlerun',action='store',
                        required=False,
                        default=None,
                        help='pick single run from input file'
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
    parseresult=parseInputFile(inputfilename,options.singlerun)
    runlist=parseresult.keys()
    irunlsdict={}
    for run in runlist:
        irunlsdict[run]=None
    session.transaction().start(True)
    (pixellumibranchid,pixellumibranchparent)=revisionDML.branchInfoByName(session.nominalSchema(),'DATA')
    print 'pixellumibranchid ',pixellumibranchid,' pixellumibranchparent ',pixellumibranchparent
    pixellumibranchinfo=(pixellumibranchid,'DATA')
    (pixel_tagid,pixel_tagname)=revisionDML.currentDataTag(session.nominalSchema(),lumitype='PIXEL')
    (hf_tagid,hf_tagname)=revisionDML.currentDataTag(session.nominalSchema(),lumitype='HF')    
    hfdataidmap=revisionDML.dataIdsByTagId(session.nominalSchema(),hf_tagid,runlist,withcomment=False,lumitype='HF')
    beamstatusdata=dataDML.beamstatusByIds(session.nominalSchema(),hfdataidmap)
    #print 'beamstatusdata ',beamstatusdata
    lumirundata=dataDML.lumiRunByIds(session.nominalSchema(),hfdataidmap,lumitype='HF')#{runnum: (nominalegev(0),ncollidingbunches(1),starttime(2),stoptime(3),nls(4)}
    session.transaction().commit()
    #print 'lumirundata ', lumirundata
    alllumirundata=generateLumiRundata(inputfilename,lumirundata,runlist)
    alllumilsdata={}
    for runnum,perrundata in parseresult.items():
        pixellumidataid=0
        session.transaction().start(False)
        #session.transaction().start(True)
        hfdataidinfo=hfdataidmap[runnum]
        hflumidataid=hfdataidinfo[0]
        trgdataid=hfdataidinfo[1]
        hltdataid=hfdataidinfo[2]
        beamsta={}
        if runnum in beamstatusdata:
            beamsta=beamstatusdata[runnum]
        alllumilsdata[runnum]=generateLumiLSdataForRun(perrundata,alllumirundata[runnum],beamsta)#lumirundata [datasource,nominalenergy,ncollidingbunches,starttime,stoptime,nls]
        pixellumirundata=alllumirundata[runnum]
        (pixellumirevid,pixellumientryid,pixellumidataid)=dataDML.addLumiRunDataToBranch(session.nominalSchema(),runnum,pixellumirundata,pixellumibranchinfo,nameDealer.pixellumidataTableName())
        pixellumilsdata=alllumilsdata[runnum]
        revisionDML.addRunToCurrentDataTag(session.nominalSchema(),runnum,pixellumidataid,trgdataid,hltdataid,lumitype='PIXEL')
        session.transaction().commit()
        dataDML.bulkInsertLumiLSSummary(session,runnum,pixellumidataid,pixellumilsdata,nameDealer.pixellumisummaryv2TableName(),withDetails=False)

       
    del session
    del svc

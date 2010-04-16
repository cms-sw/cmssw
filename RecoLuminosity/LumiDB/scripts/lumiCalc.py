#!/usr/bin/env python
VERSION='2.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,selectionParser,hltTrgSeedMapper,connectstrParser,cacheconfigParser,tablePrinter,csvReporter
from RecoLuminosity.LumiDB.wordWrappers import wrap_always,wrap_onspace,wrap_onspace_strict
class constants(object):
    def __init__(self):
        self.LUMIUNIT='e30 [cm^-2]'
        self.NORM=1.0
        self.LUMIVERSION='0001'
        self.BEAMMODE='stable' #possible choices stable,quiet,either
        self.VERBOSE=False
        self.LSLENGTH=0
    def defaultfrontierConfigString(self):
        return """<frontier-connect><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier.cern.ch:3128"/><proxy url="http://cmst0frontier1.cern.ch:3128"/><proxy url="http://cmst0frontier2.cern.ch:3128"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier1.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier2.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier3.cern.ch:8000/FrontierInt"/><server url="http://cmsfrontier4.cern.ch:8000/FrontierInt"/></frontier-connect>"""
    
def lslengthsec(numorbit,numbx):
    #print numorbit, numbx
    l=numorbit*numbx*25e-09
    return l

def deliveredLumiForRun(dbsession,c,runnum):
    #
    #select sum(INSTLUMI),count(INSTLUMI) from lumisummary where runnum=124025 and lumiversion='0001';
    #apply norm factor and ls length in sec on the query result 
    #unit E27cm^-2 
    #
    #if c.VERBOSE:
    #    print 'deliveredLumiForRun : norm : ',c.NORM,' : run : ',runnum
    delivered=0.0
    totalls=0
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.tableHandle(nameDealer.lumisummaryTableName()).newQuery()
        query.addToOutputList("sum(INSTLUMI)","totallumi")
        query.addToOutputList("count(INSTLUMI)","totalls")
        queryBind=coral.AttributeList()
        queryBind.extend("runnum","unsigned int")
        queryBind.extend("lumiversion","string")
        queryBind["runnum"].setData(int(runnum))
        queryBind["lumiversion"].setData(c.LUMIVERSION)
        result=coral.AttributeList()
        result.extend("totallumi","float")
        result.extend("totalls","unsigned int")
        query.defineOutput(result)
        query.setCondition("RUNNUM =:runnum AND LUMIVERSION =:lumiversion",queryBind)
        cursor=query.execute()
        while cursor.next():
            delivereddata=cursor.currentRow()['totallumi'].data()
            totallsdata=cursor.currentRow()['totalls'].data()
            if delivereddata:
                delivered=delivereddata*c.NORM*c.LSLENGTH
                totalls=totallsdata
        del query
        dbsession.transaction().commit()
        lumidata=[]
        if delivered==0.0:
            lumidata=[str(runnum),'N/A','N/A','N/A']
        else:
            lumidata=[str(runnum),str(totalls),'%.2f'%delivered,c.BEAMMODE]
        return lumidata
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession

def deliveredLumiForRange(dbsession,c,fileparsingResult):
    #
    #in this case,only take run numbers from theinput file
    #
    lumidata=[]
    for run in fileparsingResult.runs():
        lumidata.append( deliveredLumiForRun(dbsession,c,run) )
    return lumidata

def recordedLumiForRun(dbsession,c,runnum,lslist=[]):
    recorded=0.0
    lumidata=[] #[runnumber,trgtable,deadtable]
    trgtable={} #{hltpath:[l1seed,hltprescale,l1prescale]}
    deadtable={} #{lsnum:[deadtime,instlumi,norbits]}
    lumidata.append(runnum)
    lumidata.append(trgtable)
    lumidata.append(deadtable)
    collectedseeds=[] #[(hltpath,l1seed)]
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.cmsrunsummaryTableName(),'cmsrunsummary')
        query.addToTableList(nameDealer.trghltMapTableName(),'trghltmap')
        queryCondition=coral.AttributeList()
        queryCondition.extend("runnumber","unsigned int")
        queryCondition["runnumber"].setData(int(runnum))
        query.setCondition("trghltmap.HLTKEY=cmsrunsummary.HLTKEY AND cmsrunsummary.RUNNUM=:runnumber",queryCondition)
        query.addToOutputList("trghltmap.HLTPATHNAME","hltpathname")
        query.addToOutputList("trghltmap.L1SEED","l1seed")
        result=coral.AttributeList()
        result.extend("hltpathname","string")
        result.extend("l1seed","string")
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            hltpathname=cursor.currentRow()["hltpathname"].data()
            l1seed=cursor.currentRow()["l1seed"].data()
            collectedseeds.append((hltpathname,l1seed))
        del query
        dbsession.transaction().commit()
        #loop over hltpath
        for (hname,sname) in collectedseeds:
            l1bitname=hltTrgSeedMapper.findUniqueSeed(hname,sname)
            if l1bitname:
                lumidata[1][hltpathname]=[]
                lumidata[1][hltpathname].append(l1bitname.replace('\"',''))

        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        hltprescQuery=schema.tableHandle(nameDealer.hltTableName()).newQuery()
        hltprescQuery.addToOutputList("PATHNAME","hltpath")
        hltprescQuery.addToOutputList("PRESCALE","hltprescale")
        hltprescCondition=coral.AttributeList()
        hltprescCondition.extend('runnumber','unsigned int')
        hltprescCondition.extend('cmslsnum','unsigned int')
        hltprescCondition.extend('inf','unsigned int')
        hltprescResult=coral.AttributeList()
        hltprescResult.extend('hltpath','string')
        hltprescResult.extend('hltprescale','unsigned int')
        hltprescQuery.defineOutput(hltprescResult)
        hltprescCondition['runnumber'].setData(int(runnum))
        hltprescCondition['cmslsnum'].setData(1)
        hltprescCondition['inf'].setData(0)
        hltprescQuery.setCondition("RUNNUM =:runnumber and CMSLSNUM =:cmslsnum and PRESCALE !=:inf",hltprescCondition)
        cursor=hltprescQuery.execute()
        while cursor.next():
            hltpath=cursor.currentRow()['hltpath'].data()
            hltprescale=cursor.currentRow()['hltprescale'].data()
            if lumidata[1].has_key(hltpath):
                if len(lumidata[1][hltpath])==1:
                    lumidata[1][hltpath].append(hltprescale)
        cursor.close()
        del hltprescQuery
        dbsession.transaction().commit()
                
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')
        query.addToTableList(nameDealer.trgTableName(),'trg')
        queryCondition=coral.AttributeList()
        queryCondition.extend("runnumber","unsigned int")
        queryCondition.extend("lumiversion","string")
        #queryCondition.extend("alive","bool")
        queryCondition["runnumber"].setData(int(runnum))
        queryCondition["lumiversion"].setData(c.LUMIVERSION)
        #queryCondition["alive"].setData(True)
        query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM",queryCondition)
        #query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM AND lumisummary.cmsalive=:alive AND trg.BITNUM=:bitnum",queryCondition)
        #query.addToOutputList("sum(lumisummary.INSTLUMI*(1-trg.DEADTIME/(lumisummary.numorbit*3564)))","recorded")
        query.addToOutputList("lumisummary.CMSLSNUM","cmsls")
        query.addToOutputList("lumisummary.INSTLUMI","instlumi")
        query.addToOutputList("lumisummary.NUMORBIT","norbits")
        query.addToOutputList("trg.BITNAME","bitname")
        query.addToOutputList("trg.DEADTIME","trgdeadtime")
        query.addToOutputList("trg.PRESCALE","trgprescale")
        query.addToOutputList("trg.BITNUM","trgbitnum")
        query.addToOrderList("trg.BITNAME")
        query.addToOrderList("trg.CMSLSNUM")

        result=coral.AttributeList()
        result.extend("cmsls","unsigned int")
        result.extend("instlumi","float")
        result.extend("norbits","unsigned int")
        result.extend("bitname","string")
        result.extend("trgdeadtime","unsigned long long")
        result.extend("trgprescale","unsigned int")
        result.extend("trgbitnum","unsigned int")
        trgprescalemap={}
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            cmsls=cursor.currentRow()["cmsls"].data()
            trgprescale=cursor.currentRow()["trgprescale"].data()
            trgdeadtime=cursor.currentRow()["trgdeadtime"].data()
            trgbitname=cursor.currentRow()["bitname"].data()
            trgbitnum=cursor.currentRow()["trgbitnum"].data()
            instlumi=cursor.currentRow()["instlumi"].data()
            norbits=cursor.currentRow()["norbits"].data()
            if cmsls==1:
                trgprescalemap[trgbitname]=trgprescale
            if trgbitnum==0:
                if not deadtable.has_key(cmsls):
                    deadtable[cmsls]=[]
                    deadtable[cmsls].append(trgdeadtime)
                    deadtable[cmsls].append(instlumi)
                    deadtable[cmsls].append(norbits)
        cursor.close()
        del query
        dbsession.transaction().commit()
        
        #
        #consolidate results
        #
        #trgtable
        #print trgprescalemap
        for hpath,trgdataseq in lumidata[1].items():
            bitn=trgdataseq[0]
            if trgprescalemap.has_key(bitn):
                trgdataseq.append(trgprescalemap[bitn])
        #filter selected cmsls
        lumidata[2]=filterDeadtable(deadtable,lslist)
        #if hltpath=='all':
        #    for hltname in hltTotrgMap.keys():
        #        effresult=recorded/(hltTotrgMap[hltname][1]*hltTotrgMap[hltname][2])
                #rprint.printLine('  '+hltname,effresult)
                #if c.VERBOSE:
                    #rprint.printTriggerLine(hltTotrgMap[hltname][0],hltTotrgMap[hltname][2],hltTotrgMap[hltname][1])
        #else:
        #    if hltTotrgMap.has_key(hltpath) is False:
        #        print 'Unable to calculate recorded luminosity for HLTPath ',hltpath
        #        return
        #    effresult=recorded/(hltTotrgMap[hltpath][1]*hltTotrgMap[hltpath][2])
            #rprint.printLine('  '+hltpath,effresult)
            #if c.VERBOSE:
                #rprint.printTriggerLine(hltTotrgMap[hltpath][0],hltTotrgMap[hltpath][1],hltTotrgMap[hltpath][2])
        #rprint.printOuterSeparator()
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    return lumidata

def filterDeadtable(inTable,lslist):
    if len(lslist)==0:
        return inTable
    result={}
    for existingLS in inTable.keys():
        if existingLS in lslist:
            result[existingLS]=inTable[existingLS]
    return result

def recordedLumiForRange(dbsession,c,fileparsingResult):
    #
    #in this case,only take run numbers from theinput file
    #
    lumidata=[]
    for (run,lslist) in fileparsingResult.runsandls().items():
        #print 'processing run ',run
        #print 'valid ls list ',lslist
        lumidata.append( recordedLumiForRun(dbsession,c,run,lslist) )
    return lumidata

def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Calculations")
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-n',dest='normfactor',action='store',help='normalization factor (optional, default to 1.0)')
    parser.add_argument('-r',dest='runnumber',action='store',help='run number')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file (optional)')
    parser.add_argument('-o',dest='outputfile',action='store',help='output csv file (optional)')
    parser.add_argument('-b',dest='beammode',action='store',help='beam mode, optional for delivered action, default "stable", choices "stable","quiet","either"')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',help='lumi data version, optional for all, default 0001')
    parser.add_argument('-hltpath',dest='hltpath',action='store',help='specific hltpath to calculate the recorded luminosity, default to all')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, if path undefined, fallback to cern proxy&server')
    parser.add_argument('action',choices=['overview','delivered','recorded'],help='lumi calculation types, default to overview')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose, prints additional trigger and inst lumi measurements' )
    
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
    connectparser=connectstrParser.connectstrParser(connectstring)
    connectparser.parse()
    usedefaultfrontierconfig=False
    cacheconfigpath=''
    if connectparser.needsitelocalinfo():
        if not args.siteconfpath:
            cacheconfigpath=os.environ['CMS_PATH']
            if cacheconfigpath:
                cacheconfigpath=os.path.join(cacheconfigpath,'SITECONF','local','JobConfig','site-local-config.xml')
            else:
                usedefaultfrontierconfig=True
        else:
            cacheconfigpath=args.siteconfpath
            cacheconfigpath=os.path.join(cacheconfigpath,'site-local-config.xml')
        p=cacheconfigParser.cacheconfigParser()
        if usedefaultfrontierconfig:
            p.parseString(c.defaultfrontierConfigString)
        else:
            p.parse(cacheconfigpath)
        connectstring=connectparser.fullfrontierStr(connectparser.schemaname(),p.parameterdict())
    #print 'connectstring',connectstring
    runnumber=0
    svc = coral.ConnectionService()
    isverbose=False
    if args.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
        c.VERBOSE=True
    hpath=''
    ifilename=''
    beammode='stable'
    if args.verbose :
        c.VERBOSE=True
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    if args.normfactor:
        c.NORM=float(args.normfactor)
    if args.lumiversion:
        c.LUMIVERSION=args.lumiversion
    if args.beammode:
        c.BEAMMODE=args.beammode
    if args.inputfile and len(args.inputfile)!=0:
        ifilename=args.inputfile
    if args.runnumber :
        runnumber=args.runnumber
    if len(ifilename)==0 and runnumber==0:
        raise "must specify either a run (-r) or an input run selection file (-i)"
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    inputfilecontent=''
    fileparsingResult=''
    if runnumber==0 and len(ifilename)!=0 :
        f=open(ifilename,'r')
        inputfilecontent=f.read()
        fileparsingResult=selectionParser.selectionParser(inputfilecontent)
    lumidata=[]
    if args.action == 'delivered':
        if runnumber!=0:
            lumidata.append(deliveredLumiForRun(session,c,runnumber))
        else:
            lumidata=deliveredLumiForRange(session,c,fileparsingResult)
        
        labels=[('%-*s'%(8,'run'),'%-*s'%(17,'n lumi sections'),'%-*s'%(10,'delivered'),'%-*s'%(10,'beam mode'))]
        print tablePrinter.indent(labels+lumidata,hasHeader=True,separateRows=False,prefix='| ',postfix=' |',wrapfunc=lambda x: wrap_onspace(x,10) )
        
    if args.action == 'recorded':
        if args.hltpath and len(args.hltpath)!=0:
            hpath=args.hltpath
        if runnumber!=0:
            lumidata.append(recordedLumiForRun(session,c,runnumber))
        else:
            lumidata=recordedLumiForRange(session,c,fileparsingResult)
    print lumidata
    
    del session
    del svc
if __name__=='__main__':
    main()
    

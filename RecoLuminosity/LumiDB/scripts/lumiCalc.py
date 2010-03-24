#!/usr/bin/env python
VERSION='1.02'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,selectionParser,hltTrgSeedMapper

class constants(object):
    def __init__(self):
        self.LUMIUNIT='e27 [cm^-2]'
        self.NORM=16700
        self.LUMIVERSION='0001'
        self.BEAMMODE='stable' #possible choices stable,quiet,either
        self.VERBOSE=False
        self.LSLENGTH=0
        
def lslengthsec(numorbit,numbx):
    #print numorbit, numbx
    l=numorbit*numbx*25e-09
    return l

def deliveredLumiForRun(dbsession,c,runnum):
    #
    #select sum(INSTLUMI) from lumisummary where runnum=124025 and lumiversion='0001';
    #apply norm factor and ls length in sec on the query result 
    #unit E27cm^-2 
    #
    if c.VERBOSE:
        print 'deliveredLumiForRun : norm : ',c.NORM,' : run : ',runnum
    delivered=0.0
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.tableHandle(nameDealer.lumisummaryTableName()).newQuery()
        query.addToOutputList("sum(INSTLUMI)","totallumi")
        queryBind=coral.AttributeList()
        queryBind.extend("runnum","unsigned int")
        queryBind.extend("lumiversion","string")
        queryBind["runnum"].setData(int(runnum))
        queryBind["lumiversion"].setData(c.LUMIVERSION)
        result=coral.AttributeList()
        result.extend("totallumi","float")
        query.defineOutput(result)
        query.setCondition("RUNNUM =:runnum AND LUMIVERSION =:lumiversion",queryBind)
        cursor=query.execute()
        #icount=0
        while cursor.next():
         #   if icount==0:
         delivered=cursor.currentRow()['totallumi'].data()*c.NORM*c.LSLENGTH
        del query
        dbsession.transaction().commit()
        print "Delivered Luminosity for Run "+str(runnum)+" (beam "+c.BEAMMODE+"): "+'%.3f'%(delivered)+c.LUMIUNIT
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession

    
def deliveredLumiForRange(dbsession,c,inputfile):
    #
    #in this case,only take run numbers from theinput file
    #
    if c.VERBOSE:
        print 'deliveredLumiForRange : norm : ',c.NORM,' : inputfile : ',inputfile
    
    f=open(inputfile,'r')
    content=f.read()
    s=selectionParser.selectionParser(content)
    for run in s.runs():
        deliveredLumiForRun(dbsession,c,run)
    
def recordedLumiForRun(dbsession,c,runnum):
    if c.VERBOSE:
        print 'recordedLumiForRun : run : ',runnum,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION
    #
    #LS_length=25e-9*numorbit*3564(sec)
    #LS deadfraction=deadtimecount/(numorbit*3564) 
    #select distinct lumisummary.instlumi*trg.deadtime/(lumisummary.numorbit*3564) as deadfraction from trg,lumisummary where trg.runnum=124025 and lumisummary.runnum=124025 and lumisummary.lumiversion='0001' and lumisummary.cmslsnum=1 and trg.cmslsnum=1;
    #
    #let oracle do everything!
    #
    #select sum( lumisummary.instlumi*(1-trg.deadtime/(lumisummary.numorbit*3564))) as recorded from trg,lumisummary where trg.runnum=124025 and lumisummary.runnum=124025 and lumisummary.lumiversion='0001' and lumisummary.cmslsnum=trg.cmslsnum and lumisummary.cmsalive=1 and trg.bitnum=0;
    #multiply query result by norm factor, attach unit
    #7.368e-5*16400.0=1.2083520000000001
    recorded=0.0
    lslength=0
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')
        query.addToTableList(nameDealer.trgTableName(),'trg')
        queryCondition=coral.AttributeList()
        queryCondition.extend("runnumber","unsigned int")
        queryCondition.extend("lumiversion","string")
        queryCondition.extend("alive","bool")
        queryCondition.extend("bitnum","unsigned int")
        queryCondition["runnumber"].setData(int(runnum))
        queryCondition["lumiversion"].setData(c.LUMIVERSION)
        queryCondition["alive"].setData(True)
        queryCondition["bitnum"].setData(0)
        query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM AND lumisummary.cmsalive =:alive AND trg.BITNUM=:bitnum",queryCondition)
        query.addToOutputList("sum(lumisummary.INSTLUMI*(1-trg.DEADTIME/(lumisummary.numorbit*3564)))","recorded")
        result=coral.AttributeList()
        result.extend("recorded","float")
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            recorded=cursor.currentRow()["recorded"].data()*c.NORM*c.LSLENGTH
        del query
        dbsession.transaction().commit()
        print "Recorded Luminosity for Run "+str(runnum)+" : "+'%.3f'%(recorded)+c.LUMIUNIT
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
    
def recordedLumiForRange(dbsession,c,inputfile):
    if c.VERBOSE:
        print 'recordedLumi : inputfile : ',inputfile,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION
    f=open(inputfile,'r')
    content=f.read()
    s=selectionParser.selectionParser(content)
    runsandLSStr=s.runsandlsStr()
    runsandLS=s.runsandls()
    recorded={}
    if c.VERBOSE:
        print 'recordedLumi : selected runs and LS ',runsandLS
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')
        query.addToTableList(nameDealer.trgTableName(),'trg')
        for runnumstr,LSlistStr in runsandLSStr.items():
            query.addToOutputList("sum(lumisummary.INSTLUMI*(1-trg.DEADTIME/(lumisummary.numorbit*3564)))","recorded")
            result=coral.AttributeList()
            result.extend("recorded","float")
            query.defineOutput(result)
            queryCondition=coral.AttributeList()
            queryCondition.extend("runnumber","unsigned int")
            queryCondition.extend("lumiversion","string")
            queryCondition.extend("alive","bool")
            queryCondition.extend("bitnum","unsigned int")
            realLSlist=runsandLS[int(runnumstr)]

            queryCondition["runnumber"].setData(int(runnumstr))
            queryCondition["lumiversion"].setData(c.LUMIVERSION)
            queryCondition["alive"].setData(True)
            queryCondition["bitnum"].setData(0)
            for l in realLSlist:
                queryCondition.extend(str(l),"unsigned int")
                queryCondition[str(l)].setData(int(l))
            o=[':'+x for x in LSlistStr]
            inClause='('+','.join(o)+')'
            query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM AND lumisummary.cmsalive =:alive AND trg.BITNUM=:bitnum AND lumisummary.CMSLSNUM in "+inClause,queryCondition)
            cursor=query.execute()
            while cursor.next():
                recorded[int(runnumstr)]=cursor.currentRow()['recorded'].data()
        del query
        dbsession.transaction().commit()
        for run,recd in  recorded.items():
            print "Recorded Luminosity for Run "+str(run)+" : "+'%.3f'%(recd*c.NORM*c.LSLENGTH)+c.LUMIUNIT
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
    
def effectiveLumiForRun(dbsession,c,runnum,hltpath=''):
    if len(hltpath)==0:
        hltpath='all'
    if c.VERBOSE:
        print 'effectiveLumiForRun : runnum : ',runnum,' : hltpath : ',hltpath,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION

    try:
        collectedseeds=[]
        filteredbits=[]
        finalhltData={} #{hltpath:(l1bitname,hltprescale)}
        hltTotrgMap={} #{hltpath:(l1bitname,hltprescale,l1prescale,[(lsnum,l1deadfrac)])}
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
        
        for ip in collectedseeds:
            l1bitname=hltTrgSeedMapper.findUniqueSeed(ip[0],ip[1])
            if l1bitname:
                filteredbits.append((ip[0],l1bitname.replace('\"','')))#strip quotes if any
        #print "found ",len(filteredbits)," calculable hltpaths"
        #print "filtered result : ",filteredbits

        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        for h in filteredbits:
            hltprescQuery=schema.tableHandle(nameDealer.hltTableName()).newQuery()
            hltprescQuery.addToOutputList("PRESCALE","hltprescale")
            hltprescCondition=coral.AttributeList()
            hltprescCondition.extend('runnumber','unsigned int')
            hltprescCondition.extend('pathname','string')
            hltprescCondition.extend('cmslsnum','unsigned int')
            hltprescCondition.extend('inf','unsigned int')
            hltprescResult=coral.AttributeList()
            hltprescResult.extend('hltprescale','unsigned int')
            hltprescQuery.defineOutput(hltprescResult)
            hltprescCondition['runnumber'].setData(int(runnum))
            hltprescCondition['pathname'].setData(h[0])
            hltprescCondition['cmslsnum'].setData(1)
            hltprescCondition['inf'].setData(0)
            hltprescQuery.setCondition("RUNNUM =:runnumber AND PATHNAME =:pathname and CMSLSNUM =:cmslsnum and PRESCALE !=:inf",hltprescCondition)
            cursor=hltprescQuery.execute()
            while cursor.next():
                hltprescale=cursor.currentRow()['hltprescale'].data()
                #print 'hlt prescale for '+h[0]+' : ',str(prescale)
                finalhltData[h[0]]=(h[1],hltprescale)
            cursor.close()
            del hltprescQuery
        dbsession.transaction().commit()

        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        for myhltpath,(myl1bitname,myhltprescale) in finalhltData.items():
            #print 'querying here ',myhltpath,myl1bitname,myhltprescale
            trgQuery=schema.tableHandle(nameDealer.trgTableName()).newQuery()
            trgQuery.addToOutputList("CMSLSNUM","cmslsnum")
            trgQuery.addToOutputList("PRESCALE","trgprescale")
            trgQuery.addToOutputList("DEADTIME","trgdeadtime")
            trgQueryCondition=coral.AttributeList()
            trgQueryCondition.extend('runnumber','unsigned int')
            trgQueryCondition.extend('bitname','string')
            trgQueryCondition['runnumber'].setData(int(runnum))
            trgQueryCondition['bitname'].setData(myl1bitname)
            trgResult=coral.AttributeList()
            trgResult.extend("cmslsnum","unsigned int")
            trgResult.extend("trgprescale","unsigned int")
            trgResult.extend("trgdeadtime","unsigned long long")
            trgQuery.defineOutput(trgResult)
            trgQuery.setCondition("RUNNUM =:runnumber AND BITNAME =:bitname order by CMSLSNUM",trgQueryCondition)
            cursor=trgQuery.execute()
            counter=0
            while cursor.next():
                trglsnum=cursor.currentRow()['cmslsnum'].data()
                trgprescale=cursor.currentRow()['trgprescale'].data()
                trgdeadtime=cursor.currentRow()['trgdeadtime'].data()
                #print myhltpath,myl1bitname,myhltprescale,trgprescale
                if counter==0:
                    hltTotrgMap[myhltpath]=(myl1bitname,myhltprescale,trgprescale,[])
                hltTotrgMap[myhltpath][-1].append((trglsnum,'%.3f'%(25.0e-09*trgdeadtime/c.LSLENGTH)))
                counter=counter+1
            cursor.close()
            del trgQuery
        dbsession.transaction().commit()
        #print 'hltTotrgMap : ',hltTotrgMap
       
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        query=schema.newQuery()
        query.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')
        query.addToTableList(nameDealer.trgTableName(),'trg')
        queryCondition=coral.AttributeList()
        queryCondition.extend("runnumber","unsigned int")
        queryCondition.extend("lumiversion","string")
        queryCondition.extend("alive","bool")
        queryCondition.extend("bitnum","unsigned int")
        queryCondition["runnumber"].setData(int(runnum))
        queryCondition["lumiversion"].setData(c.LUMIVERSION)
        queryCondition["alive"].setData(True)
        queryCondition["bitnum"].setData(0)
        query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM AND lumisummary.cmsalive =:alive AND trg.BITNUM=:bitnum",queryCondition)
        query.addToOutputList("sum(lumisummary.INSTLUMI*(1-trg.DEADTIME/(lumisummary.numorbit*3564)))","recorded")
        result=coral.AttributeList()
        result.extend("recorded","float")
        query.defineOutput(result)
        cursor=query.execute()
        while cursor.next():
            recorded=cursor.currentRow()["recorded"].data()*c.NORM*c.LSLENGTH
        del query
        dbsession.transaction().commit()

        print 'Effective Luminosity for Run '+str(runnum)
        #print 'requested hltpath ',hltpath
        if hltpath=='all':
            for hltname in hltTotrgMap.keys():
                effresult=recorded/(hltTotrgMap[hltname][1]*hltTotrgMap[hltname][2])
                print '    '+hltname+' : '+'%.3f'%(effresult)+c.LUMIUNIT
                if c.VERBOSE:
                    print '     ### L1 :'+str(hltTotrgMap[hltname][0])+', HLT Prescale : '+str(hltTotrgMap[hltname][1])+', L1 Prescale : '+str(hltTotrgMap[hltname][2])+', Deadfrac : ',hltTotrgMap[hltname][3]
        else:
            if hltTotrgMap.has_key(hltpath) is False:
                print 'Unable to calculate effective luminosity for HLTPath ',hltpath
                return
            effresult=recorded/(hltTotrgMap[hltpath][1]*hltTotrgMap[hltpath][2])
            print '    '+hltpath+' : '+'%.3f'%(effresult)+c.LUMIUNIT
            if c.VERBOSE:
                print '     ### L1 :'+str(hltTotrgMap[hltpath][0])+', HLT Prescale : '+str(hltTotrgMap[hltpath][1])+', L1 Prescale : '+str(hltTotrgMap[hltpath][2])+', Deadfrac : '+str(hltTotrgMap[hltpath][3])
                
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def effectiveLumiForRange(dbsession,c,inputfile,hltpath=''):
    if len(hltpath)==0:
        hltpath='all'
    if c.VERBOSE:
        print 'effectiveLumiForRange : inputfile : ',inputfile,' : hltpath : ',hltpath,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION
    f=open(inputfile,'r')
    content=f.read()
    s=selectionParser.selectionParser(content)
    runsandLSStr=s.runsandlsStr()
    runsandLS=s.runsandls()
    recorded={}
    hltTotrgMapAllRuns={}
    try:
        for runnumstr,LSlistStr in runsandLSStr.items():
            dbsession.transaction().start(True)
            schema=dbsession.nominalSchema()
            query=schema.newQuery()
            query.addToTableList(nameDealer.lumisummaryTableName(),'lumisummary')
            query.addToTableList(nameDealer.trgTableName(),'trg')
            query.addToOutputList("sum(lumisummary.INSTLUMI*(1-trg.DEADTIME/(lumisummary.numorbit*3564)))","recorded")
            result=coral.AttributeList()
            result.extend("recorded","float")
            query.defineOutput(result)
            queryCondition=coral.AttributeList()
            queryCondition.extend("runnumber","unsigned int")
            queryCondition.extend("lumiversion","string")
            queryCondition.extend("alive","bool")
            queryCondition.extend("bitnum","unsigned int")
            realLSlist=runsandLS[int(runnumstr)]
            queryCondition["runnumber"].setData(int(runnumstr))
            queryCondition["lumiversion"].setData(c.LUMIVERSION)
            queryCondition["alive"].setData(True)
            queryCondition["bitnum"].setData(0)
            for l in realLSlist:
                queryCondition.extend(str(l),"unsigned int")
                queryCondition[str(l)].setData(int(l))
            o=[':'+x for x in LSlistStr]
            inClause='('+','.join(o)+')'
            query.setCondition("trg.RUNNUM =:runnumber AND lumisummary.RUNNUM=:runnumber and lumisummary.LUMIVERSION =:lumiversion AND lumisummary.CMSLSNUM=trg.CMSLSNUM AND lumisummary.cmsalive =:alive AND trg.BITNUM=:bitnum AND lumisummary.CMSLSNUM in "+inClause,queryCondition)
            cursor=query.execute()
            while cursor.next():
                recorded[int(runnumstr)]=cursor.currentRow()['recorded'].data()*c.NORM*c.LSLENGTH
            del query
            dbsession.transaction().commit()

            #start hlt and trg queries
            collectedseeds=[]
            filteredbits=[]
            finalhltData={} #{hltpath:(l1bitname,hltprescale)}
            hltTotrgMap={} #{hltpath:(l1bitname,hltprescale,l1prescale,[(lsnum,l1deadfrac)])}
            dbsession.transaction().start(True)
            schema=dbsession.nominalSchema()
            query=schema.newQuery()
            query.addToTableList(nameDealer.cmsrunsummaryTableName(),'cmsrunsummary')
            query.addToTableList(nameDealer.trghltMapTableName(),'trghltmap')
            queryCondition=coral.AttributeList()
            queryCondition.extend("runnumber","unsigned int")
            queryCondition["runnumber"].setData(int(runnumstr))
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
        
            for ip in collectedseeds:
                l1bitname=hltTrgSeedMapper.findUniqueSeed(ip[0],ip[1])
                if l1bitname:
                    filteredbits.append((ip[0],l1bitname.replace('\"','')))

            dbsession.transaction().start(True)
            schema=dbsession.nominalSchema()
            for h in filteredbits:
                hltprescQuery=schema.tableHandle(nameDealer.hltTableName()).newQuery()
                hltprescQuery.addToOutputList("PRESCALE","hltprescale")
                hltprescCondition=coral.AttributeList()
                hltprescCondition.extend('runnumber','unsigned int')
                hltprescCondition.extend('pathname','string')
                hltprescCondition.extend('cmslsnum','unsigned int')
                hltprescCondition.extend('inf','unsigned int')
                hltprescResult=coral.AttributeList()
                hltprescResult.extend('hltprescale','unsigned int')
                hltprescQuery.defineOutput(hltprescResult)
                hltprescCondition['runnumber'].setData(int(runnumstr))
                hltprescCondition['pathname'].setData(h[0])
                hltprescCondition['cmslsnum'].setData(1)
                hltprescCondition['inf'].setData(0)
                hltprescQuery.setCondition("RUNNUM =:runnumber AND PATHNAME =:pathname and CMSLSNUM =:cmslsnum and PRESCALE !=:inf",hltprescCondition)
                cursor=hltprescQuery.execute()
                while cursor.next():
                    hltprescale=cursor.currentRow()['hltprescale'].data()
                    finalhltData[h[0]]=(h[1],hltprescale)
                cursor.close()
                del hltprescQuery
            dbsession.transaction().commit()

            dbsession.transaction().start(True)
            schema=dbsession.nominalSchema()
            for myhltpath,(myl1bitname,myhltprescale) in finalhltData.items():
                trgQuery=schema.tableHandle(nameDealer.trgTableName()).newQuery()
                trgQuery.addToOutputList("CMSLSNUM","cmslsnum")
                trgQuery.addToOutputList("PRESCALE","trgprescale")
                trgQuery.addToOutputList("DEADTIME","trgdeadtime")
                trgQueryCondition=coral.AttributeList()
                trgQueryCondition.extend('runnumber','unsigned int')
                trgQueryCondition.extend('bitname','string')
                trgQueryCondition['runnumber'].setData(int(runnumstr))
                trgQueryCondition['bitname'].setData(myl1bitname)
                trgResult=coral.AttributeList()
                trgResult.extend("cmslsnum","unsigned int")
                trgResult.extend("trgprescale","unsigned int")
                trgResult.extend("trgdeadtime","unsigned long long")
                trgQuery.defineOutput(trgResult)
                trgQuery.setCondition("RUNNUM =:runnumber AND BITNAME =:bitname order by CMSLSNUM",trgQueryCondition)
                cursor=trgQuery.execute()
                counter=0
                while cursor.next():
                    trglsnum=cursor.currentRow()['cmslsnum'].data()
                    trgprescale=cursor.currentRow()['trgprescale'].data()
                    trgdeadtime=cursor.currentRow()['trgdeadtime'].data()
                    if counter==0:
                        hltTotrgMap[myhltpath]=(myl1bitname,myhltprescale,trgprescale,[])
                    hltTotrgMap[myhltpath][-1].append('%.3f'%(trglsnum,25.0e-09*trgdeadtime/c.LSLENGTH))
                    counter=counter+1
                cursor.close()
                del trgQuery
            dbsession.transaction().commit()
            hltTotrgMapAllRuns[int(runnumstr)]=hltTotrgMap
       # print 'recorded '
       # print recorded
       # print 'hltTotrgMap all runs '
       # print hltTotrgMapAllRuns
        if len(recorded)!=len(hltTotrgMapAllRuns):
            raise "inconsistent number of runs in recorded and hltTotrgMap result"

        for run in recorded.keys():
             print 'Effective Luminosity for Run '+str(run)
             if hltpath=='all':
                 for hltname in hltTotrgMapAllRuns[run].keys():
                     effresult=recorded[run]/(hltTotrgMapAllRuns[run][hltname][1]*hltTotrgMapAllRuns[run][hltname][2])
                     print '    '+hltname+' : ','%.3f'%(effresult)+c.LUMIUNIT
                 if c.VERBOSE:
                     print '     ### L1 :'+str(hltTotrgMapAllRuns[run][hltname][0])+', HLT Prescale : '+str(hltTotrgMapAllRuns[run][hltname][1])+', L1 Prescale : '+str(hltTotrgMapAllRuns[run][hltname][2])+', Deadtime : '+str(hltTotrgMapAllRuns[run][hltname][3])
             else:
                 if hltTotrgMapAllRuns[run].has_key(hltpath) is False:
                     print 'Unable to calculate effective luminosity for HLTPath ',hltpath
                     return
                 effresult=recorded[run]/(hltTotrgMapAllRuns[run][hltpath][1]*hltTotrgMapAllRuns[run][hltpath][2])
                 print '    '+hltpath+' : ','%.3f'%(effresult)+c.LUMIUNIT
                 if c.VERBOSE:
                     print '     ### L1 :'+str(hltTotrgMapAllRuns[run][hltpath][0])+', HLT Prescale : '+str(hltTotrgMapAllRuns[run][hltpath][1])+', L1 Prescale : '+str(hltTotrgMapAllRuns[run][hltpath][2])+', Deadtime : '+str(hltTotrgMapAllRuns[run][hltpath][3])
                
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Lumi Calculations")
    # add required arguments
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    # add optional arguments
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-n',dest='normfactor',action='store',help='normalization factor')
    parser.add_argument('-r',dest='runnumber',action='store',help='run number')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file, optional for recorded and effective actions, not taken by delivered action')
    parser.add_argument('-b',dest='beammode',action='store',help='beam mode, optional for delivered action, default "stable", choices "stable","quiet","either"')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',help='lumi data version, optional for all, default 0001')
    parser.add_argument('-hltpath',dest='hltpath',action='store',help='specific hltpath to calculate the effective luminosity, default to all')
    parser.add_argument('action',choices=['delivered','recorded','effective'],help='lumi calculation types')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
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
        c.NORM=args.normfactor
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
    
    #
    #one common query on the number of orbits and check if the run is available in db
    #
    try:
        session.transaction().start(True)
        schema=session.nominalSchema()
        query=schema.tableHandle(nameDealer.lumisummaryTableName()).newQuery()
        query.addToOutputList("NUMORBIT","numorbit")
        queryBind=coral.AttributeList()
        queryBind.extend("runnum","unsigned int")
        queryBind.extend("lumiversion","string")
        queryBind["runnum"].setData(int(runnumber))
        queryBind["lumiversion"].setData(c.LUMIVERSION)
        result=coral.AttributeList()
        result.extend("numorbit","unsigned int")
        query.defineOutput(result)
        query.setCondition("RUNNUM =:runnum AND LUMIVERSION =:lumiversion",queryBind)
        query.limitReturnedRows(1)
        cursor=query.execute()
        icount=0
        while cursor.next():
            c.LSLENGTH=lslengthsec(cursor.currentRow()['numorbit'].data(),3564)
            icount=icount+1
        del query
        session.transaction().commit()
        if icount==0:
            print 'Run ',runnumber,' does not exist in LumiDB, do nothing...'
            return
    except Exception,e:
        print str(e)
        session.transaction().rollback()
        del session
    
    if args.action == 'delivered':
        if runnumber!=0:
            deliveredLumiForRun(session,c,runnumber)
        else:
            deliveredLumiForRange(session,c,ifilename);
    if args.action == 'recorded':
        if runnumber!=0:
            recordedLumiForRun(session,c,runnumber)
        else:
            recordedLumiForRange(session,c,ifilename)
    if args.action == 'effective':
        if args.hltpath and len(args.hltpath)!=0:
            hpath=args.hltpath
        if runnumber!=0:
            effectiveLumiForRun(session,c,runnumber,hpath)
        else:
            effectiveLumiForRange(session,c,ifilename,hpath)
    del session
    del svc
if __name__=='__main__':
    main()
    

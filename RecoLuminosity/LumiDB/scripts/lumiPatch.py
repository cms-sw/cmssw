#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
from RecoLuminosity.LumiDB import argparse,selectionParser,csvSelectionParser

'''
--on wbm db
select  LUMISEGMENTNR,DEADTIMEBEAMACTIVE from cms_wbm.LEVEL1_TRIGGER_CONDITIONS where RUNNUMBER=:runnumber order by LUMISEGMENTNR;
--on lumidb  
update TRG set DEADTIME=:deadtimebeamactive where RUNNUM=:runnum and CMSLSNUM=:lsnum
--reapply calibration to inst lumi
update LUMISUMMARY set INSTLUMI=1.006319*INSTLUMI where RUNNUM in (1,3,57,90)
'''

class constants(object):
    def __init__(self):
        self.debug=False
        self.isdryrun=None
        self.wbmschema='CMS_WBM'
        self.wbmdeadtable='LEVEL1_TRIGGER_CONDITIONS'
        self.gtmonschema='CMS_GT_MON'
        self.gtdeadview='GT_MON_TRIG_DEAD_VIEW'
        self.lumitrgtable='TRG'
        self.lumisummarytable='LUMISUMMARY'
        self.runsummarytable='CMSRUNSUMMARY'
        
def recalibrateLumiForRun(dbsession,c,delta,runnums):
    '''
    update LUMISUMMARY set INSTLUMI=:delta*INSTLUMI where RUNNUM in (1,3,57,90)
    '''
    updaterows=0
    try:
        dbsession.transaction().start(False)
        schema=dbsession.nominalSchema()
        if not schema:
            raise 'cannot connect to schema'
        if not schema.existsTable(c.lumisummarytable):
            raise 'non-existing table '+c.lumisummarytable
        runliststring=','.join([str(x) for x in runnums])
        print 'applying delta '+delta+' on run list '+runliststring
        nchanged=0
        inputData=coral.AttributeList()
        inputData.extend('delta','float')
        inputData['delta'].setData(float(delta))
        nchanged=schema.tableHandle(c.lumisummarytable).dataEditor().updateRows('INSTLUMI=INSTLUMI*:delta','RUNNUM in ('+runliststring+')',inputData)
        print 'total number of row changed ',nchanged
        if c.isdryrun:
            dbsession.transaction().rollback()
        else:
            dbsession.transaction().commit()
        return nchanged
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def GTdeadtimeBeamActiveForRun(dbsession,c,runnum):
    '''
    select lsnr,counts from cms_gt_mon.gt_mon_trig_dead_view where runnr=:runnumber and deadcounter='DeadtimeBeamActive' order by lsnr;
    return result{lumisection:deadtimebeamactive}
    
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.schema(c.gtmonschema)

        if not schema:
            raise Exception('cannot connect to schema '+c.gtmonschema)
        if not schema.existsView(c.gtdeadview):
            raise Exception('non-existing view '+c.gtdeadview)

        deadOutput=coral.AttributeList()
        deadOutput.extend("lsnr","unsigned int")
        deadOutput.extend("deadcount","unsigned long long")
        
        deadBindVarList=coral.AttributeList()
        deadBindVarList.extend("runnumber","unsigned int")
        deadBindVarList.extend("countername","string")
        deadBindVarList["runnumber"].setData(int(runnum))
        deadBindVarList["countername"].setData('DeadtimeBeamActive')
        
        query=schema.newQuery()
        query.addToTableList(c.gtdeadview)
        query.addToOutputList('LSNR','lsnr')
        query.addToOutputList('COUNTS','deadcount')
        query.setCondition('RUNNR=:runnumber AND DEADCOUNTER=:countername',deadBindVarList)
        query.addToOrderList('lsnr')
        query.defineOutput(deadOutput)

        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            deadcount=cursor.currentRow()['deadcount'].data()
            result[cmslsnum]=deadcount
            #print 'deadcount',deadcount
        del query
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def WBMdeadtimeBeamActiveForRun(dbsession,c,runnum):
    '''
    select  LUMISEGMENTNR,DEADTIMEBEAMACTIVE from cms_wbm.LEVEL1_TRIGGER_CONDITIONS where RUNNUMBER=:runnum order by LUMISEGMENTNR;
    return result{lumisection:deadtimebeamactive}
    
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        if not schema:
            raise Exception('cannot connect to schema'+c.wbmschema)
        if not schema.existsTable(c.wbmdeadtable):
            raise Exception('non-existing table'+c.wbmdeadtable)

        deadOutput=coral.AttributeList()
        deadOutput.extend("lsnr","unsigned int")
        deadOutput.extend("deadcount","unsigned long long")
        
        deadBindVarList=coral.AttributeList()
        deadBindVarList.extend("runnum","unsigned int")
        deadBindVarList["runnum"].setData(int(runnum))
        
        query=schema.newQuery()
        query.addToTableList(c.wbmdeadtable)
        query.addToOutputList('LUMISEGMENTNR','lsnr')
        query.addToOutputList('DEADTIMEBEAMACTIVE','deadcount')
        query.setCondition('RUNNUMBER=:runnum',deadBindVarList)
        query.addToOrderList('LUMISEGMENTNR')
        query.defineOutput(deadOutput)
        
        cursor=query.execute()
        while cursor.next():
            cmslsnum=cursor.currentRow()['lsnr'].data()
            deadcount=cursor.currentRow()['deadcount'].data()
            result[cmslsnum]=deadcount
            #print 'deadcount',deadcount
        del query
        return result
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def patchDeadtimeForRun(dbsession,c,runnum,deadtimeDict):
    '''
    input: deadtimeDict{ls:deadtimebeamactive}
    loop over input
    update TRG set DEADTIME=:deadtimebeamactive where RUNNUM=:runnum and CMSLSNUM=:lsnum
    output: number of rows changed
    '''
    totalchanged=0
    try:
        dbsession.transaction().start(False)
        schema=dbsession.nominalSchema()
        if not schema:
            raise Exception('cannot connect to schema ')
        if not schema.existsTable(c.lumitrgtable):
            raise Exception('non-existing table '+c.lumitrgtable)
        for lsnum,deadtimebeamactive in deadtimeDict.items():
            nchanged=0
            inputData=coral.AttributeList()
            inputData.extend('deadtimebeamactive','unsigned int')
            inputData.extend('runnum','unsigned int')
            inputData.extend('lsnum','unsigned int')
            inputData['deadtimebeamactive'].setData(deadtimebeamactive)
            inputData['runnum'].setData(runnum)
            inputData['lsnum'].setData(lsnum)
            nchanged=schema.tableHandle(c.lumitrgtable).dataEditor().updateRows('DEADTIME=:deadtimebeamactive','RUNNUM=:runnum AND CMSLSNUM=:lsnum',inputData)
            print 'rows changed for ls ',str(lsnum),str(nchanged)
            totalchanged+=nchanged
        dbsession.transaction().commit()
        return totalchanged
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession
        
def main():
    c=constants()
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Patch LumiData")
    parser.add_argument('-c',dest='destination',action='store',required=True,help='destination lumi db (required)')
    parser.add_argument('-s',dest='source',action='store',required=False,help='source db (required except for lumicalib)')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file (required)')
    parser.add_argument('-r',dest='runnumber',action='store',required=False,help='run number (optional)')
    parser.add_argument('-i',dest='inputfile',action='store',required=False,help='run selection file(optional)')
    parser.add_argument('-delta',dest='delta',action='store',required=False,help='calibration factor wrt old data in lumiDB (required for lumicalib)')
    parser.add_argument('action',choices=['deadtimeGT','deadtimeWBM','lumicalib','runtimestamp'],help='deadtimeGT: patch deadtime to deadtimebeamactive,\ndeadtimeWBM: patch deadtimeWBM to deadtimebeamactive,\nlumicalib: recalibrate inst lumi by delta where delta>1,\nruntimestamp: fill startrun timestamp if empty')
    parser.add_argument('--dryrun',dest='dryrun',action='store_true',help='only print datasource query result, do not update destination')
    
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=args.runnumber
    destConnect=args.destination
    sourceConnect=args.source
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    svc=coral.ConnectionService()
    sourcesession=None
    if sourceConnect:
        sourcesession=svc.connect(sourceConnect,accessMode=coral.access_ReadOnly)
        sourcesession.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
        sourcesession.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    destsession=svc.connect(destConnect,accessMode=coral.access_Update)
    destsession.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    destsession.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    if args.dryrun:
        c.isdryrun=True
    else:
        c.isdryrun=False
        
    deadresult={}

    if args.action == 'deadtimeGT':
        if not sourceConnect:
            raise Exception('deadtimeGT action requies -s option for source connection string')
        deadresult=GTdeadtimeBeamActiveForRun(sourcesession,c,runnumber)
        print 'reading from ',sourceConnect
        print 'run : ',runnumber
        print 'LS:deadtimebeamactive'
        #print deadresult
        if deadresult and len(deadresult)!=0:
            for cmsls,deadtimebeamactive in deadresult.items():
                print cmsls,deadtimebeamactive
        else:
            print 'no deadtime found for run ',runnumber
            print 'exit'
            return
        print 'total LS: ',len(deadresult)
#        if len(deadresult)!=max( [ (deadresult[x],x) for x in deadresult] )[1]:
        if len(deadresult)!=max( [ x for x in deadresult.keys() ] ):
            print 'total ls: ',len(deadresult)
            #print 'max key: ',max( [ x for x in deadresult.keys()])
            print 'alert: missing Lumi Sections in the middle'
            for x in range(1,max( [ x for x in deadresult.keys()] ) ):
                if not deadresult.has_key(x):
                    print 'filling up LS deadtime with 0: LS : ',x
                    deadresult[x]=0
        #print deadresult
        if not args.dryrun:
            print 'updating ',destConnect
            nupdated=patchDeadtimeForRun(destsession,c,int(runnumber),deadresult)
            print 'number of updated rows ',nupdated
    elif args.action == 'deadtimeWBM':
        if not sourceConnect:
            raise Exception('deadtimeWBM action requies -s option for source connection string')
        deadresult=WBMdeadtimeBeamActiveForRun(sourcesession,c,runnumber)
        print 'reading from ',sourceConnect
        print 'run : ',runnumber
        print 'LS:deadtimebeamactive'
        #print deadresult
        if deadresult and len(deadresult)!=0:
            for cmsls,deadtimebeamactive in deadresult.items():
                print cmsls,deadtimebeamactive
        else:
            print 'no deadtime found for run ',runnumber
            print 'exit'
            return
        print 'total LS: ',len(deadresult)
        if len(deadresult)!=max( [ (deadresult[x],x) for x in deadresult])[1]:
            print 'alert: missing Lumi Sections in the middle'
            for x in range(1,max( [ (deadresult[x],x) for x in deadresult])[1]):
                if not deadresult.has_key(x):
                    print 'filling up LS deadtime with 0: LS : ',x
                    deadresult[x]=0
        print deadresult
        if not args.dryrun:
            print 'updating ',destConnect
            nupdated=patchDeadtimeForRun(destsession,c,int(runnumber),deadresult)
            print 'number of updated rows ',nupdated
    elif args.action == 'lumicalib':
        if not args.delta or args.delta==0:
            raise Exception('Must provide non-zero -delta argument')
        runnums=[]
        if args.runnumber:
            runnums.append(args.runnumber)
        elif args.inputfile:
            basename,extension=os.path.splitext(args.inputfile)
            if extension=='.csv':#if file ends with .csv,use csv parser,else parse as json file
                fileparsingResult=csvSelectionParser.csvSelectionParser(args.inputfile)            
            else:
                f=open(args.inputfile,'r')
                inputfilecontent=f.read()
                fileparsingResult=selectionParser.selectionParser(inputfilecontent)
            if not fileparsingResult:
                raise Exception('failed to parse the input file '+ifilename)
            #print fileparsingResult.runsandls()
            runnums=fileparsingResult.runs()
            #print runnums
        else:
            raise Exception('Must provide -r or -i argument as input')
        nupdated=recalibrateLumiForRun(destsession,c,args.delta,runnums)
    if sourcesession:  
        del sourcesession
    del destsession
    del svc
        
if __name__=='__main__':
    main()
    

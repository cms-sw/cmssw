#!/usr/bin/env python
VERSION='1.00'
import os,sys
import coral
import json,csv
#import optparse
from RecoLuminosity.LumiDB import inputFilesetParser,selectionParser,csvReporter,argparse,CommonUtil,dbUtil,nameDealer,lumiQueryAPI 

def getValidationData(dbsession,run=None,cmsls=None):
    '''retrieve validation data per run or all
    input: runnum, if not runnum, retrive all
    output: {run:[[cmslsnum,flag,comment]]}
    '''
    result={}
    try:
        dbsession.transaction().start(True)
        schema=dbsession.nominalSchema()
        queryHandle=dbsession.nominalSchema().newQuery()
        result=lumiQueryAPI.validation(queryHandle,run,cmsls)
        del queryHandle
        dbsession.transaction().commit()
    except Exception, e:
        dbsession.transaction().rollback()
        del dbsession
        raise Exception, 'lumiValidate.getValidationData:'+str(e)
    return result

def insertupdateValidationData(dbsession,data):
    '''
    input: data {run:[[ls,status,comment]]}
    '''
    toreplacenocomment=[]#[[('RUNNUM',runnum),('CMSLSNUM',cmslsnum),('FLAG',flag)],[]]
    toreplacewcomment=[]#[[('RUNNUM',runnum),('CMSLSNUM',cmslsnum),('FLAG',flag),('COMMENT',comment)],[]]
    toinsert=[] #[[('RUNNUM',runnum),('CMSLSNUM',cmslsnum),('FLAG',flag),('COMMENT',comment)],[]]
    try:
        dbsession.transaction().start(True)
        dbutil=dbUtil.dbUtil(dbsession.nominalSchema())
        for run,alllsdata in data.items():
            lsselection=[]
            if len(alllsdata)==0:#cross query lumisummary to get all the cmslsnum for this run,then insert all to default
                queryHandle=dbsession.nominalSchema().newQuery()
                lumisummaryData=lumiQueryAPI.lumisummaryByrun(queryHandle,run,'0001')
                del queryHandle
                for lumisummary in lumisummaryData:
                    if lumisummary[-1]==1:#cmsalive
                        lsselection.append([lumisummary[0],'UNKNOWN','NA'])
            else:
                lsselection=alllsdata
            if len(lsselection)==0:
                print 'no LS found for run '+str(run)+' do nothing'
                continue
            for lsdata in lsselection:
                condition='RUNNUM=:runnum AND CMSLSNUM=:cmslsnum'
                conditionDefDict={}
                conditionDefDict['runnum']='unsigned int'
                conditionDefDict['cmslsnum']='unsigned int'
                conditionDict={}
                conditionDict['runnum']=run
                conditionDict['cmslsnum']=lsdata[0]
                if dbutil.existRow(nameDealer.lumivalidationTableName(),condition,conditionDefDict,conditionDict):
                    if len(lsdata)>2 and lsdata[2]:
                        toreplacewcomment.append([('runnum',run),('cmslsnum',lsdata[0]),('flag',lsdata[1]),('comment',lsdata[2])])
                    else: 
                        toreplacenocomment.append([('runnum',run),('cmslsnum',lsdata[0]),('flag',lsdata[1]),('comment','')])
                else:
                    if len(lsdata)>2 and lsdata[2]:
                        toinsert.append([('RUNNUM',run),('CMSLSNUM',lsdata[0]),('FLAG',lsdata[1]),('COMMENT',lsdata[2])])
                    else:
                        toinsert.append([('RUNNUM',run),('CMSLSNUM',lsdata[0]),('FLAG',lsdata[1]),('COMMENT','N/A')])
        dbsession.transaction().commit()
        #print 'toreplace with comment ',toreplacewcomment
        #print 'toreplace without comment ',toreplacenocomment
        #print 'toinsert ',toinsert
        #perform insert
        if len(toinsert)!=0:
            dbsession.transaction().start(False)
            dbutil=dbUtil.dbUtil(dbsession.nominalSchema())
            tabrowDef=[]
            tabrowDef.append(('RUNNUM','unsigned int'))
            tabrowDef.append(('CMSLSNUM','unsigned int'))
            tabrowDef.append(('FLAG','string'))
            tabrowDef.append(('COMMENT','string'))
            dbutil.bulkInsert(nameDealer.lumivalidationTableName(),tabrowDef,toinsert)
            dbsession.transaction().commit()
        #perform update with comment
        if len(toreplacewcomment)!=0:
            dbsession.transaction().start(False)
            dbutil=dbUtil.dbUtil(dbsession.nominalSchema())
            updateAction='FLAG=:flag,COMMENT=:comment'
            updateCondition='RUNNUM=:runnum and CMSLSNUM=:cmslsnum'
            bindvarDef=[]        
            bindvarDef.append(('flag','string'))
            bindvarDef.append(('comment','string'))
            bindvarDef.append(('runnum','unsigned int'))
            bindvarDef.append(('cmslsnum','unsigned int'))
            dbutil.updateRows(nameDealer.lumivalidationTableName(),updateAction,updateCondition,bindvarDef,toreplacewcomment)
            dbsession.transaction().commit()
        #perform update with NO comment
        if len(toreplacenocomment)!=0:
            dbsession.transaction().start(False)
            dbutil=dbUtil.dbUtil(dbsession.nominalSchema())
            updateAction='FLAG=:flag'
            updateCondition='RUNNUM=:runnum and CMSLSNUM=:cmslsnum'
            bindvarDef=[]        
            bindvarDef.append(('flag','string'))
            bindvarDef.append(('runnum','unsigned int'))
            bindvarDef.append(('cmslsnum','unsigned int'))
            dbutil.updateRows(nameDealer.lumivalidationTableName(),updateAction,updateCondition,bindvarDef,toreplacenocomment)
            dbsession.transaction().commit()
    except Exception, e:
        dbsession.transaction().rollback()
        del dbsession
        raise Exception, 'lumiValidate.insertupdateValidationData:'+str(e)
    
##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Validation",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['batchupdate','update','dump']
    allowedFlags = ['UNKNOWN','GOOD','BAD','SUSPECT']
    # parse arguments
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='path to authentication file')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file,required for batchupdate action')
    parser.add_argument('-o',dest='outputfile',action='store',help='output to csv file')
    parser.add_argument('-r',dest='runnumber',action='store',type=int,help='run number,optional')
    parser.add_argument('-runls',dest='runls',action='store',help='selection string,optional. Example [1234:[],4569:[1,1],[2,100]]')
    parser.add_argument('-flag',dest='flag',action='store',default='UNKNOWN',help='flag string,optional')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.flag.upper() not in allowedFlags:
        print 'unrecognised flag ',options.flag.upper()
        raise
    os.environ['CORAL_AUTH_PATH'] = options.authpath
    connectstring=options.connect
    svc = coral.ConnectionService()
    msg=coral.MessageStream('')
    if options.debug:
        msg.setMsgVerbosity(coral.message_Level_Debug)
    else:
        msg.setMsgVerbosity(coral.message_Level_Error)
    session=svc.connect(connectstring,accessMode=coral.access_Update)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    
    result={}#parsing result {run:[[ls,status,comment]]}
    if options.debug :
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    
    if options.action=='batchupdate':
        #populate from csv file, require -i argument
        if not options.inputfile:
            print 'inputfile -i option is required for batchupdate'
            raise
        csvReader=csv.reader(open(options.inputfile),delimiter=',')
        for row in csvReader:
            if len(row)==0:
                continue
            fieldrun=str(row[0]).strip()
            fieldls=str(row[1]).strip()
            fieldstatus=row[2]
            fieldcomment=row[3]
            if not result.has_key(int(fieldrun)):
                result[int(fieldrun)]=[]
            result[int(fieldrun)].append([int(fieldls),fieldstatus,fieldcomment])
        insertupdateValidationData(session,result)
    if options.action=='update':
        #update flag interactively, require -runls argument
        #runls={run:[]} populate all CMSLSNUM found in LUMISUMMARY
        #runls={run:[[1,1],[2,5]],run:[[1,1],[2,5]]}
        #default value
        if not options.runls and not options.runnumber:
            print 'option -runls or -r is required for update'
            raise
        if not options.flag:
            print 'option -flag is required for update'
            raise
        if options.flag.upper() not in allowedFlags:
            print 'unrecognised flag ',options.flag
            raise
        if options.runnumber:
            runlsjson='{"'+str(options.runnumber)+'":[]}'
        elif options.runls:
            runlsjson=CommonUtil.tolegalJSON(options.runls)
        sparser=selectionParser.selectionParser(runlsjson)
        runsandls=sparser.runsandls()
        commentStr='NA'
        statusStr=options.flag
        for run,lslist in runsandls.items():
            if not result.has_key(run):
                result[run]=[]
            for ls in lslist:
                result[run].append([ls,statusStr,commentStr])
        insertupdateValidationData(session,result)
    if options.action=='dump':
        if options.runls or options.inputfile:
            if options.runls:
                runlsjson=CommonUtil.tolegalJSON(options.runls)
                sparser=selectionParser.selectionParser(runlsjson)
                runsandls=sparser.runsandls()
            if options.inputfile:
                p=inputFilesetParser.inputFilesetParser(options.inputfile)
                runsandls=p.runsandls()
            for runnum,lslist in runsandls.items():
                dataperrun=getValidationData(session,run=runnum,cmsls=lslist)
                if dataperrun.has_key(runnum):
                    result[runnum]=dataperrun[runnum]
                else:
                    result[runnum]=[]
        else:
            result=getValidationData(session,run=options.runnumber)
        runs=result.keys()
        runs.sort()
        if options.outputfile:
            r=csvReporter.csvReporter(options.outputfile)
            for run in runs:
                for perrundata in result[run]:
                    r.writeRow([str(run),str(perrundata[0]),perrundata[1],perrundata[2]])
        else:
            for run in runs:
                print '== ='
                if len(result[run])==0:
                    print str(run),'no validation data'
                    continue
                for lsdata in result[run]:
                    print str(run)+','+str(lsdata[0])+','+lsdata[1]+','+lsdata[2]
                
    del session
    del svc
if __name__ == '__main__':
    main()

    


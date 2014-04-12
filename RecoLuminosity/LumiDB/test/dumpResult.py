#!/usr/bin/env python
#
# dump all fills into files.
# allfills.txt all the existing fills.
# fill_num.txt all the runs in the fill
# dumpResult lumibyday --begin --end 
#
import os,os.path,sys
import coral,datetime
from RecoLuminosity.LumiDB import argparse,sessionManager,lumiTime,RegexValidator


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Dump Result",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('--begin',dest='begin',action='store',
                        default='01/01/12 00:00:00',
                        required=False,
                        type=RegexValidator.RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","wrong format"),
                        help='min run start time (mm/dd/yy hh:mm:ss)'
                        )                        
    parser.add_argument('--end',dest='end',action='store',
                        default='01/01/13 00:00:00',
                        required=False,
                        type=RegexValidator.RegexValidator("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$","wrong format"),
                        help='max run start time (mm/dd/yy hh:mm:ss)'
                        )
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='debug'
                        )
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
    qResult=coral.AttributeList()
    qResult.extend('timestr','string')
    qResult.extend('DELIVERED','float')
    session.transaction().start(True)
    lumiquery=session.nominalSchema().newQuery()
    lumiquery.addToTableList('HFLUMIRESULT')
    qCondition=coral.AttributeList()
    qCondition.extend('begintime','time stamp')
    qCondition.extend('endtime','time stamp')
    lumiquery.addToOutputList('TO_CHAR(TIME,\'MM/DD/YY HH24:MI:SS\')','timestr')
    lumiquery.addToOutputList('DELIVERED')
    lumiquery.defineOutput(qResult)
    lute=lumiTime.lumiTime()
    begtimeStr='01/01/12 00:00:00'
    reqtimemaxT=datetime.datetime.now()
    print options.begin,options.end
    if options.begin:
        begtimeStr=options.begin
    reqtimeminT=lute.StrToDatetime(options.begin,customfm='%m/%d/%y %H:%M:%S')
    if options.end:
        reqtimemaxT=lute.StrToDatetime(options.end,customfm='%m/%d/%y %H:%M:%S')
    qCondition['begintime'].setData(coral.TimeStamp(reqtimeminT.year,reqtimeminT.month,reqtimeminT.day,reqtimeminT.hour,reqtimeminT.minute,reqtimeminT.second,0))
    qCondition['endtime'].setData(coral.TimeStamp(reqtimemaxT.year,reqtimemaxT.month,reqtimemaxT.day,reqtimemaxT.hour,reqtimemaxT.minute,reqtimemaxT.second,0))
    lumiquery.setCondition('TIME>=:begintime AND TIME<=:endtime',qCondition)
    cursor=lumiquery.execute()
    result={}#{ordinalnumber:delivered}
    while cursor.next():
        timeStr=cursor.currentRow()['timestr'].data()
        runTime=lute.StrToDatetime(timeStr,customfm='%m/%d/%y %H:%M:%S')
        delivered=cursor.currentRow()['DELIVERED'].data()
        ordinalday=runTime.toordinal()
        if not result.has_key(ordinalday):
            result[ordinalday]=0.
        result[ordinalday]+=delivered
    session.transaction().commit()
    del lumiquery
    del session
    del svc
    alldays=result.keys()
    alldays.sort()
    for ordi in alldays:
        print datetime.datetime.fromordinal(ordi).date(),',',result[ordi]
    print '#total running days: ',len(alldays)
    print '#total delivered: ',sum(result.values())

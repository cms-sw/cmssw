#! /usr/bin/env python
import os,sys,commands,time,datetime
import coral
from RecoLuminosity.LumiDB import argparse,lumiQueryAPI,lumiTime,csvReporter
###
#Script to fill the lumi monitoring site. This is not a generic tool
###
def createRunList(c,p='.',o='.',dryrun=False):
    '''
     input:
      c connect string
      p authenticaion path
    '''
    msg=coral.MessageStream('')
    msg.setMsgVerbosity(coral.message_Level_Error)
    os.environ['CORAL_AUTH_PATH']='/afs/cern.ch/cms/DB/lumi'
    svc = coral.ConnectionService()
    connectstr=c
    session=svc.connect(connectstr,accessMode=coral.access_ReadOnly)
    session.typeConverter().setCppTypeForSqlType("unsigned int","NUMBER(10)")
    session.typeConverter().setCppTypeForSqlType("unsigned long long","NUMBER(20)")
    session.transaction().start(True)
    schema=session.nominalSchema()
    allruns=lumiQueryAPI.allruns(schema,requireLumisummary=True,requireTrg=True,requireHlt=True)
    session.transaction().commit()  
    del session
    del svc
    allruns.sort()
    if not dryrun:
        report=csvReporter.csvReporter(os.path.join(o,'runlist.txt'))
        for run in allruns:
            report.writeRow([run])
    else:
        print allruns
def totalLumivstime(c,p='.',i='',o='.',begTime="03/30/10 10:00:00.00",endTime=None,selectionfile=None,beamstatus=None,beamenergy=None,beamenergyfluctuation=None,dryrun=False):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file name
      o output path
    '''
    plotoutname='totallumivstime.png'
    textoutname='totallumivstime.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin','"'+begTime+'"','-batch',os.path.join(o,plotoutname),'time']
    if endTime:
        elements.append('-end ')
        elements.append('"'+endTime+'"')
    command=' '.join(elements)
    print command
    if not dryrun:
        statusAndOutput=commands.getstatusoutput(command)
        
def totalLumivstimeLastweek(c,p='.',i='',o='.',selectionfile=None,beamstatus=None,beamenergy=None,beamenergyfluctuation=None,dryrun=False):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file name
      o output path
      ##fix me: year boundary is not considered!
    '''
    plotoutname='totallumivstime-weekly.png'
    textoutname='totallumivstime-weekly.csv'
    nowTime=datetime.datetime.now()
    lastMondayStr=' '.join([str(nowTime.isocalendar()[0]),str(nowTime.isocalendar()[1]-1),str(nowTime.isocalendar()[2])])
    
    lastweekMonday=datetime.datetime(*time.strptime(lastMondayStr,'%Y %W %w')[0:5])
    lastweekEndSunday=lastweekMonday+datetime.timedelta(days=7,hours=24)
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin','"'+lastweekMonday+'"','-end','"'+lastweekEndSunday+'"','-batch',os.path.join(o,plotoutname),'time']
    command=' '.join(elements)
    print command
    if not dryrun:
        statusAndOutput=commands.getstatusoutput(command)

def lumiPerDay(c,p='.',i='',o='',begTime="03/30/10 10:00:00.00",endTime=None,selectionfile=None,beamstatus=None,beamenergy=None,beamenergyfluctuation=None,dryrun=False):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file
      o outputpath
    '''
    plotoutname='lumiperday.png'
    textoutname='lumiperday.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin','"'+begTime+'"','-batch',os.path.join(o,plotoutname),'perday']
    if endTime:
        elements.append('-end')
        elements.append('"'+endTime+'"')
    command=' '.join(elements)
    print command
    if not dryrun:
        statusAndOutput=commands.getstatusoutput(command)

def totalLumivsRun(c,p='.',i='',o='',begRun="132440",endRun=None,selectionfile=None,beamenergy=None,beamenergyfluctuation=None,dryrun=False):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file
      o outputpath 
    '''
    plotoutname='totallumivsrun.png'
    textoutname='totallumivsrun.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin',begRun,'-batch',os.path.join(o,plotoutname),'run']
    if endRun:
        elements.append('-end')
        elements.append(endRun)
    if textoutname:
        elements.append('-o')
        elements.append(textoutname)
    command=' '.join(elements)
    print command
    if not dryrun:
        statusAndOutput=commands.getstatusoutput(command)

def totalLumivsFill(c,p='.',i='',o='',begFill="1005",endFill=None,selectionfile=None,beamenergy=None,beamenergyfluctuation=None,dryrun=False):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file
      o output path
    '''
    plotoutname='totallumivsfill.png'
    textoutname='totallumivsfill.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin',begFill,'-batch',os.path.join(o,plotoutname),'fill']
    if endFill:
        elements.append('-end')
        elements.append(endFill)
    command=' '.join(elements)
    print command
    if not dryrun:
        statusAndOutput=commands.getstatusoutput(command)

def instLumiForRuns(c,runnumbers,p='.',o='.',dryrun=False):
    '''
    draw instlumperrun plot for the given runs
    input:
      c connect string
      runnumbers []
      p authenticaion path
      o output path
    '''
    plotoutname='rollinginstlumi_'
    textoutname='rollinginstlumi_'
    for idx,run in enumerate(runnumbers):
        batch=os.path.join(o,plotoutname+str(idx+1)+'.png')
        elements=['lumiInstPlot.py','-c',c,'-P',p,'-begin',str(run),'-batch',batch,'run']
        command=' '.join(elements)
        print command
        if not dryrun:
            statusAndOutput=commands.getstatusoutput(command)

def instPeakPerday(c,p='.',o='.',begTime="03/30/10 10:00:00.00",endTime=None,dryrun=False):
    '''
    input:
      c connect string
      p authenticaion path
      o outout text/csv file
    '''
    plotoutname='lumipeak.png'
    textoutname='lumipeak.csv'
    elements=['lumiInstPlot.py','-c',c,'-P',p,'-begin','"'+begTime+'"','-batch',os.path.join(o,plotoutname),'peakperday']
    if endTime:
        elements.append('-end')
        elements.append('"'+endTime+'"')
    command=' '.join(elements)
    print command
    if not dryrun:
        statusAndOutput=commands.getstatusoutput(command)

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Produce lumi plots")
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',required=False,help='auth path. Optional. Default to .')
    parser.add_argument('-L',dest='logpath',action='store',required=False,help='log path. Optional. Default to .')
    parser.add_argument('-i',dest='ifile',action='store',required=False,help='input selection file. Optional.')
    parser.add_argument('-o',dest='opath',action='store',required=False,help='output file path. Optional')
    parser.add_argument('--dryrun',dest='dryrun',action='store_true',help='dryrun mode')
    parser.add_argument('action',choices=['instperrun','instpeakvstime','totalvstime','totallumilastweek','totalvsfill','totalvsrun','perday','createrunlist'],help='command actions')
    args=parser.parse_args()

    authpath='.'
    logpath='.'
    connectstr=args.connect
    isDryrun=False
    if args.dryrun:
        isDryrun=True
    if args.authpath:
        authpath=args.authpath
    if args.logpath:
        lumilogpath=args.logpath
    if args.ifile:
        ifile=args.ifile
    if args.opath:
        opath=args.opath
    if args.action == 'createrunlist':
        createRunList(connectstr,authpath,o=opath,dryrun=isDryrun)
    if args.action == 'instperrun':
        if not args.ifile:
            print 'option -i is required for action instperrun'
            return 2
        f=open(args.ifile,'r')
        runs=[]
        for run in f:
            runs.append(int(run))
        last2runs=[runs[-2],runs[-1]]
        instLumiForRuns(connectstr,last2runs,p=authpath,o=opath,dryrun=isDryrun)
    if args.action == 'instpeakvstime':
        instPeakPerday(connectstr,p=authpath,o=opath,dryrun=isDryrun)
    if args.action == 'totalvstime':
        totalLumivstime(connectstr,p=authpath,o=opath,dryrun=isDryrun)
    if args.action == 'totallumilastweek':
        totalLumivstimeLastweek(connectstr,p=authpath,o=opath,dryrun=isDryrun)
    if args.action == 'totalvsfill':
        totalLumivsFill(connectstr,p=authpath,o=opath,dryrun=isDryrun)
    if args.action == 'totalvsrun':
        totalLumivsRun(connectstr,p=authpath,o=opath,dryrun=isDryrun)
    if args.action == 'perday':       
        lumiPerDay(connectstr,p=authpath,o=opath,dryrun=isDryrun)
if __name__=='__main__':
    main()

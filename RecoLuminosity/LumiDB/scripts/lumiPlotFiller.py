#! /usr/bin/env python
import os,os.path,commands,sys
import coral
from RecoLuminosity.LumiDB import argparse,lumiQueryAPI,lumiTime,csvReporter
###
#Script to fill the lumi monitoring site. This is not a generic tool
###
def createRunList(c,p='.',o='.'):
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
    report=csvReporter.csvReporter(os.path.join(o,'runlist.txt'))
    for run in allruns:
        report.writeRow([run])
def totalLumivstime(c,p='.',i='',o='.',begTime="03/30/10 10:00:00.00",endTime=None,selectionfile=None,beamstatus=None,beamenergy=None,beamenergyfluctuation=None):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file name
      o output path
    '''
    plotoutname='totallumivstime.png'
    textoutname='totallumivstime.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin',begTime,'-batch',os.path.join(o,plotoutname),'time']
    if endTime:
        elements.append('-end '+endTime)
    command=' '.join(elements)
    #statusAndOutput=commands.getstatusoutput(command)
    print command
    #print 'output ',statusAndOutput[1]
    
def totalLumivstimeLastweek(c,p='.',i='',o='.',selectionfile=None,beamstatus=None,beamenergy=None,beamenergyfluctuation=None):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file name
      o output path
    '''
    plotoutname='totallumivstime-weekly.png'
    textoutname='totallumivstime-weekly.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin',begTime,'-batch',os.path.join(o,plotoutname),'time']
    if endTime:
        elements.append('-end '+endTime)
    command=' '.join(elements)
    #statusAndOutput=commands.getstatusoutput(command)
    print command
    #print 'output ',statusAndOutput[1]

def lumiPerDay(c,p='.',i='',o='',begTime="03/30/10 10:00:00.00",endTime=None,selectionfile=None,beamstatus=None,beamenergy=None,beamenergyfluctuation=None):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file
      o outputpath
    '''
    plotoutname='lumiperday.png'
    textoutname='lumiperday.csv'
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin',begTime,'-batch',os.path.join(o,plotoutname),'perday']
    if endTime:
        elements.append('-end '+endTime)
    command=' '.join(elements)
    #statusAndOutput=commands.getstatusoutput(command)
    print command
    #print 'output ',statusAndOutput[1]
    
def totalLumivsRun(c,p='.',i='',o='',begRun="132440",endRun=None,selectionfile=None,beamenergy=None,beamenergyfluctuation=None):
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
        elements.append('-end '+endRun)
    command=' '.join(elements)
    print command
    #statusAndOutput=commands.getstatusoutput(command)
    #print 'output ',statusAndOutput[1]

def totalLumivsFill(c,p='.',i='',o='',begFill="1005",endFill=None,selectionfile=None,beamenergy=None,beamenergyfluctuation=None):
    '''
    input:
      c connect string
      p authenticaion path
      i input selection file
      o output path
    '''
    plotoutname='totallumivsfill.png'
    textoutname='totallumivsfill.csv'
    if len(batch)==0:
        batch=defaultbatch
    elements=['lumiSumPlot.py','-c',c,'-P',p,'-begin',begFill,'-batch',os.path.join(o,plotoutname),'fill']
    if endFill:
        elements.append('-end '+endFill)
    command=' '.join(elements)
    print command
    #statusAndOutput=commands.getstatusoutput(command)
    #print 'output ',statusAndOutput[1]

def instLumiForRuns(c,runnumbers,p='.',o='.'):
    '''
    draw instlumperrun plot for the given runs
    input:
      c connect string
      p authenticaion path
      o output path
    '''
    plotoutname='rollinginstlumi'
    textoutname='rollinginstlumi'
    for idx,run in enumerate(runnumbers):
        batch=os.path.join(o,plotoutname+str(idx+1)+'.png')
        elements=['lumiInstPlot.py','-c',c,'-P',p,'-begin',str(run),'-batch',batch,'run']
        command=' '.join(elements)
        print command
        #statusAndOutput=commands.getstatusoutput(command)
        #print 'output ',statusAndOutput[1]

def instPeakPerday(c,p='.',o='.',begTime="03/30/10 10:00:00.00",endTime=None):
    '''
    input:
      c connect string
      p authenticaion path
      o outout text/csv file
    '''
    plotoutname='lumipeak.png'
    textoutname='lumipeak.csv'
    elements=['lumiInstPlot.py','-c',c,'-P',p,'-begin',begTime,'-batch',os.path.join(o,plotoutname),'peakperday']
    command=' '.join(elements)
    print command
    #statusAndOutput=commands.getstatusoutput(command)
    #print 'output ',statusAndOutput[1]

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Produce plots for lumi monitor site")
    parser.add_argument('action',choices=['createrunlist','instperrun','instpeakvstime','totalvstime','totalvsfill','totalvsrun','perday'],help='command actions')  
    parser.add_argument('-c',dest='connect',action='store',required=True,help='connect string to lumiDB')
    parser.add_argument('-P',dest='authpath',action='store',required=False,help='auth path. Optional. Default to .')
    parser.add_argument('-L',dest='logpath',action='store',required=False,help='log path. Optional. Default to .')
    parser.add_argument('-i',dest='ifile',action='store',required=False,help='input selection file. Optional.')
    parser.add_argument('-o',dest='opath',action='store',required=False,help='output file path. Optional')
    args=parser.parse_args()

    authpath='.'
    logpath='.'
    connectstr=args.connect
    
    if args.authpath:
        authpath=args.authpath
    if args.logpath:
        lumilogpath=args.logpath
    if args.ifile:
        ifile=args.ifile
    if args.opath:
        opath=args.opath
    if args.action == 'createrunlist':
        createRunList(connectstr,authpath)
    if args.action == 'instperrun':
        if not args.opath:
            print 'option -i is required for action instperrun'
            return 2
    if args.action == 'instpeakvstime':
        pass
    if args.action == 'totalvstime':
        pass
    if args.action == 'totalvstimeLastweek':
        pass
    if args.action == 'totalvsfill':
        pass
    if args.action == 'totalvsrun':
        pass
    if args.action == 'perday':
        pass
    
if __name__=='__main__':
    main()

#!/usr/bin/env python
VERSION='1.00'
import os,sys,ctypes
import coral
from RecoLuminosity.LumiDB import argparse,nameDealer,selectionParser

class constants(object):
    def __init__(self):
        self.LUMIUNIT='E27cm^-2'
        self.NORM=16400
        self.LUMIVERSION='0001'
        self.BEAMMODE='stable' #possible choices stable,quiet,either
    
def deliveredLumiForRun(dbsession,c,runnum):
    #
    #select sum(INSTLUMI) from lumisummary where runnum=124025 and lumiversion='0001';
    #apply norm factor on the query result 
    #unit E27cm^-2 
    #
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
        while cursor.next():
            delivered=cursor.currentRow()['totallumi'].data()*c.NORM
        del query
        dbsession.transaction().commit()
        print "Delivered Luminosity for Run "+str(runnum)+" (beam "+c.BEAMMODE+"): "+str(delivered)+c.LUMIUNIT
    except Exception,e:
        print str(e)
        dbsession.transaction().rollback()
        del dbsession

    
def deliveredLumiForRange(dbsession,c,inputfile):
    #
    #in this case,only take run numbers from theinput file
    #
    #print 'inside deliveredLumi : norm : ',c.NORM,' : inputfile : ',inputfile
    
    f=open(inputfile,'r')
    content=f.read()
    s=selectionParser.selectionParser(content)
    for run in s.runs():
        deliveredLumiForRun(dbsession,c,run)
    
def recordedLumiForRun(dbsession,c,runnum):
    print 'inside recordedLumi : run : ',runnum,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION

def recordedLumiForRange(dbsession,c,inputfile):
    print 'inside recordedLumi : inputfile : ',inputfile,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION
    
def effectiveLumiForRun(dbsession,c,runnum,hltpath=''):
    print 'inside effectiveLumi : runnum : ',runnum,' : hltpath : ',hltpath,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION

def effectiveLumiForRange(dbsession,c,inputfile,hltpath=''):
    print 'inside effectiveLumi : inputfile : ',inputfile,' : hltpath : ',hltpath,' : norm : ',c.NORM,' : LUMIVERSION : ',c.LUMIVERSION


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
    parser.add_argument('-hltpath',dest='hltpath',action='store',help='specific hltpath to calculate the effective luminosity, default to All')
    parser.add_argument('action',choices=['delivered','recorded','effective'],help='lumi calculation types')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose')
    # parse arguments
    args=parser.parse_args()
    connectstring=args.connect
    runnumber=0
    svc = coral.ConnectionService()
    hpath=''
    ifilename=''
    beammode='stable'
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
    if args.verbose :
        print 'verbose mode'
    del session
    del svc
if __name__=='__main__':
    main()
    

#!/usr/bin/env python
VERSION='1.02'
import os,sys,re
import coral
from RecoLuminosity.LumiDB import argparse,lumiQueryAPI

def main():
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description="Dump Prescale info for selected hltpath and trg path",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c',dest='connect',action='store',help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file')
    parser.add_argument('-r',dest='runnumber',action='store',help='run number')
    parser.add_argument('-hltpath',dest='hltpath',action='store',required=True,help='hltpath')
    parser.add_argument('-trgbits',dest='trgbits',action='store',help='trgbits',default='all')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    args=parser.parse_args()
    runnumber=args.runnumber
    if args.authpath and len(args.authpath)!=0:
        os.environ['CORAL_AUTH_PATH']=args.authpath
    parameters = lumiQueryAPI.ParametersObject()
    session,svc =  lumiQueryAPI.setupSession (args.connect or \
                                              'frontier://LumiCalc/CMS_LUMI_PROD',
                                               args.siteconfpath,parameters,args.debug)
    session.transaction().start(True)
    schema=session.nominalSchema()
    runlist=[]
    if args.debug:
        msg=coral.MessageStream('')
        msg.setMsgVerbosity(coral.message_Level_Debug)
    if args.runnumber:
        runlist.append(int(args.runnumber))
    else:
        runlist=lumiQueryAPI.allruns(schema,True,True,True,True)
    runlist.sort()
    bitlist=[]
    hltpathStr=re.sub('\s','',args.hltpath)
    #print bitlistStr
    if args.trgbits and args.trgbits!='all':
        bitlistStr=args.trgbits
        bitlistStr=bitlistStr.strip()
        bitlistStr=re.sub('\s','',bitlistStr)
        bitlist=bitlistStr.split(',')
    result={}#{run:{hltpath:prescle},{trgname:prescale}}
    trgdict={}#{run:[(trgname,trgprescale),()]}
    hltdict={}#{run:{hltname:hltprescale}}
    for runnum in runlist:
        q=schema.newQuery()
        hltdump=lumiQueryAPI.hltBypathByrun(q,runnum,hltpathStr)
        del q
        if len(hltdump)>0:
            hltdict[runnum]=hltdump[1][-1]
        else:
            print 'run ',runnum,' hltpath ','"'+hltpathStr+'"','not found'
            continue
        if not args.trgbits or args.trgbits=='all':
            q=schema.newQuery()
            l1seeds = lumiQueryAPI.hlttrgMappingByrun(q,runnum)
            del q
            if len(l1seeds)==0 or not l1seeds.has_key(hltpathStr):
                print 'hlt path',hltpathStr,'has no l1 seed'
                continue
            l1seed=l1seeds[hltpathStr]
            rmQuotes=l1seed.replace('\"','')
            rmOR=rmQuotes.replace(' OR','')
            rmNOT=rmOR.replace(' NOT','')
            rmAND=rmOR.replace(' AND','')
            bitlist=rmAND.split(' ')
        if not trgdict.has_key(runnum):
            trgdict[runnum]=[]
        for bitname in bitlist:
            q=schema.newQuery()
            trgbitdump=lumiQueryAPI.trgBybitnameByrun(q,runnum,bitname)
            del q
            if len(trgbitdump)>0:
                trgprescale=trgbitdump[1][-1]
                trgdict[runnum].append((bitname,trgprescale))
            else:
                print 'run ',runnum,' bit ','"'+bitname+'"',' not found'
                continue
    session.transaction().commit()
    del session
    del svc
    if len(hltdict)<1:
        print 'no result found for',hltpathStr
        sys.exit(-1)
    runs=hltdict.keys()
    runs.sort()
    for r in runs:
        if not hltdict.has_key(r): continue
        hltprescale=hltdict[r]
        toprint=[str(r),hltpathStr,str(hltprescale)]
        if trgdict.has_key(r):
            trgbitsinfo=trgdict[r]
            for trgbitinfo in trgbitsinfo:
                trgname=trgbitinfo[0]
                toprint.append(trgname)
                trgprescale=trgbitinfo[1]
                toprint.append(str(trgprescale))
        print ' '.join(toprint)
    #print trgdict
    #print hltdict
if __name__=='__main__':
    main()
    

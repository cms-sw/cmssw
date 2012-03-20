#!/usr/bin/env python
#
# dump all fills into files.
# allfills.txt all the existing fills.
# fill_num.txt all the runs in the fill
# dumpFill -o outputdir
# dumpFill -f fillnum generate runlist for the given fill
#
import os,os.path,sys
import coral
from RecoLuminosity.LumiDB import argparse,sessionManager,lumiCalcAPI
MINFILL=1800
MAXFILL=9999
allfillname='allfills.txt'

def tofiles(allfills,runsperfill,runtimes,outdir):
    f=open(os.path.join(outdir,allfillname),'w')
    for fill in allfills:
        print >>f,'%d'%(fill)
    f.close()
    for fill,runs in runsperfill.items():
        filename='fill_'+str(fill)+'.txt'
        if len(runs)!=0:
            f=open(os.path.join(outdir,filename),'w')
            for run in runs:
                print >>f,'%d,%s'%(run,runtimes[run])
            f.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Dump Fill",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiCalc/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-o',dest='outputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('--amodetag',dest='amodetag',action='store',required=False,help='amodetag',default='PROTPHYS')
    parser.add_argument('-f','--fill',
                        dest='fillnum',
                        action='store',
                        required=False,
                        help='specific fill',default=None)
    parser.add_argument('--minfill',dest='minfill',
                        type=int,
                        action='store',
                        required=False,
                        default=MINFILL,
                        help='min fill')
    parser.add_argument('--maxfill',dest='maxfill',
                        type=int,
                        action='store',
                        required=False,
                        default=MAXFILL,
                        help='maximum fillnumber '
                        )
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
    session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        
    session.transaction().start(True)
    allfills=lumiCalcAPI.fillInRange(session.nominalSchema(),fillmin=options.minfill,fillmax=options.maxfill,amodetag=options.amodetag)
    if len(allfills)==0:
        print 'no qualified fills found, do nothing... '
        exit(-1)
    allfills.sort()
    runsperfill={}
    runtimes={}
    irunlsdict={}
    if options.fillnum:
        if int(options.fillnum) in allfills:
            runsperfill=lumiCalcAPI.fillrunMap(session.nominalSchema(),fillnum=int(options.fillnum))
            allruns=runsperfill[ int(options.fillnum) ]
            allls=[None]*len(allruns)
            irunlsdict=dict(zip(allruns,allls))
            runresults=lumiCalcAPI.runsummary(session.nominalSchema(),irunlsdict)
            for r in runresults:
                runtimes[r[0]]=r[7]
    else:
        for fill in allfills:
            runtimes={}
            runsperfill=lumiCalcAPI.fillrunMap(session.nominalSchema(),fillnum=fill)
            runs=runsperfill.values()#list of lists
            allruns=[item for sublist in runs for item in sublist]
            allls=[None]*len(allruns)
            irunlsdict=dict(zip(allruns,allls))
            runresults=lumiCalcAPI.runsummary(session.nominalSchema(),irunlsdict)
            for r in runresults:
                runtimes[r[0]]=r[7]
            tofiles(allfills,runsperfill,runtimes,options.outputdir)
    session.transaction().commit()



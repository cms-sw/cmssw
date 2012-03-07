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
from RecoLuminosity.LumiDB import argparse,lumiQueryAPI
#
#lumiQueryAPI.runsByFillrange 
#lumiQueryAPI.allfills
#
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
    parser.add_argument('-f',dest='fillnum',action='store',required=False,help='specific full',default=None)
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    parameters = lumiQueryAPI.ParametersObject()
    session,svc =  lumiQueryAPI.setupSession (options.connect or \
                                              'frontier://LumiCalc/CMS_LUMI_PROD',
                                               options.siteconfpath,parameters,options.debug)
    session.transaction().start(True)
    q=session.nominalSchema().newQuery()
    allfills=lumiQueryAPI.allfills(q)
    del q
    allfills.sort()
    runsperfill={}
    runtimes={}
    if options.fillnum:
        if int(options.fillnum) in allfills:
            q=session.nominalSchema().newQuery()
            runsperfill=lumiQueryAPI.runsByfillrange(q,int(options.fillnum),int(options.fillnum))
            del q
            for run in runsperfill[int(options.fillnum)]:
                q=session.nominalSchema().newQuery()
                runtimes[run]=lumiQueryAPI.runsummaryByrun(q,run)[3]
                del q
    else:
        q=session.nominalSchema().newQuery()
        runsperfill=lumiQueryAPI.runsByfillrange(q,min(allfills),max(allfills))
        del q
        runs=runsperfill.values()#list of lists
        allruns=[item for sublist in runs for item in sublist]
        allruns.sort()
        for run in allruns:
            q=session.nominalSchema().newQuery()
            runtimes[run]=lumiQueryAPI.runsummaryByrun(q,run)[3]
            del q
    session.transaction().commit()
    #print runsperfill
    tofiles(allfills,runsperfill,runtimes,options.outputdir)

#!/usr/bin/env python
#
import os,os.path,sys,math,array,datetime,time,re
import coral

from RecoLuminosity.LumiDB import argparse,lumiTime,CommonUtil,lumiCalcAPI,lumiCorrections
MINFILL=1800
allfillname='allfills.txt'

def lastcompleteFill(infile):
    lastfill=None
    hlinepat=r'(LASTCOMPLETEFILL )([0-9]{4})'
    h=re.compile(hlinepat)
    dqmfile=open(infile,'r')
    for line in dqmfile:
        result=h.match(line)
        if result:
            lastfill=result.group(2)
            break
    return int(lastfill)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Dump Fill",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='oracle://cms_orcoff_prod/cms_lumi_prod')
    parser.add_argument('-P',dest='authpath',action='store',required=True,help='authentication.xml dir')
    parser.add_argument('-i',dest='inputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('-o',dest='outputdir',action='store',required=False,help='output dir',default='.')
    parser.add_argument('-f',dest='fillnum',action='store',required=False,help='specific fill',default=None)
    parser.add_argument('-norm',dest='norm',action='store',required=False,help='norm',default=None)
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    parser.add_argument('--with-correction',dest='withFineCorrection',action='store_true',required=False,help='with fine correction',default=None)
    options=parser.parse_args()
    
    allfillsFromFile=[]
    fillstoprocess=[]
    minfillnum=1700
    maxfillnum=None
    summaryfilename='_summary_CMS.txt'
    dbname=options.connect
    authdir=options.authpath
    if options.fillnum is not None: #if process a specific single fill
        fillstoprocess.append(int(options.fillnum))
    else: #if process fills automatically
        session.transaction().start(True)
        schema=session.nominalSchema()
        schema.transaction().start(True)
        allfillsFromDB=lumiCalcAPI.fillInRange(schema,minfillnum,maxfillnum)
        processedfills=listfilldir(options.outputdir)
        lastcompletedFill=lastcompleteFill(os.path.join(options.inputdir,'runtofill_dqm.txt'))
        print 'last complete fill : ',lastcompletedFill
        print 'processedfills in '+options.outputdir+' ',processedfills
        for pf in processedfills:
            if pf>lastcompletedFill:
                print '\tremove unfinished fill from processed list ',pf
                processedfills.remove(pf)
        print 'final processed fills : ',sorted(processedfills)
        for fill in allfillsFromDB:
            if fill not in processedfills :
                if fill<=lastcompletedFill:
                    #print 'fill less than last complet fill ',fill
                    if fill>minfillnum:
                        fillstoprocess.append(fill)
                else:
                    print 'ongoing fill...',fill
        schema.transaction().start(True)
    print 'fills to process : ',fillstoprocess
    if len(fillstoprocess)==0:
        print 'no fill to process, exit '
        exit(0)

    lslength=23.357
    import commands,os,RecoLuminosity.LumiDB.lumiTime,datetime,time
    for fillnum in fillstoprocess:
        clineElements=['lumiCalc2.py','lumibyls','-b stable','-c',dbname,'-P',authdir,'-f',str(fillnum),'-o','tmp.out']
        (exestat,resultStr)=commands.getstatusoutput(' '.join(clineElements))
        if exestat!=0:
            print 'lumiCalc2.py execution error ',resultStr
            exit(exestat)
        f=open('tmp.out','r')
        lcount=0
        lines=f.readlines()
        stablefillmap={}#{run:([ts],[lumi])}
        for line in lines:
            lcount=lcount+1
            if lcount==1:
                continue
            #print line.strip()
            line=line.strip()
            lineList=line.split(',')
            runnum=int(lineList[0])
            if not stablefillmap.has_key(runnum):
                stablefillmap[runnum]=([],[])
            timestamp=lineList[2]
            bstatus=lineList[3]
            t=lumiTime.lumiTime()
            pydate=t.StrToDatetime(timestamp,'%m/%d/%y %H:%M:%S')
            os.environ['TZ']='UTC'
            time.tzset()
            unixts=int(time.mktime(pydate.timetuple()))
            deliveredintl=float(lineList[5])
            if bstatus=='STABLE BEAMS':
                stablefillmap[runnum][0].append(unixts)
                stablefillmap[runnum][1].append(deliveredintl)
        summaryfilename=os.path.join(options.outputdir,str(fillnum)+summaryfilename)
        print summaryfilename
        ofile=open(summaryfilename,'w')
        for r in sorted(stablefillmap):
            rundata=stablefillmap[r]
            print >>ofile,'%d\t%d\t%.6e\t%.6e'%(min(rundata[0]),max(rundata[0]), max(rundata[1])/lslength,sum(rundata[1]))
        ofile.close()
        os.remove('tmp.out')
        f.close()
        

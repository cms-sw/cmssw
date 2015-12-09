#!/usr/bin/env python
#
#########################################################################
# Command to produce fill summary lumi files using lumiCalc2.py lumibyls#
# output                                                                # 
#                                                                       #
# Author:      Zhen Xie                                                 #
#########################################################################
#
import os,os.path,sys,math,array,datetime,time,re

from RecoLuminosity.LumiDB import argparse,lumiTime,lumiCalcAPI,sessionManager,lumiParameters
MINFILL=1800
MAXFILL=9999
allfillname='allfills.txt'
runtofilldqmfile='runtofill_dqm.txt'

def listfilldir(indir):
    fillnamepat=r'^[0-9]{4}$'
    p=re.compile(fillnamepat)
    processedfills=[]
    dirList=os.listdir(indir)
    for fname in dirList:
        if p.match(fname) and os.path.isdir(os.path.join(indir,fname)):#found fill dir
            allfs=os.listdir(os.path.join(indir,fname))
            for myfile in allfs:
                sumfilenamepat=r'^[0-9]{4}_bxsum_CMS.txt$'
                s=re.compile(sumfilenamepat)
                if s.match(myfile):
                    #only if fill_summary_CMS.txt file exists
                    processedfills.append(int(fname))
    return processedfills

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

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Dump Fill",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parse arguments
    parser.add_argument('-c',
                        dest='connect',
                        action='store',
                        required=False,
                        help='connect string to lumiDB,optional',
                        default='oracle://cms_orcon_adg/cms_lumi_prod')
    parser.add_argument('-P',
                        dest='authpath',
                        action='store',
                        required=True,
                        help='authentication.xml dir')
    parser.add_argument('-i',
                        dest='inputdir',
                        action='store',
                        required=False,
                        help='input dir to runtofill_dqm.txt',
                        default='.')
    parser.add_argument('-o',
                        dest='outputdir',
                        action='store',
                        required=False,
                        help='output dir',
                        default='.')
    parser.add_argument('-f','--fill',
                        dest='fillnum',
                        action='store',
                        required=False,
                        help='specific fill',
                        default=None)
    parser.add_argument('--datatag',
                        dest='datatag',
                        action='store',
                        required=False,
                        help='datatag',
                        default=None)
    parser.add_argument('--normtag',
                        dest='normtag',
                        action='store',
                        required=False,
                        help='normtag',
                        default=None)
    parser.add_argument('--minfill',
                        dest='minfill',
                        action='store',
                        required=False,
                        help='minimal fillnumber',
                        default=None)
    parser.add_argument('--maxfill',
                        dest='maxfill',
                        action='store',
                        required=False,
                        help='maximum fillnumber ',
                        default=MAXFILL)
    parser.add_argument('--amodetag',
                        dest='amodetag',
                        action='store',
                        required=False,
                        help='specific accelerator mode choices [PROTOPHYS,IONPHYS,PAPHYS] (optional)')
    parser.add_argument('--beamenergy',
                        dest='beamenergy',
                        action='store',
                        type=float,
                        default=None,
                        help='nominal beam energy in GeV')
    parser.add_argument('--beamfluctuation',
                        dest='beamfluctuation',
                        type=float,action='store',
                        default=0.2,
                        required=False,
                        help='fluctuation in fraction allowed to nominal beam energy, default 0.2, to be used together with -beamenergy  (optional)')
    parser.add_argument('--debug',
                        dest='debug',
                        action='store_true',
                        help='debug')
    parser.add_argument('--without-stablebeam',
                        dest='withoutStablebeam',
                        action='store_true',
                        required=False,
                        help='without requirement on stable beams')
    parser.add_argument('--without-correction',
                        dest='withoutFineCorrection',
                        action='store_true',
                        required=False,
                        help='without fine correction')
    options=parser.parse_args()
    if options.minfill:
        MINFILL=int(options.minfill)
    fillstoprocess=[]
    maxfillnum=options.maxfill
    summaryfilenameTMP='_summary_CMS.txt'
    dbname=options.connect
    authdir=options.authpath
    if options.fillnum is not None: #if process a specific single fill
        fillstoprocess.append(int(options.fillnum))
    else: #if process fills automatically
        svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)
        session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        session.transaction().start(True)
        schema=session.nominalSchema()
        allfillsFromDB=lumiCalcAPI.fillInRange(schema,fillmin=MINFILL,fillmax=maxfillnum,amodetag=options.amodetag)
        session.transaction().commit()
        processedfills=listfilldir(options.outputdir)
        lastcompletedFill=lastcompleteFill(os.path.join(options.inputdir,'runtofill_dqm.txt'))
        for pf in processedfills:
            if pf>lastcompletedFill:
                print '\tremove unfinished fill from processed list ',pf
                processedfills.remove(pf)
        for fill in allfillsFromDB:
            if fill not in processedfills :
                if int(fill)<=lastcompletedFill:
                    if int(fill)>MINFILL:
                        fillstoprocess.append(fill)
                else:
                    print 'ongoing fill...',fill
    print 'fills to process : ',fillstoprocess
    if len(fillstoprocess)==0:
        print 'no fill to process, exit '
        exit(0)
    lumip=lumiParameters.ParametersObject()
    lslength=lumip.lslengthsec()
    import commands,os,RecoLuminosity.LumiDB.lumiTime,datetime,time
    for fillnum in fillstoprocess:
        clineElements=['lumiCalc2.py','lumibyls','-c',dbname,'-P',authdir,'-f',str(fillnum),'-o','tmp.out','--without-checkforupdate','--nowarning']
        if not options.withoutStablebeam:
            clineElements.append('-b stable')
        if options.withoutFineCorrection:
            clineElements.append('--without-correction')
        if options.datatag:
            clineElements.append('--datatag '+options.datatag)
        if options.normtag:
            clineElements.append('--normtag '+options.normtag)
        if options.beamenergy:
            clineElements.append('--beamenergy '+str(options.beamenergy))
        if options.beamfluctuation:
            clineElements.append('--beamfluctuation '+str(options.beamfluctuation))
        
        finalcmmd=' '.join(clineElements)
        print 'cmmd executed:',finalcmmd
        (exestat,resultStr)=commands.getstatusoutput(finalcmmd)
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
            runnum=int(lineList[0].split(':')[0])
            if runnum not in stablefillmap:
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
        filloutdir=os.path.join(options.outputdir,str(fillnum))
        if not os.path.exists(filloutdir):
            os.mkdir(filloutdir)
        #print 'options.outputdir ',options.outputdir
        #print 'fillnum ',fillnum
        #print 'str(fillnum)+summaryfilename ',str(fillnum)+summaryfilenameTMP
        summaryfilename=os.path.join(options.outputdir,str(fillnum),str(fillnum)+summaryfilenameTMP)
        #print 'summaryfilename ',summaryfilename
        ofile=open(summaryfilename,'w')
        if len(stablefillmap)==0:
            print >>ofile,'%s'%('#no stable beams')
        else:
            for r in sorted(stablefillmap):
                rundata=stablefillmap[r]
                print >>ofile,'%d\t%d\t%.6e\t%.6e'%(min(rundata[0]),max(rundata[0]), max(rundata[1])/lslength,sum(rundata[1]))
        ofile.close()
        os.remove('tmp.out')
        f.close()
        

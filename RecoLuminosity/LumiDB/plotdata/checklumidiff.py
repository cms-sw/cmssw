import sys,os,os.path,glob,csv,math,datetime
def parseplotcache(filelist,fillmin,fillmax):
    result={}#{fill:{run:delivered}}
    for f in filelist:
        fileobj=open(f,'rb')
        plotreader=csv.reader(fileobj,delimiter=',')
        idx=0
        for row in plotreader:
            if idx!=0:
                [run,fill]=row[0].split(':')
                [lumils,cmsls]=row[1].split(':')
                if int(fill) not in range(fillmin,fillmax+1):
                    continue
                delivered=float(row[5])
                if not result.has_key(int(fill)):
                    result[int(fill)]={}
                if result[int(fill)].has_key(int(run)):
                    result[int(fill)][int(run)]+=delivered
                else:
                    result[int(fill)][int(run)]=delivered
            idx+=1    
        fileobj.close()
    return result
def findlpcdir(lpcdir,fillmin):
    result=[]
    cachedir=lpcdir
    lpcfilldir=[f for f in glob.glob(cachedir+'/????') if os.path.isdir(f) ]
    lpcfills=[os.path.split(f)[1] for f in lpcfilldir]
    #print lpcfills
    result=[int(f) for f in lpcfills if int(f)>=fillmin]
    return result

if __name__ == "__main__" :
    ofile=open('checklumi.log','w')
    lpcdir='/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/LHCFILES/'
    plotcachedir='/afs/cern.ch/cms/lumi/www/publicplots/public_lumi_plots_cache/pp_all'
    plotfiles=[f for f in glob.glob(os.path.join(plotcachedir,'lumicalc_cache_2012*.csv')) if os.path.getsize(f)>0]
    fillmin=2450
    lpcfill2012=findlpcdir(lpcdir,fillmin)
    lpcfill2012.sort()
    lpcresult={}#{fill:[delivered]}

    plotfilldata={}#{fill:{run:delivered}}
    plotfilldata=parseplotcache(plotfiles,min(lpcfill2012),max(lpcfill2012))

    ofile.write('checking fills %s\n'%str(lpcfill2012))
    ofile.write("-----------------------------------------------------------------------------------\n")
    ofile.write('on %s\n'%str(datetime.datetime.now()) )
    ofile.write("-----------------------------------------------------------------------------------\n")
    tot_lumipplot=0.
    tot_nrunpplot=0
    tot_lumilpc=0.
    tot_nrunlpc=0

    for fill in lpcfill2012:
        nruns_pplot=0
        nrun_lpc=0
        lumi_pplot=0.
        lumi_lpc=0.
        dellumi=0.
        delta=0.
        lpcfile=os.path.join(lpcdir,str(fill),str(fill)+'_summary_CMS.txt')
        if not os.path.exists(lpcfile):
            continue
        l=open(lpcfile,'rb')
        for line in l.readlines():
            line=line.strip()
            rundataline=line.split()
            if len(rundataline)!=4: continue
            lpcdelperrun=float(rundataline[3])
            lpcresult.setdefault(fill,[]).append(lpcdelperrun)
            nrun_lpc+=1
            lumi_lpc+=lpcdelperrun
            tot_nrunlpc+=1
        l.close()
        if plotfilldata.has_key(fill):
            runs=plotfilldata[fill].keys()
            if not runs: continue
            nruns_pplot=len(runs)
            tot_nrunpplot+=nruns_pplot
            runs.sort()
            for run in runs:
                lumi_pplot+=plotfilldata[fill][run]
        tot_lumipplot+=lumi_pplot
        tot_lumilpc+=lumi_lpc
        delta=lumi_pplot-lumi_lpc
        if lumi_lpc:
            dellumi=1.-math.fabs(lumi_pplot/lumi_lpc)
        if nruns_pplot!=nrun_lpc:
            ofile.write('fill: %d plot=%.2f plotrun=%d lpc=%.2f lpcrun=%d diff=%.2f rel=%.3f *\n'%(fill,lumi_pplot,nruns_pplot,lumi_lpc,nrun_lpc,delta,dellumi))            
        else:
            ofile.write('fill: %d plot=%.2f plotrun=%d lpc=%.2f lpcrun=%d diff=%.2f rel=%.3f \n'%(fill,lumi_pplot,nruns_pplot,lumi_lpc,nrun_lpc,delta,dellumi))
    tot_diff=tot_lumipplot-tot_lumilpc
    tot_rel=0.
    if tot_lumilpc!=0:
        tot_rel=1.-tot_lumipplot/tot_lumilpc
    
    ofile.write("-----------------------------------------------------------------------------------\n")
    ofile.write("tot : plot=%.2f lpc=%.2f diff=%.2f rel=%.3f\n" % (tot_lumipplot,tot_lumilpc,tot_diff,tot_rel))
    ofile.write("tot : plotnrun=%d lpcnrun=%d\n"%(tot_nrunpplot,tot_nrunlpc))

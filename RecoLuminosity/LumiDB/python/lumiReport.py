###########################################################
# Luminosity/LumiTag/LumiCorrection report API            #
#                                                         #
# Author:      Zhen Xie                                   #
###########################################################

import os,sys,time
from RecoLuminosity.LumiDB import tablePrinter, csvReporter,CommonUtil
from RecoLuminosity.LumiDB.wordWrappers import wrap_always, wrap_onspace, wrap_onspace_strict

def dumptocsv(fieldnames,result,filename):
    '''
    utility method to dump result to csv file
    '''
    assert(filename)
    if filename.upper()=='STDOUT':
        r=sys.stdout
        r.write(','.join(fieldnames)+'\n')
        for l in result:
            r.write(str(l)+'\n')
    else:
        r=csvReporter.csvReporter(filename)
        r.writeRow(fieldnames)
        r.writeRows(result)
        r.close()
        
def toScreenHeader(commandname,datatagname,normtag,worktag,updatetag,lumitype,toFile=None):
    '''
    input:
       commandname: commandname
       datataginfo: tagname
       normtag: normtag
       worktag: working version
       updatetag: updated version if amy
    '''
    gmtnowStr=time.asctime(time.gmtime())+' UTC'
    updatetagStr='None'
    if updatetag:
        updatetagStr=updatetag
    header=''.join(['*']*80)+'\n'
    header+='* '+gmtnowStr+'\n'
    header+='* lumitype: '+lumitype+' , datatag: '+datatagname+' , normtag: '+normtag+' , worktag: '+worktag+'\n'
    header+='* \n'
    header+='* by:\n'
    header+='* '+commandname+'\n'
    header+='* \n'
    header+='* update: '+updatetag+'\n'
    header+=''.join(['*']*80)+'\n'
    if not toFile:
        sys.stdout.write(header)
    else:
        assert(toFile)
        if toFile.upper()=='STDOUT':
            r=sys.stdout
        else:
            r=open(toFile,'wb')
        r.write(header)
        
def toScreenNormSummary(allnorms):
    '''
    list all known norms summary
    input: {normname:[data_id(0),lumitype(1),istypedefault(2),comment(3),creationtime(4)]}
    '''
    result=[]
    labels=[('Name','Type','IsTypeDefault','Comment','CreationTime')]
    print ' ==  = '
    sorted_allnorms=sorted(allnorms.iteritems(),key=lambda x:x[0],reverse=True)
    for (normname,normvalues) in sorted_allnorms:
        lumitype=normvalues[1]
        istypedefault=str(normvalues[2])
        commentStr=normvalues[3]
        creationtime=normvalues[4]
        result.append([normname,lumitype,istypedefault,commentStr,creationtime])
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'left',delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) ) 

def toScreenNormDetail(normname,norminfo,normvalues):
    '''
    list norm detail
    input:
        normname
        norminfo=[data_id[0],lumitype(1)istypedefault[2],comment[3],creationtime[4]]
        normvalues={since:[corrector(0),{paramname:paramvalue}(1),amodetag(2),egev(3),comment(4)]}
    '''
    lumitype=norminfo[1]
    istypedefault=norminfo[2]
    print '=========================================================='
    print '* Norm: '+normname
    print '* Type: '+lumitype
    print '* isDefault: '+str(istypedefault)
    print '=========================================================='
    labels=[('Since','Func','Parameters','amodetag','egev','comment')]

    result=[]
    print ' ==  = '
    for since in sorted(normvalues):
        normdata=normvalues[since]
        correctorStr=normdata[0]
        paramDict=normdata[1]
        paramDictStr=''
        count=0
        for pname in sorted(paramDict):
            pval=paramDict[pname]
            if count!=0:
                paramDictStr+=' '
            try:
                fpval=float(pval)
                if fpval<1.:
                    paramDictStr+=pname+':'+'%.4f'%fpval
                else:
                    paramDictStr+=pname+':'+'%.2f'%fpval
            except ValueError:
                paramDictStr+=pname+':'+pval
            count+=1
        amodetag=normdata[2]
        egev=str(normdata[3])
        comment=normdata[4]
        result.append([str(since),correctorStr,paramDictStr,amodetag,egev,comment])
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'left',delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,40) ) 

def toScreenTags(tagdata):
    result=[]
    labels=[('Name','Min Run','Max Run','Creation Time')]
    print ' ==  = '
    for tagid in sorted(tagdata):
        taginfo=tagdata[tagid]
        name=taginfo[0]
        minRun=str(taginfo[1])
        maxRun='Open'
        if taginfo[2]!=0:
            maxRun=str(taginfo[2])
        creationtime=taginfo[3]
        result.append([name,minRun,maxRun,creationtime])
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'left',delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) ) 

def toScreenSingleTag(taginfo):
    '''
    input: {run:(lumidataid,trgdataid,hltdataid,comment)}
    '''
    result=[]
    labels=[('Run','Data Id','Insertion Time','Patch Comment')]
    print ' ==  = '
    for run in sorted(taginfo):
        (lumidataid,trgdataid,hltdataid,(ctimestr,comment))=taginfo[run]
        payloadid='-'.join([str(lumidataid),str(trgdataid),str(hltdataid)])
        result.append([str(run),payloadid,ctimestr,comment])
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'left',delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,25) )
    
def toScreenTotDelivered(lumidata,resultlines,scalefactor,irunlsdict=None,noWarning=True,toFile=None):
    '''
    inputs:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),calibratedlumierror(6),(bxidx,bxvalues,bxerrs)(7),(bxidx,b1intensities,b2intensities)(8),fillnum)(9)]}  
    resultlines [[resultrow1],[resultrow2],...,] existing result row
                ('Run:Fill', 'N_LS','N_CMSLS','Delivered','UTCTime','E(GeV)')
    irunlsdict: run/ls selection list. irunlsdict=None means no filter
    '''
    result=[]
    totOldDeliveredLS=0
    totOldCMSLS=0
    totOldDelivered=0.0
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with
    for r in resultlines:
        runfillstr=r[0]
        [runnumstr,fillnumstr]=runfillstr.split(':')
        if irunlsdict and not noWarning:
            if r[1] is not 'n/a':
                datarunlsdict[int(runnumstr)]=[]
        dl=0.0
        if(r[3]!='n/a'):            #delivered
            dl=float(r[3])#in /ub because it comes from file!
            (rr,lumiu)=CommonUtil.guessUnit(dl)
            r[3]='%.3f'%(rr)+' ('+lumiu+')'            
        sls=0
        if(r[1]!='n/a'): #n_ls
            sls=int(r[1])
        totcmsls=0
        if(r[2]!='n/a'):#n_cmsls
            totcmsls=int(r[2])
        totOldDeliveredLS+=sls
        totOldCMSLS+=totcmsls
        totOldDelivered+=dl
        if(r[5]!='n/a'): #egev
            egv=float(r[5])
            r[5]='%.1f'%egv
        result.append(r)
    totls=0
    totcmsls=0
    totdelivered=0.0
    totaltable=[]    
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if not lsdata:
            result.append([str(run)+':0','n/a','n/a','n/a','n/a','n/a'])
            if irunlsdict and not noWarning:
                datarunlsdict[run]=None
            continue
        fillnum=0
        if lsdata[0] and lsdata[0][9]:
            fillnum=lsdata[0][9]
        deliveredData=[]
        nls=0
        existdata=[]
        selectedcmsls=[]
        for perlsdata in lsdata:
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            if not noWarning:
                if cmslsnum:
                    existdata.append(cmslsnum)
            if irunlsdict and irunlsdict[run]:
                if lumilsnum and lumilsnum in irunlsdict[run]:
                    deliveredData.append(perlsdata[5])
                    if cmslsnum:
                        selectedcmsls.append(cmslsnum)
            else:
                deliveredData.append(perlsdata[5])
                if cmslsnum:                    
                    selectedcmsls.append(cmslsnum)
        datarunlsdict[run]=existdata
        nls=len(deliveredData)
        ncmsls=0
        if selectedcmsls:
            ncmsls=len(selectedcmsls)
            totcmsls+=ncmsls
        totls+=nls
        totlumi=sum(deliveredData)
        totdelivered+=totlumi
        (totlumival,lumiunit)=CommonUtil.guessUnit(totlumi)
        beamenergyPerLS=[float(x[4]) for x in lsdata if x[3]=='STABLE BEAMS']
        avgbeamenergy=0.0
        if len(beamenergyPerLS):
            avgbeamenergy=sum(beamenergyPerLS)/len(beamenergyPerLS)
        runstarttime='n/a'
        if lsdata[0] and lsdata[0][2]:
            runstarttime=lsdata[0][2]
            runstarttime=runstarttime.strftime("%m/%d/%y %H:%M:%S")
        if not toFile:
            result.append([str(run)+':'+str(fillnum),str(nls),str(ncmsls),'%.3f'%(totlumival*scalefactor)+' ('+lumiunit+')',runstarttime,'%.1f'%(avgbeamenergy)])
        else:
            result.append([str(run)+':'+str(fillnum),str(nls),str(ncmsls),(totlumi*scalefactor),runstarttime,'%.1f'%(avgbeamenergy)])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    #print 'sortedresult ',sortedresult
    if not toFile:
        labels = [('Run:Fill', 'N_LS','N_CMSLS','Delivered','UTCTime','E(GeV)')]
        print ' ==  = '
        print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'right',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,40) )
        print ' ==  =  Total : '
        (totalDeliveredVal,totalDeliveredUni)=CommonUtil.guessUnit(totdelivered+totOldDelivered)
        totrowlabels = [('Delivered LS','Total CMS LS','Delivered('+totalDeliveredUni+')')]
        totaltable.append([str(totls+totOldDeliveredLS),str(totcmsls+totOldCMSLS),'%.3f'%(totalDeliveredVal*scalefactor)])
        print tablePrinter.indent (totrowlabels+totaltable, hasHeader = True, separateRows = False, prefix = '| ',
                                   postfix = ' |', justify = 'right', delim = ' | ',
                                   wrapfunc = lambda x: wrap_onspace (x, 20))
    else:
        fieldnames = ['Run:Fill', 'N_LS','N_CMSLS','Delivered(/ub)','UTCTime','E(GeV)']
        filename=toFile
        dumptocsv(fieldnames,sortedresult,filename)
                
def toScreenOverview(lumidata,resultlines,scalefactor,irunlsdict=None,noWarning=True,toFile=None):
    '''
    input:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),(bxidx,b1intensities,b2intensities)(9),fillnum(10)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]

    totOldDeliveredLS=0
    totOldSelectedLS=0
    totOldDelivered=0.0
    totOldRecorded=0.0
    
    totaltable=[]
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with
    for r in resultlines:
        runfillstr=r[0]
        [runnumstr,fillnumstr]=runfillstr.split(':')
        if irunlsdict and not noWarning:
            if r[1] is not 'n/a':
                datarunlsdict[int(runnumstr)]=[]
        dl=0.0
        if(r[2]!='n/a'):            
            dl=float(r[2])#delivered in /ub because it comes from file!
            (rr,lumiu)=CommonUtil.guessUnit(dl)
            r[2]='%.3f'%(rr)+' ('+lumiu+')'
        dls=0
        if(r[1]!='n/a'):
            dls=int(r[1])
        totOldDeliveredLS+=dls
        totOldDelivered+=dl
        rls=0
        if(r[3]!='n/a'):
            rlsstr=r[3]
            listcomp=rlsstr.split(', ')
            for lstr in listcomp:
                enddigs=lstr[1:-1].split('-')
                lsmin=int(enddigs[0])
                lsmax=int(enddigs[1])
                rls=lsmax-lsmin+1
                totOldSelectedLS+=rls
        if(r[4]!='n/a'):
            rcd=float(r[4])#recorded in /ub because it comes from file!
            (rrcd,rlumiu)=CommonUtil.guessUnit(rcd)
            r[4]='%.3f'%(rrcd)+' ('+rlumiu+')'
        totOldRecorded+=rcd
        result.append(r)
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if not lsdata:
            result.append([str(run)+':0','n/a','n/a','n/a','n/a'])
            if irunlsdict and irunlsdict[run] and not noWarning:
                datarunlsdict[run]=None
            continue
        fillnum=0
        if lsdata[0] and lsdata[0][10]:
            fillnum=lsdata[0][10]
        deliveredData=[]
        recordedData=[]
        nls=0
        existdata=[]
        selectedcmsls=[]
        for perlsdata in lsdata:
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]
            if not noWarning:
                if cmslsnum:
                    existdata.append(cmslsnum)
            if irunlsdict and irunlsdict[run]:
                if lumilsnum and lumilsnum in irunlsdict[run]:
                    if perlsdata[5] is not None:
                        deliveredData.append(perlsdata[5])
                    if perlsdata[6]:
                        recordedData.append(perlsdata[6])
                    selectedcmsls.append(lumilsnum)
            else:
                deliveredData.append(perlsdata[5])
                if perlsdata[6]:
                    recordedData.append(perlsdata[6])
                if cmslsnum:                    
                    selectedcmsls.append(cmslsnum)
        datarunlsdict[run]=existdata
        nls=len(deliveredData)
        totdelivered=sum(deliveredData)
        totalDelivered+=totdelivered
        totalDeliveredLS+=len(deliveredData)
        (totdeliveredlumi,deliveredlumiunit)=CommonUtil.guessUnit(totdelivered)
        totrecorded=sum(recordedData)
        totalRecorded+=totrecorded
        (totrecordedlumi,recordedlumiunit)=CommonUtil.guessUnit(totrecorded)
        totalSelectedLS+=len(selectedcmsls)
        if len(selectedcmsls)==0:
            selectedlsStr='n/a'
        else:
            selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        if not toFile:
            result.append([str(run)+':'+str(fillnum),str(nls),'%.3f'%(totdeliveredlumi*scalefactor)+' ('+deliveredlumiunit+')',selectedlsStr,'%.3f'%(totrecordedlumi*scalefactor)+' ('+recordedlumiunit+')'])
        else:
            result.append([str(run)+':'+str(fillnum),nls,totdelivered*scalefactor,selectedlsStr,totrecorded*scalefactor])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    if irunlsdict and not noWarning:
        for run,cmslslist in irunlsdict.items():
            if run not in datarunlsdict.keys() or datarunlsdict[run] is None:
                sys.stdout.write('[WARNING] selected run '+str(run)+' not in lumiDB or has no qualified data\n')
                continue
            if cmslslist:
                for ss in cmslslist:
                    if ss not in datarunlsdict[run]:
                        sys.stdout.write('[WARNING] lumi or trg for selected run/ls '+str(run)+' '+str(ss)+' not in lumiDB\n')
    if not toFile:
        labels = [('Run:Fill', 'Delivered LS', 'Delivered','Selected LS','Recorded')]    
        print ' ==  = '
        print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'right',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
        print ' ==  =  Total : '
        (totalDeliveredVal,totalDeliveredUni)=CommonUtil.guessUnit(totalDelivered+totOldDelivered)
        (totalRecordedVal,totalRecordedUni)=CommonUtil.guessUnit(totalRecorded+totOldRecorded)
        totrowlabels = [('Delivered LS','Delivered('+totalDeliveredUni+')','Selected LS','Recorded('+totalRecordedUni+')')]
        totaltable.append([str(totalDeliveredLS+totOldDeliveredLS),'%.3f'%(totalDeliveredVal*scalefactor),str(totalSelectedLS+totOldSelectedLS),'%.3f'%(totalRecordedVal*scalefactor)])
        print tablePrinter.indent (totrowlabels+totaltable, hasHeader = True, separateRows = False, prefix = '| ',
                                   postfix = ' |', justify = 'right', delim = ' | ',
                                   wrapfunc = lambda x: wrap_onspace (x, 20))
    else:
        fieldnames = ['Run:Fill', 'DeliveredLS', 'Delivered(/ub)','SelectedLS','Recorded(/ub)']
        filename=toFile
        dumptocsv(fieldnames,sortedresult,filename)
        
def toScreenLumiByLS(lumidata,resultlines,scalefactor,irunlsdict=None,noWarning=True,toFile=None):
    '''
    input:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),(bxidx,b1intensities,b2intensities)(9),fillnum(10),pu(11)]}
    {run:None}  None means no run in lumiDB, 
    {run:[]} [] means no lumi for this run in lumiDB
    {run:[....deliveredlumi(5),recordedlumi(6)None]} means no trigger in lumiDB
    {run:cmslsnum(1)==0} means either not cmslsnum or is cms but not selected, therefore set recordedlumi=0,efflumi=0
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    totalrow = []
    
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0

    totOldDeliveredLS = 0
    totOldSelectedLS = 0
    totOldDelivered = 0.0
    totOldRecorded = 0.0

    maxlslumi = 0.0
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with
    for rline in resultlines:
        runfillstr=rline[0]
        [runnumstr,fillnumstr]=runfillstr.split(':')
        if irunlsdict and not noWarning:
            if rline[1] is not 'n/a':
                datarunlsdict[int(runnumstr)]=[]
        myls=rline[1]
        if myls!='n/a':
            [luls,cmls]=myls.split(':')
            totOldDeliveredLS+=1
            if cmls!='0':
                totOldSelectedLS+=1
                if irunlsdict and not noWarning:
                    datarunlsdict[int(runnumstr)].append(int(myls))                    
        dl=rline[5]
        if rline[5]!='n/a':
            dl=float(rline[5])#delivered in /ub
            if dl>maxlslumi: maxlslumi=dl
            rline[5]=dl
            totOldDelivered+=dl
        rl=rline[6]
        if rline[6]!='n/a':
           rl=float(rline[6])#recorded in /ub
           rline[6]=rl
           totOldRecorded+=rl
        result.append(rline)
        
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if not lsdata:
            #result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a','n/a'])
            if irunlsdict and irunlsdict[run] and not noWarning:
                datarunlsdict[run]=None
                #print '[WARNING] selected but no lumi data for run '+str(run)
            continue
        fillnum=0
        if lsdata[0] and lsdata[0][10]:
            fillnum=lsdata[0][10]
        existdata=[]
        #if irunlsdict and not noWarning:
        #    existdata=[x[1] for x in rundata if x[1] ]
        #    datarunlsdict[run]=existdata
        for perlsdata in lsdata:
            lumilsnum=perlsdata[0]
            cmslsnum=perlsdata[1]#triggered ls
            if not noWarning:
                if cmslsnum:
                    existdata.append(cmslsnum)
            ts=perlsdata[2]
            bs=perlsdata[3]
            begev=perlsdata[4]
            deliveredlumi=perlsdata[5]
            npu=perlsdata[11]
            if deliveredlumi>maxlslumi: maxlslumi=deliveredlumi
            recordedlumi=0.
            if perlsdata[6]:
                recordedlumi=perlsdata[6]
            if irunlsdict and irunlsdict[run]:
                if run in irunlsdict and lumilsnum in irunlsdict[run]:
                    result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),ts.strftime('%m/%d/%y %H:%M:%S'),bs,'%.1f'%begev,deliveredlumi,recordedlumi,npu])                
                    totalDelivered+=deliveredlumi
                    totalRecorded+=recordedlumi
                    totalDeliveredLS+=1
                    totalSelectedLS+=1
            else:
                result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),ts.strftime('%m/%d/%y %H:%M:%S'),bs,'%.1f'%begev,deliveredlumi,recordedlumi,npu])
                totalDelivered+=deliveredlumi
                totalRecorded+=recordedlumi
                totalDeliveredLS+=1
                if cmslsnum :
                    totalSelectedLS+=1
        datarunlsdict[run]=existdata       
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))    
    if irunlsdict and not noWarning:
        for run,cmslslist in irunlsdict.items():
            if run not in datarunlsdict.keys() or datarunlsdict[run] is None:
                sys.stdout.write('[WARNING] selected run '+str(run)+' not in lumiDB or has no qualified data\n')
                continue
            if cmslslist:
                for ss in cmslslist:
                    if ss not in datarunlsdict[run]:
                        sys.stdout.write('[WARNING] lumi or trg for selected run/ls '+str(run)+' '+str(ss)+' not in lumiDB\n')
    if not toFile:                    
        (lsunitstring,unitdenomitor)=CommonUtil.lumiUnitForPrint(maxlslumi*scalefactor)
        labels = [ ('Run:Fill','LS','UTCTime','Beam Status','E(GeV)','Del('+lsunitstring+')','Rec('+lsunitstring+')','avgPU') ]                    
        perlsresult=[]
        for entry in sortedresult:
            delumi=entry[5]
            if delumi!='n/a':
                delumi='%.3f'%float(float(delumi*scalefactor)/float(unitdenomitor))
            reclumi=entry[6]
            if reclumi!='n/a':
                reclumi='%.3f'%float(float(reclumi*scalefactor)/float(unitdenomitor))
            avgPU=entry[7]
            if avgPU!='n/a':                
                if avgPU>0:
                    avgPU='%.3f'%avgPU
                else:
                    avgPU='0'
            perlsresult.append([entry[0],entry[1],entry[2],entry[3],entry[4],delumi,reclumi,avgPU])
        totdeliveredlumi=0.0
        deliveredlumiunit='/ub'
        (totdeliveredlumi,deliveredlumiunit)=CommonUtil.guessUnit((totalDelivered+totOldDelivered)*scalefactor)
        totrecordedlumi=0.0
        recordedlumiunit='/ub'
        (totrecordedlumi,recordedlumiunit)=CommonUtil.guessUnit((totalRecorded+totOldRecorded)*scalefactor)
        lastrowlabels = [ ('Delivered LS','Selected LS', 'Delivered('+deliveredlumiunit+')', 'Recorded('+recordedlumiunit+')')]
        totalrow.append ([str(totalDeliveredLS+totOldDeliveredLS),str(totalSelectedLS+totOldSelectedLS),'%.3f'%(totdeliveredlumi),'%.3f'%(totrecordedlumi)])
        print ' ==  = '
        print tablePrinter.indent (labels+perlsresult, hasHeader = True, separateRows = False, prefix = '| ',
                                   postfix = ' |', justify = 'right', delim = ' | ',
                                   wrapfunc = lambda x: wrap_onspace_strict (x, 22))
        print ' ==  =  Total : '
        print tablePrinter.indent (lastrowlabels+totalrow, hasHeader = True, separateRows = False, prefix = '| ',
                                   postfix = ' |', justify = 'right', delim = ' | ',
                                   wrapfunc = lambda x: wrap_onspace (x, 20))    
    else:
        fieldnames=['Run:Fill','LS','UTCTime','Beam Status','E(GeV)','Delivered(/ub)','Recorded(/ub)','avgPU']
        filename=toFile
        dumptocsv(fieldnames,sortedresult,filename)

def toScreenLSEffective(lumidata,resultlines,scalefactor,irunlsdict=None,noWarning=True,toFile=None):
    '''
    input:  {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]}(8),bxdata(9),beamdata(10),fillnum(11)]}
    '''
    result=[]#[run,ls,hltpath,l1bitname,hltpresc,l1presc,efflumi]
    totalrow=[]
    totSelectedLSDict={}
    totRecordedDict={}
    totEffectiveDict={}

    totOldSelectedLSDict={}
    totOldRecordedDict={}
    totOldEffectiveDict={}

    maxlslumi = 0.0
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with
    for rline in resultlines:
        runfillstr=rline[0]
        [runnumstr,fillnumstr]=runfillstr.split(':')
        if irunlsdict and not noWarning:
            if rline[1] is not 'n/a':
                datarunlsdict[int(runnumstr)]=[]
        myls=rline[1]
        mypath=rline[2]

        if myls and myls!='n/a' and mpath and mpath!='n/a':
            totOldSelectedLSDict[mypath]=0
            totOldRecordedDict[mypath]=0.
            totOldEffectiveDict[mypath]=0.
        if myls!='n/a':
            [luls,cmls]=myls.split(':')
            if cmls!='0':
                if totOldSelectedLSDict.has_key(mypath):
                    totOldSelectedLSDict[mypath]+=1
                if irunlsdict and not noWarning:
                    datarunlsdict[int(runnumstr)].append(int(myls))         
        myrecorded=0.
        if rline[6]!='n/a':
            myrecorded=float(rline[6])
            if myrecorded>maxlslumi:maxlslumi=myrecorded
            if totOldRecordedDict.has_key(mypath):
                totOldRecordedDict[mypath]+=myrecorded
            rline[6]=myrecorded
        myeff={}
        if rline[7]!='n/a':
            myeff=float(rline[7])
            if totOldEffectiveDict.has_key(mypath):
                totOldEffectiveDict[mypath]+=myeff
            rline[7]=myeff
        result.append(rline)        

    for run in lumidata.keys():#loop over runs
        lsdata=lumidata[run]
        if not lsdata:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a','n/a'])
            if irunlsdict and irunlsdict[run] and not noWarning:
                datarunlsdict[run]=None
            continue
        fillnum=0
        if lsdata[0] and lsdata[0][11]:
            fillnum=lsdata[0][11]
        datarunlsdict[run]=[]
        for thisls in lsdata:
            lumilsnum=thisls[0]
            cmslsnum=thisls[1]#triggered ls
            if not cmslsnum: continue
            efflumiDict=thisls[8]# this ls has no such path?            
            recordedlumi=0.
            if thisls[6]:
                recordedlumi=thisls[6]
            if recordedlumi>maxlslumi:maxlslumi=recordedlumi
            if not efflumiDict:
                result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),'n/a','n/a','n/a','n/a',recordedlumi,'n/a'])
                continue

            for hltpathname in sorted(efflumiDict):
                if hltpathname and hltpathname !='n/a' :
                    if not totRecordedDict.has_key(hltpathname):
                        totRecordedDict[hltpathname]=0. 
                    if not totSelectedLSDict.has_key(hltpathname):
                        totSelectedLSDict[hltpathname]=0
                    if not totEffectiveDict.has_key(hltpathname):
                        totEffectiveDict[hltpathname]=0.
                    totSelectedLSDict[hltpathname]+=1
                    totRecordedDict[hltpathname]+=recordedlumi
                pathdata=efflumiDict[hltpathname]
                l1name=pathdata[0]
                cleanl1name='n/a'
                if l1name:
                    cleanl1name=l1name.replace('"','')
                l1presc='0'
                if pathdata[1]:
                    l1presc=str(pathdata[1])
                hltpresc='0'
                if pathdata[2]:
                    hltpresc=str(pathdata[2])
                lumival=0.
                if pathdata[3]:
                    lumival=pathdata[3]
                result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),hltpathname,cleanl1name,hltpresc,l1presc,recordedlumi,lumival])
                if hltpathname and hltpathname !='n/a' :
                    totEffectiveDict[hltpathname]+=lumival
                if irunlsdict and not noWarning:
                    datarunlsdict[run].append(int(cmslsnum))
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    if irunlsdict and not noWarning:
        for run,cmslslist in irunlsdict.items():
            if run not in datarunlsdict.keys() or datarunlsdict[run] is None:
                sys.stdout.write('[WARNING] selected run '+str(run)+' not in lumiDB or has no HLT data\n')
                continue
            if cmslslist:
                for ss in cmslslist:
                    if ss not in datarunlsdict[run]:
                        sys.stdout.write('[WARNING] selected run/ls '+str(run)+' '+str(ss)+' not in lumiDB or has no qualified data\n')
                        
    if not toFile:
        (lsunitstring,unitdenomitor)=CommonUtil.lumiUnitForPrint(maxlslumi*scalefactor)
        labels = [('Run:Fill','LS','HLTpath','L1bit','HLTpresc','L1presc','Recorded('+lsunitstring+')','Effective('+lsunitstring+')')]
        perlsresult=[]
        for entry in sortedresult:
            reclumi=entry[6]
            if reclumi!='n/a':
                reclumi='%.3f'%float(float(reclumi*scalefactor)/float(unitdenomitor))
            efflumi=entry[7]
            if efflumi!='n/a':
                efflumi='%.3f'%float(float(efflumi*scalefactor)/float(unitdenomitor))
            perlsresult.append([entry[0],entry[1],entry[2],entry[3],entry[4],entry[5],reclumi,efflumi])
        print ' ==  = '
        print tablePrinter.indent (labels+perlsresult, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'right',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace_strict(x,25) )
        for mpath in sorted(totRecordedDict):
            totSelectedLS=totSelectedLSDict[mpath]
            if totOldSelectedLSDict.has_key(mpath):
                totSelectedLS+=totOldSelectedLS[mpath]
            totRecorded=totRecordedDict[mpath]
            if totOldRecordedDict.has_key(mpath):
                totRecorded+=totOldRecorded[mpath]
            totRecorded=float(totRecorded*scalefactor)/float(unitdenomitor)
            totEffective=totEffectiveDict[mpath]
            if totOldEffectiveDict.has_key(mpath):
                totEffective+=totOldEffective[mpath]
            totEffective=float(totEffective*scalefactor)/float(unitdenomitor)
            totalrow.append([str(totSelectedLS),mpath,'%.3f'%(totRecorded),'%.3f'%(totEffective)])
        lastrowlabels = [ ('Selected LS','HLTPath','Recorded('+lsunitstring+')','Effective('+lsunitstring+')')]
        print ' ==  =  Total : '
        print tablePrinter.indent (lastrowlabels+totalrow, hasHeader = True, separateRows = False, prefix = '| ',
                                   postfix = ' |', justify = 'right', delim = ' | ',
                                   wrapfunc = lambda x: wrap_onspace (x, 20))
    else:
        fieldnames = ['Run:Fill','LS','HLTpath','L1bit','HLTpresc','L1presc','Recorded(/ub)','Effective(/ub)']
        filename=toFile
        dumptocsv(fieldnames,sortedresult,filename)
        
def toScreenTotEffective(lumidata,resultlines,scalefactor,irunlsdict=None,noWarning=True,toFile=None):

    '''
    input:  {run:[lumilsnum(0),triggeredls(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]}(8),bxdata(9),beamdata](10),fillnum(11)}
    screen Run,SelectedLS,Recorded,HLTPath,L1Bit,Effective
    '''
    result=[]#[run,selectedlsStr,recordedofthisrun,hltpath,l1bit,efflumi]
    
    totdict={}#{hltpath:[nls,toteff]}
    selectedcmsls=[]
    alltotrecorded=0.0
    alleffective=0.0
    recordedPerpathPerrun={}#{path:{run:recorded}}
    selectedPerpathPerrun={}#{path:{run:totselected}}
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with
    for rline in resultlines:
        runfillstr=rline[0]
        [runnumstr,fillnumstr]=runfillstr.split(':')
        myls=rline[1]
        if irunlsdict and not noWarning:
            if myls is not 'n/a':
                datarunlsdict[int(runnumstr)]=[]  
        mypath=rline[3]
        if mypath!='n/a':
            mypath=mypath.split('(')[0]
            if not totdict.has_key(mypath):
                totdict[mypath]=[0,0.0]
                recordedPerpathPerrun[mypath]={}
                selectedPerpathPerrun[mypath]={}
        if myls!='n/a':
            listcomp=myls.split(', ')
            for lstr in listcomp:
                enddigs=lstr[1:-1].split('-')
                lsmin=int(enddigs[0])
                lsmax=int(enddigs[1])
                rls=lsmax-lsmin+1
                totdict[mypath][0]+=rls                
            selectedPerrun[mypath].setdefault(int(myrun),totdict[mypath][0])
        myrecorded=rline[2]
        if myrecorded!='n/a':
            recordedPerpathPerrun[mypath].setdefault(int(myrun),float(myrecorded))
            (rr,lumiu)=CommonUtil.guessUnit(float(myrecorded))
            rline[2]='%.3f'%(rr)+' ('+lumiu+')'
        myeff=rline[5]
        if myeff!='n/a':
            reff=float(myeff)
            (rr,lumiu)=CommonUtil.guessUnit(float(reff))
            rline[5]='%.3f'%(rr)+' ('+lumiu+')'
            totdict[mypath][1]+=reff
        result.append(rline)
    for run in lumidata.keys():#loop over runs
        lsdata=lumidata[run]
        hprescdict={}
        lprescdict={}
        if not lsdata:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a'])
            if irunlsdict and irunlsdict[run] and not noWarning:
                datarunlsdict[run]=None
            continue
        fillnum=0
        if lsdata[0] and lsdata[0][11]:
            fillnum=lsdata[0][11]
        selectedcmsls=[x[1] for x in lsdata if x[1]]
        totefflumiDict={}
        totrecorded=0.0
        toteffective=0.0
        pathmap={}#{hltpathname:1lname}
        existdata=[]
        for thisls in lsdata:            
            cmslsnum=thisls[1]
            if not noWarning:
                if cmslsnum:
                    existdata.append(cmslsnum)
            efflumiDict=thisls[8]# this ls has no such path?
            recordedlumi=0.0
            if thisls[6]:
                recordedlumi=thisls[6]
            totrecorded+=recordedlumi
            if not efflumiDict:#no hltdata for this LS
                lumival=0.
                if cmslsnum in selectedcmsls:
                    selectedcmsls.remove(cmslsnum)
                continue            
            for hltpathname in sorted(efflumiDict):
                pathdata=efflumiDict[hltpathname]
                if not totefflumiDict.has_key(hltpathname):
                    totefflumiDict[hltpathname]=0.0
                    pathmap[hltpathname]='n/a'
                l1name=pathdata[0]
                l1presc=pathdata[1]
                hltpresc=pathdata[2]
                lumival=pathdata[3]
                recordedPerpathPerrun.setdefault(hltpathname,{})
                selectedPerpathPerrun.setdefault(hltpathname,{})
                if not totdict.has_key(hltpathname):
                    totdict[hltpathname]=[0,0.0]
                if l1presc is None or hltpresc is None:#if found all null prescales and if it is in the selectedcmsls, remove it because incomplete
                    if cmslsnum in selectedcmsls:
                        selectedcmsls.remove(cmslsnum)
                else:                    
                    if not hprescdict.has_key(hltpathname):
                        hprescdict[hltpathname]=[]
                    hprescdict[hltpathname].append(hltpresc)
                    if not lprescdict.has_key(l1name):
                        lprescdict[l1name]=[]
                    lprescdict[l1name].append(l1presc)
                    if cmslsnum!=0:
                        totdict[hltpathname][0]+=1                    
                    if lumival:
                        totdict[hltpathname][1]+=lumival
                        totefflumiDict[hltpathname]+=lumival
                        pathmap[hltpathname]=l1name
                    recordedPerpathPerrun[hltpathname][run]=totrecorded
                    selectedPerpathPerrun[hltpathname][run]=len(selectedcmsls)
        if len(selectedcmsls)==0:
            selectedlsStr='n/a'
        else:
            selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        if irunlsdict and not noWarning:
            datarunlsdict[run]=selectedcmsls
        
        for name in sorted(totefflumiDict):
            lname=pathmap[name]
            totrecordedinrun=recordedPerpathPerrun[name][run]
            hprescs=list(set(hprescdict[name]))
            hprescStr='('+','.join(['%d'%(x) for x in hprescs])+')'
            (totrecval,totrecunit)=CommonUtil.guessUnit(totrecordedinrun*scalefactor)
            effval='n/a'

            effvalStr='n/a'
            lprescStr='n/a'
            cleanlname=''
            if lname!='n/a':
                effval=totefflumiDict[name]*scalefactor
                lprescs=list(set(lprescdict[lname]))
                lprescStr='('+','.join(['%d'%(x) for x in lprescs])+')'
                cleanlname=lname.replace('"','')
                (efflumival,efflumiunit)=CommonUtil.guessUnit(effval)
                effvalStr='%.3f'%(efflumival)+'('+efflumiunit+')'
            if not toFile:
                result.append([str(run)+':'+str(fillnum),selectedlsStr,'%.3f'%(totrecval)+'('+totrecunit+')',name+hprescStr,cleanlname+lprescStr,effvalStr])
            else:
                result.append([str(run)+':'+str(fillnum),selectedlsStr,totrecordedinrun*scalefactor,name+hprescStr,cleanlname+lprescStr,effval])
                
    if irunlsdict and not noWarning:
        for run,cmslslist in irunlsdict.items():
            if run not in datarunlsdict.keys() or datarunlsdict[run] is None:
                sys.stdout.write('[WARNING] selected run '+str(run)+' not in lumiDB or has no HLT data\n')
                continue
            if cmslslist:
                for ss in cmslslist:
                    if ss not in datarunlsdict[run]:
                        sys.stdout.write('[WARNING] selected run/ls '+str(run)+' '+str(ss)+' not in lumiDB or has no HLT data\n')
                        
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))

    if not toFile:
        labels = [('Run:Fill','SelectedLS','Recorded','HLTpath(Presc)','L1bit(Presc)','Effective')]
        print ' ==  = '
        print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'right',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace_strict(x,22) )
        print ' ==  =  Total : '
        lastrowlabels=[('HLTPath','SelectedLS','Recorded','Effective')]
        totresult=[]
        for hname in sorted(totdict):
            hdata=totdict[hname]
            totnls=hdata[0]
            (toteffval,toteffunit)=CommonUtil.guessUnit(hdata[1]*scalefactor)
            alltotrecorded=0.0
            selectedThispath=selectedPerpathPerrun[hname]
            for runnumber,nselected in selectedThispath.items():
                if nselected==0: continue
                alltotrecorded+=recordedPerpathPerrun[hname][runnumber]
            (alltotrecordedVal,alltotrecordedunit)=CommonUtil.guessUnit(alltotrecorded*scalefactor)                                                   
            totresult.append([hname,str(totnls),'%.3f'%(alltotrecordedVal)+'('+alltotrecordedunit+')','%.3f'%(toteffval)+'('+toteffunit+')'])
        print tablePrinter.indent (lastrowlabels+totresult, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'right',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    else:
        fieldnames=['Run:Fill','SelectedLS','Recorded','HLTpath(Presc)','L1bit(Presc)','Effective(/ub)']
        filename=toFile
        dumptocsv(fieldnames,sortedresult,filename)
            
def toCSVLumiByLSXing(lumidata,scalefactor,filename,irunlsdict=None,noWarning=True):
    '''
    input:{run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),bxdata(8),beamdata(9),fillnum(10)]}
    output:
    fieldnames=['Run:Fill','LS','UTCTime','Delivered(/ub)','Recorded(/ub)','BX']
    '''
    result=[]
    assert(filename)
    fieldnames=['run:fill','ls','UTCTime','delivered(/ub)','recorded(/ub)','[bx,Hz/ub]']
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with     
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run)+':0','n/a','n/a','n/a','n/a','n/a'])
            if irunlsdict and irunlsdict[run]:
                print '[WARNING] selected but no lumi data for run '+str(run)
            continue
        fillnum=0
        if rundata and rundata[0][10]:
            fillnum=rundata[0][10]
        if irunlsdict and not noWarning:
            existdata=[x[1] for x in rundata if x[1] ]
            datarunlsdict[run]=existdata
        for lsdata in rundata:
            lumilsnum=lsdata[0]
            cmslsnum=0
            if lsdata and lsdata[1]:
                cmslsnum=lsdata[1]
            tsStr='n/a'
            if lsdata and lsdata[2]:
                ts=lsdata[2]
                tsStr=ts.strftime('%m/%d/%y %H:%M:%S')
            deliveredlumi=0.
            if lsdata[5]:
                deliveredlumi=lsdata[5]
            recordedlumi=0.
            if lsdata[6]:
                recordedlumi=lsdata[6]
            (bxidxlist,bxvaluelist,bxerrorlist)=lsdata[8]
            if irunlsdict and irunlsdict[run]:
                if run in irunlsdict and cmslsnum in irunlsdict[run]:
                    if bxidxlist and bxvaluelist:
                        bxresult=[]
                        bxinfo=CommonUtil.transposed([bxidxlist,bxvaluelist])
                        bxresult=CommonUtil.flatten([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),tsStr,deliveredlumi*scalefactor,recordedlumi*scalefactor,bxinfo])
                        result.append(bxresult)
                    else:
                        result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),tsStr,deliveredlumi*scalefactor,recordedlumi*scalefactor])
            else:
                if bxidxlist and bxvaluelist:
                    bxresult=[]
                    bxinfo=CommonUtil.transposed([bxidxlist,bxvaluelist])
                    bxresult=CommonUtil.flatten([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),tsStr,deliveredlumi*scalefactor,recordedlumi*scalefactor,bxinfo])
                    result.append(bxresult)
                else:
                    result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),tsStr,deliveredlumi*scalefactor,recordedlumi*scalefactor])
    r=None
    if filename.upper()=='STDOUT':
        r=sys.stdout
        r.write(','.join(fieldnames)+'\n')
        for l in result:
            r.write(str(l)+'\n')
    else:
        r=csvReporter.csvReporter(filename)
        r.writeRow(fieldnames)
        r.writeRows(result)
    
def toScreenLSTrg(trgdata,iresults=[],irunlsdict=None,noWarning=True,toFile=None):
    '''
    input:{run:[[cmslsnum,deadfrac,deadtimecount,bitzero_count,bitzero_prescale,[(name,count,presc),]],..]
    '''
    result=[]
    datarunlsdict={}#{run:[ls,...]}from data. construct it only if there is irunlsdict to compare with
    for rline in iresults:
        runnumStr=rline[0]
        cmslsnumStr=rline[1]
        if irunlsdict and not noWarning:
            if runnumStr is not 'n/a' and not datarunlsdict.has_key(int(runnumStr)):
                datarunlsdict[int(runnumstr)]=[]
            if cmslsnumStr!='n/a':
                datarunlsdict[int(runnumStr)].append(int(cmslsnumStr))
        result.append(rline)
    for run in trgdata.keys():
        rundata=trgdata[run]
        if not rundata:
            ll=[str(run),'n/a','n/a','n/a']
            result.append(ll)
            if irunlsdict and not noWarning:
                print '[WARNING] selected but no trg data for run '+str(run)
            continue
        if irunlsdict and not noWarning:
            existdata=[x[0] for x in rundata if x[0] ]
            datarunlsdict[run]=existdata
        deadfrac=0.0
        bitdataStr='n/a'
        for lsdata in rundata:
            cmslsnum=lsdata[0]
            deadfrac=lsdata[1]
            deadcount=lsdata[2]
            bitdata=lsdata[5]# already sorted by name
            if bitdata:
              flatbitdata=["("+x[0]+',%d'%x[1]+',%d'%x[2]+")" for x in bitdata if x[0]!='False']
              bitdataStr=' '.join(flatbitdata)
            if irunlsdict and irunlsdict[run]:
                if run in irunlsdict and cmslsnum in irunlsdict[run]:
                    result.append([str(run),str(cmslsnum),'%.4f'%(deadfrac),bitdataStr])
            else:
                result.append([str(run),str(cmslsnum),'%.4f'%(deadfrac),bitdataStr])
    if irunlsdict and not noWarning:
        for run,cmslslist in irunlsdict.items():
            if run not in datarunlsdict.keys() or datarunlsdict[run] is None:
                sys.stdout.write('[WARNING] selected run '+str(run)+' not in lumiDB or has no qualified data\n')
                continue
            if cmslslist:
                for ss in cmslslist:
                    if ss not in datarunlsdict[run]:
                        sys.stdout.write('[WARNING] selected run/ls '+str(run)+' '+str(ss)+' not in lumiDB\n')
                        
    if not toFile:
        print ' ==  = '
        labels = [('Run', 'LS', 'dfrac','(bitname,count,presc)')]
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'left',delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,70) )
    else:
        filename=toFile
        fieldnames=['Run','LS','dfrac','(bitname,count,presc)']
        dumptocsv(fieldnames,result,filename)

def toScreenLSHlt(hltdata,iresults=[],toFile=None):
    '''
    input:{runnumber:[(cmslsnum,[(hltpath,hltprescale,l1pass,hltaccept),...]),(cmslsnum,[])})}
    '''
    result=[]
    for r in iresults:
        result.append(r)
    for run in hltdata.keys():
        if hltdata[run] is None:            
            ll=[str(run),'n/a','n/a','n/a','n/a','n/a']
            continue
        perrundata=hltdata[run]
        for lsdata in perrundata:
            cmslsnum=lsdata[0]
            allpathinfo=lsdata[1]
            allpathresult=[]
            for thispathinfo in allpathinfo:
                thispathname=thispathinfo[0]
                thispathpresc=thispathinfo[1]
                thisl1pass=None
                thishltaccept=None
                thispathresult=[]
                thispathresult.append(thispathname)
                if thispathpresc is None:
                    thispathpresc='n/a'
                else:
                    thispathresult.append('%d'%thispathpresc)
                thisl1pass=thispathinfo[2]
                if thispathinfo[2] is None:
                    thispathresult.append('n/a')
                else:
                    thispathresult.append('%d'%thisl1pass)
                thishltaccept=thispathinfo[3]
                if thispathinfo[3] is None:
                    thispathresult.append('n/a')
                else:
                    thispathresult.append('%d'%thishltaccept)

                thispathresultStr='('+','.join(thispathresult)+')'
                allpathresult.append(thispathresultStr)
            result.append([str(run),str(cmslsnum),', '.join(allpathresult)])
            
    if not toFile:
        print ' ==  = '
        labels = [('Run', 'LS', '(hltpath,presc,l1pass,hltaccept)')]
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'left',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,70) )
    else:
        fieldnames=['Run','LS','(hltpath,presc,l1pass,hltaccept)']
        filename=toFile
        dumptocsv(fieldnames,result,filename)
    
def toScreenConfHlt(hltconfdata,iresults=[],toFile=None):
    '''
    input : {runnumber,[(hltpath,l1seedexpr,l1bitname),...]}
    '''
    result=[]
    for r in iresults:
        pp=r[1]
        pp=' '.join([pp[i:i+25] for i in range(0,len(pp),25)])
        sdepr=r[2]
        sdepr=' '.join([sdepr[i:i+25] for i in range(0,len(sdepr),25)])
        lb=r[3]
        lb=' '.join([lb[i:i+25] for i in range(0,len(lb),25)])
        result.append([r[0],pp,sdepr,lb])
    for run in sorted(hltconfdata):
        pathdata=hltconfdata[run]
        if pathdata is None:
            result.append([str(run),'n/a','n/a','n/a'])
            continue
        for thispathinfo in pathdata:
            thispath=thispathinfo[0]
            thispath=' '.join([thispath[i:i+25] for i in range(0,len(thispath),25)])
            thisseed=thispathinfo[1]
            thisseed=' '.join([thisseed[i:i+25] for i in range(0,len(thisseed),25)]).replace('"','')
            thisbit=thispathinfo[2]
            if not thisbit:
                thisbit='n/a'
            else:
                thisbit=' '.join([thisbit[i:i+25] for i in range(0,len(thisbit),25)]).replace('"','')
            result.append([str(run),thispath,thisseed,thisbit])
    if not toFile:
        labels=[('Run','hltpath','l1seedexpr','l1bit')]
        print ' ==  = '
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'left',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace(x,25) )
    else:
        filename=toFile
        fieldnames=['Run','hltpath','l1seedexpr','l1bit']
        dumptocsv(fieldnames,sortedresult,filename)

def toScreenLSBeam(beamdata,iresults=[],dumpIntensity=False,toFile=None):
    '''
    input: {run:[(lumilsnum(0),cmslsnum(1),beamstatus(2),beamenergy(3),ncollidingbunches(4),beaminfolist(4)),..]}
    beaminfolist:[(bxidx,b1,b2)]
    '''
    result=[]
    for rline in iresults:
        result.append(rline)
    for run in sorted(beamdata):
        perrundata=beamdata[run]
        if perrundata is None:            
            ll=[str(run),'n/a','n/a']
            if dumpIntensity:
                ll.extend('n/a')
            continue
        for lsdata in perrundata:
            lumilsnum=lsdata[0]
            cmslsnum=lsdata[1]
            beamstatus=lsdata[2]
            beamenergy=lsdata[3]
            ncollidingbx=lsdata[4]
            if not dumpIntensity:
                result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),beamstatus,'%.2f'%beamenergy,str(ncollidingbx)])
                continue
            allbxinfo=lsdata[5]
            allbxresult=[]
            for thisbxinfo in allbxinfo:
                thisbxresultStr='(n/a,n/a,n/a,n/a)'
                bxidx=thisbxinfo[0]
                b1=thisbxinfo[1]
                b2=thisbxinfo[2]
                thisbxresultStr=','.join(['%d'%bxidx,'%.3e'%b1,'%.3e'%b2])
                allbxresult.append(thisbxresultStr)
            allbxresultStr=' '.join(allbxresult)
            result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),beamstatus,'%.2f'%beamenergy,str(ncollidingbx),allbxresultStr])

    if not toFile:
        labels=[('Run','LS','beamstatus','egev','ncollidingbx')]
        if dumpIntensity:
            labels=[('Run','LS','beamstatus','egev','ncollidingbx','(bxidx,b1,b2)')]
        print ' ==  = '
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                                   prefix = '| ', postfix = ' |', justify = 'left',
                                   delim = ' | ', wrapfunc = lambda x: wrap_onspace(x,25) )
    else:
        fieldnames=['Run','LS','beamstatus','egev','ncollidingbx']
        if dumpIntensity:
            fieldnames.append('(bxidx,b1,b2)')
        filename=toFile
        dumptocsv(fieldnames,result,filename)

if __name__ == "__main__":
    toScreenHeader('lumiCalc2.py','V04-00-00','v0','pp8TeV')

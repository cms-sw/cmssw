import os,sys
from RecoLuminosity.LumiDB import tablePrinter, csvReporter,CommonUtil
from RecoLuminosity.LumiDB.wordWrappers import wrap_always, wrap_onspace, wrap_onspace_strict
def toScreenNorm(normdata):
    result=[]
    labels=[('Name','amode','E(GeV)','Norm')]
    print ' ==  = '
    for name,thisnorm in normdata.items():
        amodetag=str(thisnorm[0])
        normval='%.2f'%thisnorm[1]
        egev='%.0f'%thisnorm[2]
        result.append([name,amodetag,egev,normval])
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,prefix = '| ', postfix = ' |', justify = 'left',delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) ) 

def toScreenTotDelivered(lumidata,resultlines,scalefactor,isverbose):
    '''
    inputs:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),calibratedlumierror(6),(bxidx,bxvalues,bxerrs)(7),(bxidx,b1intensities,b2intensities)(8),fillnum)(9)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    totOldDeliveredLS=0
    totOldDelivered=0.0
    for r in resultlines:
        dl=0.0
        if(r[2]!='n/a'):            
            dl=float(r[2])#in /ub because it comes from file!
            (rr,lumiu)=CommonUtil.guessUnit(dl)
            r[2]='%.3f'%(rr)+' ('+lumiu+')'
        sls=0
        if(r[1]!='n/a'):
            sls=int(r[1])
        totOldDeliveredLS+=sls
        totOldDelivered+=dl
        if(r[4]!='n/a'):
            egv=float(r[4])
            r[4]='%.1f'%egv
        result.append(r)
    totls=0
    totdelivered=0.0
    totaltable=[]
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if not lsdata:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a'])
            if isverbose:
                result.extend(['n/a'])
            continue
        nls=0
        if len(lsdata):
            nls=len(lsdata)
        totls+=nls
        totlumi=sum([x[5] for x in lsdata])
        totdelivered+=totlumi
        (totlumival,lumiunit)=CommonUtil.guessUnit(totlumi)
        beamenergyPerLS=[float(x[4]) for x in lsdata if x[3]=='STABLE BEAMS']
        avgbeamenergy=0.0
        if len(beamenergyPerLS):
            avgbeamenergy=sum(beamenergyPerLS)/len(beamenergyPerLS)
        runstarttime='n/a'
        if nls!=0:
            runstarttime=lsdata[0][2]
            runstarttime=runstarttime.strftime("%m/%d/%y %H:%M:%S")
            fillnum=0
            if lsdata[0][9]:
                fillnum=lsdata[0][9]
        if isverbose:
            selectedls='n/a'
            if nls:
                selectedls=[(x[0],x[1]) for x in lsdata]
            result.append([str(run)+':'+str(fillnum),str(nls),'%.3f'%(totlumival*scalefactor)+' ('+lumiunit+')',runstarttime,'%.1f'%(avgbeamenergy), str(selectedls)])
        else:
            if runstarttime!='n/a':
                result.append([str(run)+':'+str(fillnum),str(nls),'%.3f'%(totlumival*scalefactor)+' ('+lumiunit+')',runstarttime,'%.1f'%(avgbeamenergy)])
            else:
                result.append([str(run)+':'+str(fillnum),str(nls),'%.3f'%(totlumival*scalefactor)+' ('+lumiunit+')','n/a','%.1f'%(avgbeamenergy)])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    #print 'sortedresult ',sortedresult
    print ' ==  = '
    if isverbose:
        labels = [('Run:Fill', 'N_LS', 'Delivered','UTCTime','E(GeV)','Selected LS')]
        print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    else:
        labels = [('Run:Fill', 'N_LS', 'Delivered','UTCTime','E(GeV)')]
        print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,40) )
    print ' ==  =  Total : '
    #if (totdelivered+totOldDelivered)!=0:
    (totalDeliveredVal,totalDeliveredUni)=CommonUtil.guessUnit(totdelivered+totOldDelivered)
    totrowlabels = [('Delivered LS','Delivered('+totalDeliveredUni+')')]
    totaltable.append([str(totls+totOldDeliveredLS),'%.3f'%(totalDeliveredVal*scalefactor)])
    print tablePrinter.indent (totrowlabels+totaltable, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))
    
def toCSVTotDelivered(lumidata,filename,resultlines,scalefactor,isverbose):
    '''
    input:  {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),calibratedlumierror(6),(bxidx,bxvalues,bxerrs)(7),(bxidx,b1intensities,b2intensities)(8),fillnum(9)]}
    '''
    result=[]
    fieldnames = ['Run:Fill', 'N_LS', 'Delivered(/ub)','UTCTime','E(GeV)']
    if isverbose:
        fieldnames.append('Selected LS')
    for rline in resultlines:
        result.append(rline)
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if not lsdata:
            result.append([run,'n/a','n/a','n/a','n/a'])
            if isverbose:
                result.extend(['n/a'])
            continue
        nls=len(lsdata)
        fillnum=0
        if lsdata[0][9]:
            fillnum=lsdata[0][9]
        totlumival=sum([x[5] for x in lsdata])
        beamenergyPerLS=[float(x[4]) for x in lsdata if x[3]=='STABLE BEAMS']
        avgbeamenergy=0.0
        if len(beamenergyPerLS):
            avgbeamenergy=sum(beamenergyPerLS)/len(beamenergyPerLS)
        runstarttime='n/a'
        if nls!=0:
            runstarttime=lsdata[0][2]
            runstarttime=runstarttime.strftime("%m/%d/%y %H:%M:%S")
        if isverbose:
            selectedls='n/a'
            if nls:
                selectedls=[(x[0],x[1]) for x in lsdata]
            result.append([str(run)+':'+str(fillnum),nls,totlumival*scalefactor,runstarttime,avgbeamenergy, str(selectedls)])
        else:
            result.append([str(run)+':'+str(fillnum),nls,totlumival*scalefactor,runstarttime,avgbeamenergy])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    r=None
    assert(filename)
    if filename.upper()=='STDOUT':
        r=sys.stdout
        r.write(','.join(fieldnames)+'\n')
        for l in sortedresult:
            r.write(str(l)+'\n')
    else:
        r=csvReporter.csvReporter(filename)
        r.writeRow(fieldnames)
        r.writeRows(sortedresult)

def toScreenOverview(lumidata,resultlines,scalefactor,isverbose):
    '''
    input:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),(bxidx,b1intensities,b2intensities)(9),fillnum(10)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    labels = [('Run:Fill', 'Delivered LS', 'Delivered','Selected LS','Recorded')]
    totOldDeliveredLS=0
    totOldSelectedLS=0
    totOldDelivered=0.0
    totOldRecorded=0.0
    
    totaltable=[]
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0

    for r in resultlines:
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
            result.append([str(run),'n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if lsdata[0][10]:
            fillnum=lsdata[0][10]
        nls=len(lsdata)
        deliveredData=[x[5] for x in lsdata]
        totdelivered=sum(deliveredData)

        totalDelivered+=totdelivered
        totalDeliveredLS+=len(deliveredData)
        (totdeliveredlumi,deliveredlumiunit)=CommonUtil.guessUnit(totdelivered)
        recordedData=[x[6] for x in lsdata if x[6] is not None]
        totrecorded=sum(recordedData)
        totalRecorded+=totrecorded
        (totrecordedlumi,recordedlumiunit)=CommonUtil.guessUnit(totrecorded)
        #print 'x[1] ',[x[1] for x in lsdata]
        selectedcmsls=[x[1] for x in lsdata if x[1]!=0]
        #print 'selectedcmsls ',selectedcmsls
        totalSelectedLS+=len(selectedcmsls)
        if len(selectedcmsls)==0:
            selectedlsStr='n/a'
        else:
            selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        result.append([str(run)+':'+str(fillnum),str(nls),'%.3f'%(totdeliveredlumi*scalefactor)+' ('+deliveredlumiunit+')',selectedlsStr,'%.3f'%(totrecordedlumi*scalefactor)+' ('+recordedlumiunit+')'])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))    
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
    
def toCSVOverview(lumidata,filename,resultlines,scalefactor,isverbose):
    '''
    input:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),(bxidx,b1intensities,b2intensities)(9),fillnum(10)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    fieldnames = ['Run:Fill', 'DeliveredLS', 'Delivered(/ub)','SelectedLS','Recorded(/ub)']
    r=csvReporter.csvReporter(filename)
    for rline in resultlines:
        result.append(rline)
        
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if not lsdata:
            result.append([run,'n/a','n/a','n/a','n/a'])
            continue
        nls=len(lsdata)
        fillnum=0
        if lsdata[0][10]:
            fillnum=lsdata[0][10]
        deliveredData=[x[5] for x in lsdata]
        recordedData=[x[6] for x in lsdata if x[6] is not None]
        totdeliveredlumi=0.0
        totrecordedlumi=0.0
        if len(deliveredData)!=0:
            totdeliveredlumi=sum(deliveredData)
        if len(recordedData)!=0:
            totrecordedlumi=sum(recordedData)
        selectedcmsls=[x[1] for x in lsdata if x[1]!=0]
        if len(selectedcmsls)==0:
            selectedlsStr='n/a'
        else:
            selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        result.append([str(run)+':'+str(fillnum),nls,totdeliveredlumi*scalefactor,selectedlsStr,totrecordedlumi*scalefactor])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    
    r=None
    assert(filename)
    if filename.upper()=='STDOUT':
        r=sys.stdout
        r.write(','.join(fieldnames)+'\n')
        for l in sortedresult:
            r.write(str(l)+'\n')
    else:
        r=csvReporter.csvReporter(filename)
        r.writeRow(fieldnames)
        r.writeRows(sortedresult)
def toScreenLumiByLS(lumidata,resultlines,scalefactor,isverbose):
    '''
    input:
    lumidata {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),(bxidx,b1intensities,b2intensities)(9),fillnum(10)]}
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
    for rline in resultlines:
        myls=rline[1]
        if myls!='n/a':
            [luls,cmls]=myls.split(':')
            totOldDeliveredLS+=1
            if cmls!='0':
                totOldSelectedLS+=1
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
        rundata=lumidata[run]
        if not rundata:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata[0][10]:
            fillnum=rundata[0][10]
        for lsdata in rundata:
            lumilsnum=lsdata[0]
            cmslsnum=lsdata[1]#triggered ls
            ts=lsdata[2]
            bs=lsdata[3]
            begev=lsdata[4]
            deliveredlumi=lsdata[5]            
            if deliveredlumi>maxlslumi: maxlslumi=deliveredlumi
            recordedlumi=lsdata[6]
            #if cmslsnum!=0:               
            result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),ts.strftime('%m/%d/%y %H:%M:%S'),bs,'%.1f'%begev,(deliveredlumi),(recordedlumi)])
            totalDelivered+=deliveredlumi
            totalRecorded+=recordedlumi
            totalDeliveredLS+=1
            if(cmslsnum!=0):
                totalSelectedLS+=1
    #guess ls lumi unit
    (lsunitstring,unitdenomitor)=CommonUtil.lumiUnitForPrint(maxlslumi*scalefactor)
    labels = [ ('Run:Fill','LS','UTCTime','Beam Status','E(GeV)','Delivered('+lsunitstring+')','Recorded('+lsunitstring+')') ]
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    perlsresult=[]
    for entry in sortedresult:
        delumi=entry[5]
        if delumi!='n/a':
            delumi='%.3f'%float(float(delumi*scalefactor)/float(unitdenomitor))        
        reclumi=entry[6]
        if reclumi!='n/a':
            reclumi='%.3f'%float(float(reclumi*scalefactor)/float(unitdenomitor))
        perlsresult.append([entry[0],entry[1],entry[2],entry[3],entry[4],delumi,reclumi])
    totdeliveredlumi=0.0
    deliveredlumiunit='/ub'
    #if (totalDelivered+totOldDelivered)!=0:
    (totdeliveredlumi,deliveredlumiunit)=CommonUtil.guessUnit((totalDelivered+totOldDelivered)*scalefactor)
    totrecordedlumi=0.0
    recordedlumiunit='/ub'
    #if (totalRecorded+totOldRecorded)!=0:
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

                  
def toCSVLumiByLS(lumidata,filename,resultlines,scalefactor,isverbose):
    result=[]
    fieldnames=['Run:Fill','LS','UTCTime','Beam Status','E(GeV)','Delivered(/ub)','Recorded(/ub)']
    for rline in resultlines:
        result.append(rline)
        
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([run,'n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata and rundata[0][10]:
            fillnum=rundata[0][10]
        for lsdata in rundata:
            lumilsnum=lsdata[0]
            cmslsnum=lsdata[1]
            ts=lsdata[2]
            bs=lsdata[3]
            begev=lsdata[4]
            deliveredlumi=lsdata[5]
            recordedlumi=lsdata[6]
            #if cmslsnum!=0:
            result.append([str(run)+':'+str(fillnum),str(lumilsnum)+':'+str(cmslsnum),ts.strftime('%m/%d/%y %H:%M:%S'),bs,begev,deliveredlumi*scalefactor,recordedlumi*scalefactor])
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
    assert(filename)
    if filename.upper()=='STDOUT':
        r=sys.stdout
        r.write(','.join(fieldnames)+'\n')
        for l in sortedresult:
            r.write(str(l)+'\n')
    else:
        r=csvReporter.csvReporter(filename)
        r.writeRow(fieldnames)
        r.writeRows(sortedresult)

def toScreenLSEffective(lumidata,resultlines,scalefactor,isverbose):
    '''
    input:  {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]}(8),bxdata(9),beamdata(10),fillnum(11)]}
    '''
    result=[]#[run,ls,hltpath,l1bitname,hltpresc,l1presc,efflumi]
    totalrow=[]
    totSelectedLS=0
    totRecorded=0.0
    totEffective=0

    totOldSelectedLS=0
    totOldRecorded=0.0
    totOldEffective=0.0

    maxlslumi = 0.0
    for rline in resultlines:
        myls=rline[1]
        if myls!='n/a':
            totOldSelectedLS+=1
        myrecorded=rline[6]
        if myrecorded!='n/a':
            myrecorded=float(rline[6])
            if myrecorded>maxlslumi:maxlslumi=myrecorded
            totOldRecorded+=myrecorded
            rline[6]=myrecorded
        myeff=rline[7]
        if myeff!='n/a':
            myeff=float(rline[7])
            totOldEffective+=myeff
            rline[7]=myeff
        result.append(rline)
        
    totrecordedlumi=0.0
    totefflumi=0.0
   
    for run in lumidata.keys():#loop over runs
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata[0][11]:
            fillnum=rundata[0][11]
        for lsdata in rundata:
            efflumiDict=lsdata[8]# this ls has no such path?
            if not efflumiDict:
                continue
            cmslsnum=lsdata[1]
            recorded=lsdata[6]
            totSelectedLS+=1
            if not recorded:
                recorded=0.0
            if recorded>maxlslumi:maxlslumi=recorded
            totRecorded+=recorded
            for hltpathname in sorted(efflumiDict):
                pathdata=efflumiDict[hltpathname]
                l1name=pathdata[0]
                if l1name is None:
                    l1name='n/a'
                else:
                    l1name=l1name.replace('"','')
                l1prescale=pathdata[1]
                hltprescale=pathdata[2]
                lumival=pathdata[3]
                if lumival is not None:
                    result.append([str(run)+':'+str(fillnum),str(cmslsnum),hltpathname,l1name,str(hltprescale),str(l1prescale),(recorded),(lumival)])
                    totEffective+=lumival
                else:
                    result.append([str(run)+':'+str(fillnum),str(cmslsnum),hltpathname,l1name,str(hltprescale),str(l1prescale),(recorded),'n/a'])
    (totrecordedlumi,recordedlumiunit)=CommonUtil.guessUnit((totRecorded+totOldRecorded)*scalefactor)
    (totefflumi,efflumiunit)=CommonUtil.guessUnit((totEffective+totOldEffective)*scalefactor)
    #guess ls lumi unit
    (lsunitstring,unitdenomitor)=CommonUtil.lumiUnitForPrint(maxlslumi*scalefactor)
    labels = [('Run:Fill','LS','HLTpath','L1bit','HLTpresc','L1presc','Recorded('+lsunitstring+')','Effective('+lsunitstring+')')]
    sortedresult=sorted(result,key=lambda x : int(str(x[0]).split(':')[0]))
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
    totalrow.append([str(totSelectedLS+totOldSelectedLS),'%.3f'%(totrecordedlumi),'%.3f'%(totefflumi)])
    lastrowlabels = [ ('Selected LS','Recorded('+recordedlumiunit+')','Effective('+efflumiunit+')')]
    print ' ==  =  Total : '
    print tablePrinter.indent (lastrowlabels+totalrow, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))    

def toCSVLSEffective(lumidata,filename,resultlines,scalefactor,isverbose):
    '''
    input:  {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]}(8),bxdata(9),beamdata(10),fillnum(11)]}
    '''
    result=[]#[run,ls,hltpath,l1bitname,hltpresc,l1presc,efflumi]
    for rline in resultlines:
        result.append(rline)
         
    for run in sorted(lumidata):#loop over runs
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata[0][11]:
            fillnum=rundata[0][11]
        for lsdata in rundata:
            efflumiDict=lsdata[8]# this ls has no such path?
            if not efflumiDict:
                continue
            cmslsnum=lsdata[1]
            recorded=lsdata[6]
            if not recorded:
                recorded=0.0
            for hltpathname in sorted(efflumiDict):
                pathdata=efflumiDict[hltpathname]
                l1name=pathdata[0]
                if l1name is None:
                    l1name='n/a'
                else:
                    l1name=l1name.replace('"','')
                l1prescale=pathdata[1]
                hltprescale=pathdata[2]
                lumival=pathdata[3]
                if lumival is not None:
                    result.append([str(run)+':'+str(fillnum),cmslsnum,hltpathname,l1name,hltprescale,l1prescale,recorded*scalefactor,lumival*scalefactor])
                else:
                    result.append([str(run)+':'+str(fillnum),cmslsnum,hltpathname,l1name,hltprescale,l1prescale,recorded*scalefactor,'n/a'])
    fieldnames = ['Run:Fill','LS','HLTpath','L1bit','HLTpresc','L1presc','Recorded(/ub)','Effective(/ub)']
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

def toScreenTotEffective(lumidata,resultlines,scalefactor,isverbose):
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
    for rline in resultlines:
        myrun=rline[0]
        myls=rline[1]
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
    for run in sorted(lumidata):#loop over runs
        hprescdict={}
        lprescdict={}
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata[0][11]:
            fillnum=rundata[0][11]
        selectedcmsls=[x[1] for x in rundata if x[1]!=0]
        totefflumiDict={}
        totrecorded=0.0
        toteffective=0.0
        pathmap={}#{hltpathname:1lname}
        for lsdata in rundata:
            cmslsnum=lsdata[1]
            efflumiDict=lsdata[8]# this ls has no such path?
            recordedlumi=lsdata[6]
            totrecorded+=recordedlumi
            if not efflumiDict:
                if cmslsnum in selectedcmsls:
                    selectedcmsls.remove(cmslsnum)
                continue
            for hltpathname,pathdata in efflumiDict.items():
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
       
        for name in sorted(totefflumiDict):
            lname=pathmap[name]
            totrecordedinrun=recordedPerpathPerrun[name][run]
            hprescs=list(set(hprescdict[name]))
            hprescStr='('+','.join(['%d'%(x) for x in hprescs])+')'
            (totrecval,totrecunit)=CommonUtil.guessUnit(totrecordedinrun*scalefactor)
            if lname=='n/a':
                result.append([str(run)+':'+str(fillnum),selectedlsStr,'%.3f'%(totrecval)+'('+totrecunit+')',name+hprescStr,lname,'n/a'])
            else:
                (efflumival,efflumiunit)=CommonUtil.guessUnit(totefflumiDict[name]*scalefactor)
                lprescs=list(set(lprescdict[lname]))
                lprescStr='('+','.join(['%d'%(x) for x in lprescs])+')'
                cleanlname=lname.replace('"','')
                result.append([str(run)+':'+str(fillnum),selectedlsStr,'%.3f'%(totrecval)+'('+totrecunit+')',name+hprescStr,cleanlname+lprescStr,'%.3f'%(efflumival)+'('+efflumiunit+')'])
    labels = [('Run:Fill','SelectedLS','Recorded','HLTpath(Presc)','L1bit(Presc)','Effective')]
    print ' ==  = '
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
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
    print tablePrinter.indent (lastrowlabels+totresult, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    
def toCSVTotEffective(lumidata,filename,resultlines,scalefactor,isverbose):
    '''
    input:  {run:[lumilsnum(0),triggeredls(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]}(8),bxdata(9),beamdata(10),fillnum(11)]}
    screen Run,SelectedLS,Recorded,HLTPath,L1Bit,Effective
    '''
    result=[]#[run,selectedlsStr,recorded,hltpath,l1bitname,efflumi]
    totdict={}#{hltpath:[nls,toteff]}
    selectedcmsls=[]
    recordedPerpathPerrun={}#{path:{run:recorded}}
    selectedPerpathPerrun={}#{path:{run:totselected}}
    for rline in resultlines:
        result.append(rline)
    for run in sorted(lumidata):#loop over runs
        hprescdict={}
        lprescdict={}
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata[0][11]:
            fillnum=rundata[0][11]
        selectedcmsls=[x[1] for x in rundata if x[1]!=0]
        totefflumiDict={}
        totrecorded=0.0
        toteffective=0.0
        pathmap={}#{hltpathname:1lname}
        for lsdata in rundata:
            cmslsnum=lsdata[1]
            efflumiDict=lsdata[8]# this ls has no such path?
            recordedlumi=lsdata[6]
            totrecorded+=recordedlumi
            if not efflumiDict:
                if cmslsnum in selectedcmsls:
                    selectedcmsls.remove(cmslsnum)
                continue
            for hltpathname,pathdata in efflumiDict.items():
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
                    #recordedlumi=lsdata[6]
                    #totrecorded+=recordedlumi
                    if not hprescdict.has_key(hltpathname):
                        hprescdict[hltpathname]=[]
                    hprescdict[hltpathname].append(hltpresc)
                    if not lprescdict.has_key(l1name):
                        lprescdict[l1name]=[]
                    lprescdict[l1name].append(l1presc)
                    totdict[hltpathname][0]+=1
                    if lumival:
                        totdict[hltpathname][1]+=lumival
                        totefflumiDict[hltpathname]+=lumival
                        pathmap[hltpathname]=l1name.replace('\"','')
                    recordedPerpathPerrun[hltpathname][run]=totrecorded
                    selectedPerpathPerrun[hltpathname][run]=len(selectedcmsls)
        if len(selectedcmsls)==0:
            selectedlsStr='n/a'
        else:
            selectedlsStr= CommonUtil.splitlistToRangeString(selectedcmsls)
            
        for name in sorted(totefflumiDict):
            lname=pathmap[name]
            if lname=='n/a':
                continue
            totrecordedinrun=recordedPerpathPerrun[name][run]
            hprescs=list(set(hprescdict[name]))
            lprescs=list(set(lprescdict['"'+lname+'"']))
            hprescStr='('+','.join(['%d'%(x) for x in hprescs])+')'
            lprescStr='('+','.join(['%d'%(x) for x in lprescs])+')'
            result.append([str(run)+':'+str(fillnum),selectedlsStr,totrecordedinrun*scalefactor,name+hprescStr,lname+lprescStr,totefflumiDict[name]*scalefactor])
    fieldnames=['Run:Fill','SelectedLS','Recorded','HLTpath(Presc)','L1bit(Presc)','Effective(/ub)']
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
        
def toCSVLumiByLSXing(lumidata,scalefactor,filename):
    '''
    input:{run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),bxdata(8),beamdata(9),fillnum(10)]}
    output:
    fieldnames=['Run:Fill','CMSLS','Delivered(/ub)','Recorded(/ub)','BX']
    '''
    result=[]
    assert(filename)
    fieldnames=['run:fill','ls','UTCTime','delivered(/ub)','recorded(/ub)','bx']
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([run,'n/a','n/a','n/a','n/a','n/a'])
            continue
        fillnum=0
        if rundata[0][10]:
            fillnum=rundata[0][10]
        for lsdata in rundata:
            cmslsnum=lsdata[1]
            ts=lsdata[2]
            if cmslsnum==0:
                continue
            deliveredlumi=lsdata[5]
            recordedlumi=lsdata[6]
            (bxidxlist,bxvaluelist,bxerrorlist)=lsdata[8]
            bxresult=[]
            if bxidxlist and bxvaluelist:
                bxinfo=CommonUtil.transposed([bxidxlist,bxvaluelist])
                bxresult=CommonUtil.flatten([str(run)+':'+str(fillnum),cmslsnum,ts.strftime('%m/%d/%y %H:%M:%S'),deliveredlumi*scalefactor,recordedlumi*scalefactor,bxinfo])
                result.append(bxresult)
            else:
                result.append([str(run)+':'+str(fillnum),cmslsnum,ts.strftime('%m/%d/%y %H:%M:%S'),deliveredlumi*scalefactor,recordedlumi*scalefactor])
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
    
def toScreenLSTrg(trgdata,iresults=[],isverbose=False):
    '''
    input:{run:[[cmslsnum,deadfrac,deadtimecount,bitzero_count,bitzero_prescale,[(name,count,presc),]],..]
    '''
    result=[]
    for r in iresults:
        result.append(r)
    for run in trgdata.keys():
        if trgdata[run] is None:
            ll=[str(run),'n/a','n/a','n/a']
            if isverbose:
                ll.append('n/a')
            result.append(ll)
            continue
        perrundata=trgdata[run]
        deadfrac=0.0
        bitdataStr='n/a'
        for lsdata in perrundata:
            cmslsnum=lsdata[0]
            deadfrac=lsdata[1]
            deadcount=lsdata[2]
            bitdata=lsdata[5]# already sorted by name
            flatbitdata=["("+x[0]+',%d'%x[1]+',%d'%x[2]+")" for x in bitdata if x[0]!='False']
            bitdataStr=', '.join(flatbitdata)
            #print 'bitdataStr ',bitdataStr
            if isverbose:
                result.append([str(run),str(cmslsnum),'%.4f'%(deadfrac),'%d'%deadcount,bitdataStr])
            else:
                result.append([str(run),str(cmslsnum),'%.4f'%(deadfrac),'%d'%deadcount])
    print ' ==  = '
    if isverbose:
        labels = [('Run', 'LS', 'dfrac','dcount','(bit,count,presc)')]
    else:
        labels = [('Run', 'LS', 'dfrac','dcount')]
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'left',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,70) )
    
def toCSVLSTrg(trgdata,filename,iresults=[],isverbose=False):
    '''
    input:{run:[[cmslsnum,deadfrac,deadtimecount,bitzero_count,bitzero_prescale,[(name,count,presc),]],..]
    '''
    result=[]
    fieldnames=['Run','LS','dfrac','dcount','bit,cout,presc']
    for rline in iresults:
        result.append(rline)
    for run in sorted(trgdata):
        rundata=trgdata[run]
        if rundata is None:
            ll=[run,'n/a','n/a','n/a',]
            if isverbose:
                ll.append('n/a')
            result.append(ll)
            continue
        deadfrac=0.0
        bitdataStr='n/a'
        for lsdata in rundata:
            cmslsnum=lsdata[0]
            deadfrac=lsdata[1]
            dcount=lsdata[2]
            bitdata=lsdata[5]
            flatbitdata=[x[0]+',%d'%x[1]+',%d'%x[2] for x in bitdata if x[0]!='False']
            bitdataStr=';'.join(flatbitdata)
            if isverbose:                
                result.append([run,cmslsnum,deadfrac,dcount,bitdataStr])
            else:
                result.append([run,cmslsnum,deadfrac,dcount])
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
    
def toScreenConfTrg(trgconfdata,iresults=[],isverbose=False):
    '''
    input:{run:[datasource,normbitname,[allbits]]}
    '''
    if isverbose:
        labels=[('Run','source','bitnames','normbit')]
    else:
        labels=[('Run','source','bitnames')]

    result=[]
    for r in iresults:
        result.append(r)
    for  run in sorted(trgconfdata):
        if trgconfdata[run] is None:
            ll=[str(run),'n/a','n/a']
            if isverbose:
                ll.append('n/a')
            result.append(ll)
            continue
        source=trgconfdata[run][0]
        source=source.split('/')[-1]
        normbit=trgconfdata[run][1]
        allbits=trgconfdata[run][2]
        bitnames=', '.join(allbits)
        if isverbose:
            result.append([str(run),source,bitnames,normbit])
        else:
            result.append([str(run),source,bitnames])

    print ' ==  = '
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'left',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace_strict(x,60) )

def toCSVConfTrg(trgconfdata,filename,iresults=[],isverbose=False):
    '''
    input {run:[datasource,normbitname,[allbits]]}
    '''
    result=[]
    fieldnames=['Run','source','bitnames']
    if isverbose:
        fieldnames.append('normbit')
    for rline in iresults:
        result.append(rline)
    for run in sorted(trgconfdata):
        rundata=trgconfdata[run]
        if rundata is None:
            ll=[run,'n/a','n/a']
            if isverbose:
                ll.append('n/a')
            result.append(ll)            
            continue
        datasource=rundata[0]
        if datasource:
            datasource=datasource.split('/')[-1]
        normbit=rundata[1]
        bitdata=rundata[2]        
        bitnames=','.join(bitdata)
        if isverbose:
            result.append([run,datasource,bitnames,normbit])
        else:
            result.append([run,datasource,bitnames])
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

def toScreenLSHlt(hltdata,iresults=[],isverbose=False):
    '''
    input:{runnumber:[(cmslsnum,[(hltpath,hltprescale,l1pass,hltaccept),...]),(cmslsnum,[])})}
    '''
    result=[]
    for r in iresults:
        result.append(r)
    for run in hltdata.keys():
        if hltdata[run] is None:            
            ll=[str(run),'n/a','n/a']
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
                thispathresult.append('%d'%thispathpresc)
                if isverbose:
                    if thispathinfo[2] :
                        thisl1pass=thispathinfo[2]
                        thispathresult.append('%d'%thisl1pass)
                    else:
                        thispathresult.append('n/a')
                    if thispathinfo[3]:
                        thishltaccept=thispathinfo[3]
                        thispathresult.append('%d'%thishltaccept)
                    else:
                        thispathresult.append('n/a')
                thispathresultStr='('+','.join(thispathresult)+')'
                allpathresult.append(thispathresultStr)
            result.append([str(run),str(cmslsnum),', '.join(allpathresult)])
    print ' ==  = '
    if isverbose:
        labels = [('Run', 'LS', '(hltpath,presc)')]
    else:
        labels = [('Run', 'LS', '(hltpath,presc,l1pass,hltaccept)')]
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'left',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,70) )
    
def toCSVLSHlt(hltdata,filename,iresults=None,isverbose=False):
    '''
    input:{runnumber:[(cmslsnum,[(hltpath,hltprescale,l1pass,hltaccept),...]),(cmslsnum,[])})}
    '''
    result=[]
    fieldnames=['Run','LS','hltpath,hltprescale']
    if isverbose:
        fieldnames[-1]+=',l1pass,hltaccept'
    for rline in iresults:
        result.append(rline)
    for run in sorted(hltdata):
        lsdata=hltdata[run]
        if lsdata is None:
            result.append([run,'n/a','n/a'])
            continue
        for thislsdata in lsdata:
            cmslsnum=thislsdata[0]
            bitsdata=thislsdata[1]
            allbitsresult=[]
            for mybit in bitsdata:
                hltpath=mybit[0]
                if not hltpath: continue
                hltprescale=mybit[1]
                if hltprescale is None:
                    hltprescale='n/a'
                else:
                    hltprescale='%d'%hltprescale
                if isverbose:
                    l1pass=mybit[2]
                    if l1pass is None:
                        l1pass='n/a'
                    else:
                        l1pass='%d'%l1pass
                    hltaccept=mybit[3]
                    if hltaccept is None:
                        hltaccept='n/a'
                    else:
                        hltaccept='%d'%hltaccept
                    mybitStr=','.join([hltpath,hltprescale,l1pass,hltaccept])
                else:
                    mybitStr=','.join([hltpath,hltprescale])
                allbitsresult.append(mybitStr)
            allbitsresult=';'.join(allbitsresult)
            result.append([run,cmslsnum,allbitsresult])
            
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
    
def toScreenConfHlt(hltconfdata,iresults=[],isverbose=False):
    '''
    input : {runnumber,[(hltpath,l1seedexpr,l1bitname),...]}
    '''
    labels=[('Run','hltpath','l1seedexpr','l1bit')]
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
            thisseed=''.join(thisseed.split(' '))
            thisseed=' '.join([thisseed[i:i+25] for i in range(0,len(thisseed),25)])
            thisbit=thispathinfo[2]
            if not thisbit:
                thisbit='n/a'
            else:
                thisbit=' '.join([thisbit[i:i+25] for i in range(0,len(thisbit),25)])
            result.append([str(run),thispath,thisseed,thisbit])
    print ' ==  = '
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'left',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace(x,25) )


def toCSVConfHlt(hltconfdata,filename,iresults=[],isverbose=False):
    '''
    input:{runnumber,[(hltpath,l1seedexpr,l1bitname),...]}
    '''
    result=[]
    for rline in iresults:
        result.append(rline)
    for run in sorted(hltconfdata):
        pathdata=hltconfdata[run]
        if pathdata is None:
            result.append([str(run),'n/a','n/a','n/a'])
            continue
        for thispathinfo in pathdata:
            thispath=thispathinfo[0]
            thisseed=thispathinfo[1]
            thisbit=thispathinfo[2]
            if not thisbit:
                thisbit='n/a'
            result.append([str(run),thispath,thisseed,thisbit])
    fieldnames=['Run','hltpath','l1seedexpr','l1bit']
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
        
def toScreenLSBeam(beamdata,iresults=[],dumpIntensity=False,isverbose=False):
    '''
    input: {run:[(lumilsnum(0),cmslsnum(1),beamstatus(2),beamenergy(3),beaminfolist(4)),..]}
    beaminfolist:[(bxidx,b1,b2)]
    '''
    labels=[('Run','LS','beamstatus','egev')]
    if dumpIntensity:
        labels=[('Run','LS','beamstatus','egev','(bxidx,b1,b2)')]
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
            if not dumpIntensity:
                result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),beamstatus,'%.2f'%beamenergy])
                continue
            allbxinfo=lsdata[4]
            allbxresult=[]
            for thisbxinfo in allbxinfo:
                thisbxresultStr='(n/a,n/a,n/a)'
                bxidx=thisbxinfo[0]
                b1=thisbxinfo[1]
                b2=thisbxinfo[2]
                thisbxresultStr=','.join(['%d'%bxidx,'%.3e'%b1,'%.3e'%b2])
                allbxresult.append(thisbxresultStr)
            allbxresultStr=' '.join(allbxresult)
            result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),beamstatus,'%.2f'%beamenergy,allbxresultStr])
    print ' ==  = '
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'left',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace(x,25) )

def toCSVLSBeam(beamdata,filename,resultlines,dumpIntensity=False,isverbose=False):
    '''
    input: {run:[(lumilsnum(0),cmslsnum(1),beamstatus(2),beamenergy(3),beaminfolist(4)),..]}
    beaminfolist:[(bxidx,b1,b2)]
    '''
    result=[]
    fieldnames=['Run','LS','beamstatus','egev']
    if dumpIntensity:
        fieldnames.append('(bxidx,b1,b2)')
    for rline in resultlines:
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
            if not dumpIntensity:
                result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),beamstatus,'%.2f'%beamenergy])
                continue
            allbxinfo=lsdata[4]
            #print 'allbxinfo ',allbxinfo
            allbxresult=[]
            for thisbxinfo in allbxinfo:
                thisbxresultStr='(n/a,n/a,n/a)'
                bxidx=thisbxinfo[0]
                b1=thisbxinfo[1]
                b2=thisbxinfo[2]
                thisbxresultStr=','.join(['%d'%bxidx,'%.3e'%b1,'%.3e'%b2])
                allbxresult.append(thisbxresultStr)
            allbxresultStr=' '.join(allbxresult)
            result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),beamstatus,'%.2f'%beamenergy,allbxresultStr])
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


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
    lumidata {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
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
        if lsdata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a'])
            if isverbose:
                result.extend(['n/a'])
            continue
        nls=len(lsdata)
        totls+=nls
        totlumi=sum([x[5] for x in lsdata])
        totdelivered+=totlumi
        (totlumival,lumiunit)=CommonUtil.guessUnit(totlumi)
        beamenergyPerLS=[float(x[4]) for x in lsdata]
        avgbeamenergy=0.0
        if len(beamenergyPerLS):
            avgbeamenergy=sum(beamenergyPerLS)/len(beamenergyPerLS)
        runstarttime=lsdata[0][2]
        if isverbose:
            selectedls=[(x[0],x[1]) for x in lsdata]
            result.append([str(run),str(nls),'%.3f'%(totlumival*scalefactor)+' ('+lumiunit+')',runstarttime.strftime("%m/%d/%y %H:%M:%S"),'%.1f'%(avgbeamenergy), str(selectedls)])
        else:
            result.append([str(run),str(nls),'%.3f'%(totlumival*scalefactor)+' ('+lumiunit+')',runstarttime.strftime("%m/%d/%y %H:%M:%S"),'%.1f'%(avgbeamenergy)])
    sortedresult=sorted(result,key=lambda x : int(x[0]))
    #print 'sortedresult ',sortedresult
    print ' ==  = '
    if isverbose:
        labels = [('Run', 'Total LS', 'Delivered','Start Time','E(GeV)','Selected LS')]
        print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    else:
        labels = [('Run', 'Total LS', 'Delivered','Start Time','E(GeV)')]
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
    input:  {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    '''
    result=[]
    fieldnames = ['Run', 'Total LS', 'Delivered(/ub)','UTCTime','E(GeV)']
    if isverbose:
        fieldnames.append('Selected LS')
    for rline in resultlines:
        result.append(rline)
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if lsdata is None:
            result.append([run,'n/a','n/a','n/a','n/a'])
            if isverbose:
                result.extend(['n/a'])
            continue
        nls=len(lsdata)
        totlumival=sum([x[5] for x in lsdata])
        beamenergyPerLS=[float(x[4]) for x in lsdata]
        avgbeamenergy=0.0
        if len(beamenergyPerLS):
            avgbeamenergy=sum(beamenergyPerLS)/len(beamenergyPerLS)
        runstarttime=lsdata[0][2]
        if isverbose:
            selectedls=[(x[0],x[1]) for x in lsdata]
            result.append([run,nls,totlumival*scalefactor,runstarttime.strftime("%m/%d/%y %H:%M:%S"),avgbeamenergy, str(selectedls)])
        else:
            result.append([run,nls,totlumival*scalefactor,runstarttime.strftime("%m/%d/%y %H:%M:%S"),avgbeamenergy])
    sortedresult=sorted(result,key=lambda x : int(x[0]))
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
    lumidata {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    labels = [('Run', 'Delivered LS', 'Delivered','Selected LS','Recorded')]
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
        if lsdata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a'])
            continue
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
        result.append([str(run),str(nls),'%.3f'%(totdeliveredlumi*scalefactor)+' ('+deliveredlumiunit+')',selectedlsStr,'%.3f'%(totrecordedlumi*scalefactor)+' ('+recordedlumiunit+')'])
    sortedresult=sorted(result,key=lambda x : int(x[0]))    
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
    lumidata {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    fieldnames = ['Run', 'DeliveredLS', 'Delivered(/ub)','SelectedLS','Recorded(/ub)']
    r=csvReporter.csvReporter(filename)
    for rline in resultlines:
        result.append(rline)
        
    for run in lumidata.keys():
        lsdata=lumidata[run]
        if lsdata is None:
            result.append([run,'n/a','n/a','n/a','n/a'])
            continue
        nls=len(lsdata)
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
        result.append([run,nls,totdeliveredlumi*scalefactor,selectedlsStr,totrecordedlumi*scalefactor])
    sortedresult=sorted(result,key=lambda x : int(x[0]))
    
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
    lumidata {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    resultlines [[resultrow1],[resultrow2],...,] existing result row
    '''
    result=[]
    labels = [ ('Run','LS','UTCTime','Beam Status','E(GeV)','Delivered(/ub)','Recorded(/ub)')]
    totalrow = []
    
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0

    totOldDeliveredLS = 0
    totOldSelectedLS = 0
    totOldDelivered = 0.0
    totOldRecorded = 0.0

    for rline in resultlines:
        myls=rline[1]
        if myls!='n/a':
            [luls,cmls]=myls.split(':')
            totOldDeliveredLS+=1
            if cmls!='0':
                totOldSelectedLS+=1
        dl=0.0
        if rline[5]!='n/a':
            dl=float(rline[5])#delivered in /ub 
            rline[5]='%.2f'%(dl)
            totOldDelivered+=dl
        if rline[6]!='n/a':
           rl=float(rline[6])#recorded in /ub
           rline[6]='%.2f'%(rl)
           totOldRecorded+=rl
        result.append(rline)
        
    for run in lumidata.keys():
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a'])
            continue
        for lsdata in rundata:
            lumilsnum=lsdata[0]
            cmslsnum=lsdata[1]
            ts=lsdata[2]
            bs=lsdata[3]
            begev=lsdata[4]
            deliveredlumi=lsdata[5]
            recordedlumi=lsdata[6]
            result.append([str(run),str(lumilsnum)+':'+str(cmslsnum),ts.strftime('%m/%d/%y %H:%M:%S'),bs,'%.1f'%begev,'%.2f'%(deliveredlumi*scalefactor),'%.2f'%(recordedlumi*scalefactor)])
            totalDelivered+=deliveredlumi
            totalRecorded+=recordedlumi
            totalDeliveredLS+=1
            if(cmslsnum!=0):
                totalSelectedLS+=1
    totdeliveredlumi=0.0
    deliveredlumiunit='/ub'
    #if (totalDelivered+totOldDelivered)!=0:
    (totdeliveredlumi,deliveredlumiunit)=CommonUtil.guessUnit(totalDelivered+totOldDelivered)
    totrecordedlumi=0.0
    recordedlumiunit='/ub'
    #if (totalRecorded+totOldRecorded)!=0:
    (totrecordedlumi,recordedlumiunit)=CommonUtil.guessUnit(totalRecorded+totOldRecorded)
    lastrowlabels = [ ('Delivered LS','Selected LS', 'Delivered('+deliveredlumiunit+')', 'Recorded('+recordedlumiunit+')')]
    totalrow.append ([str(totalDeliveredLS+totOldDeliveredLS),str(totalSelectedLS+totOldSelectedLS),'%.3f'%(totdeliveredlumi*scalefactor),'%.3f'%(totrecordedlumi*scalefactor)])
    sortedresult=sorted(result,key=lambda x : int(x[0]))
    print ' ==  = '
    print tablePrinter.indent (labels+sortedresult, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace_strict (x, 22))
    print ' ==  =  Total : '
    print tablePrinter.indent (lastrowlabels+totalrow, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))    

                  
def toCSVLumiByLS(lumidata,filename,resultlines,scalefactor,isverbose):
    result=[]
    fieldnames=['Run','LS','UTCTime','Beam Status','E(GeV)','Delivered(/ub)','Recorded(/ub)']
    for rline in resultlines:
        result.append(rline)
        
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([run,'n/a','n/a','n/a','n/a','n/a'])
            continue
        for lsdata in rundata:
            lumilsnum=lsdata[0]
            cmslsnum=lsdata[1]
            ts=lsdata[2]
            bs=lsdata[3]
            begev=lsdata[4]
            deliveredlumi=lsdata[5]
            recordedlumi=lsdata[6]
            result.append([run,str(lumilsnum)+':'+str(cmslsnum),ts.strftime('%m/%d/%y %H:%M:%S'),bs,begev,deliveredlumi*scalefactor,recordedlumi*scalefactor])
    sortedresult=sorted(result,key=lambda x : int(x[0]))
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
    input:  {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]},bxdata,beamdata]}
    '''
    result=[]#[run,ls,hltpath,l1bitname,hltpresc,l1presc,efflumi]
    for run in sorted(lumidata):#loop over runs
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
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
                    result.append([str(run),str(cmslsnum),hltpathname,l1name,str(hltprescale),str(l1prescale),'%.2f'%(recorded*scalefactor),'%.2f'%(lumival*scalefactor)])
                else:
                    result.append([str(run),str(cmslsnum),hltpathname,l1name,str(hltprescale),str(l1prescale),'%.2f'%(recorded*scalefactor),'n/a'])
    labels = [('Run','LS','HLTpath','L1bit','HLTpresc','L1presc','Recorded(/ub)','Effective(/ub)')]
    print ' ==  = '
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace_strict(x,22) )
def toCSVLSEffective(lumidata,filename,resultlines,scalefactor,isverbose):
    '''
    input:  {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]},bxdata,beamdata]}
    '''
    result=[]#[run,ls,hltpath,l1bitname,hltpresc,l1presc,efflumi]
    for run in sorted(lumidata):#loop over runs
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
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
                    result.append([run,cmslsnum,hltpathname,l1name,hltprescale,l1prescale,recorded*scalefactor,lumival*scalefactor])
                else:
                    result.append([run,cmslsnum,hltpathname,l1name,hltprescale,l1prescale,recorded*scalefactor,'n/a'])
    fieldnames = ['Run','LS','HLTpath','L1bit','HLTpresc','L1presc','Recorded(/ub)','Effective(/ub)']
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
    input:  {run:[lumilsnum(0),triggeredls(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]},bxdata,beamdata](8)}
    screen Run,SelectedLS,Recorded,HLTPath,L1Bit,Effective
    '''
    result=[]#[run,selectedlsStr,recorded,hltpath,l1bit,efflumi]
    totdict={}#{hltpath:[nls,toteff]}
    hprescdict={}
    lprescdict={}
    alltotrecorded=0.0
    selectedcmsls=[]
    for run in sorted(lumidata):#loop over runs
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a'])
            continue
        selectedcmsls=[x[1] for x in rundata if x[1]!=0]
        totrecorded=sum([x[6] for x in rundata if x[6] is not None])
        alltotrecorded+=totrecorded
        totefflumiDict={}
        pathmap={}#{hltpathname:1lname}
        for lsdata in rundata:
            cmslsnum=lsdata[1]
            efflumiDict=lsdata[8]# this ls has no such path?
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
                if not totdict.has_key(hltpathname):
                    totdict[hltpathname]=[0,0.0]
                if l1presc and hltpresc and l1presc*hltpresc!=0:
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
                else:
                    if cmslsnum in selectedcmsls:
                        selectedcmsls.remove(cmslsnum)
        if len(selectedcmsls)==0:
            selectedlsStr='n/a'
        else:
            selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        for name in sorted(totefflumiDict):
            lname=pathmap[name]
            if lname=='n/a':
                continue
            (efflumival,efflumiunit)=CommonUtil.guessUnit(totefflumiDict[name])
            (totrecval,totrecunit)=CommonUtil.guessUnit(totrecorded)
            hprescs=list(set(hprescdict[hltpathname]))
            lprescs=list(set(lprescdict[lname]))
            hprescStr='('+','.join(['%d'%(x) for x in hprescs])+')'
            lprescStr='('+','.join(['%d'%(x) for x in lprescs])+')'
            #print 'efflumival , efflumiunit ',efflumival,efflumiunit
            result.append([str(run),selectedlsStr,'%.3f'%(totrecval*scalefactor)+'('+totrecunit+')',name+hprescStr,lname+lprescStr,'%.3f'%(efflumival*scalefactor)+'('+efflumiunit+')'])
    labels = [('Run','SelectedLS','Recorded','HLTpath','L1bit','Effective')]
    print ' ==  = '
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace_strict(x,22) )
    print ' ==  =  Total : '
    lastrowlabels=[('HLTPath','SelectedLS','Recorded','Effective')]
    totresult=[]
    (alltotrecval,alltotrecunit)=CommonUtil.guessUnit(alltotrecorded)
    for hname in sorted(totdict):
        hdata=totdict[hname]
        totnls=hdata[0]
        (toteffval,toteffunit)=CommonUtil.guessUnit(hdata[1])
        totresult.append([hname,str(totnls),'%.2f'%(alltotrecval*scalefactor)+'('+alltotrecunit+')','%.2f'%(toteffval*scalefactor)+'('+toteffunit+')'])
    print tablePrinter.indent (lastrowlabels+totresult, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    
def toCSVTotEffective(lumidata,filename,resultlines,scalefactor,isverbose):
    result=[]#[run,hltpath,l1bitname,totefflumi]
    for run in sorted(lumidata):#loop over runs
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a'])
            continue
        selectedcmslsStr=[x[1] for x in rundata if x[1]!=0]
        totrecorded=sum([x[6] for x in rundata if x[6] is not None])
        totefflumiDict={}
        pathmap={}
        for lsdata in rundata:
            efflumiDict=lsdata[8]# this ls has no such path?
            if not efflumiDict:
                continue
            for hltpathname,pathdata in efflumiDict.items():
                if not totefflumiDict.has_key(hltpathname):
                    totefflumiDict[hltpathname]=0.0
                    pathmap[hltpathname]='n/a'                
                lumival=pathdata[3]
                if lumival:
                    totefflumiDict[hltpathname]+=lumival
                    if pathdata[0] is None:
                        pathmap[hltpathname]='n/a'
                    else:
                        pathmap[hltpathname]=pathdata[0].replace('\"','')
        for name in sorted(totefflumiDict):
            result.append([run,name,pathmap[name],totefflumiDict[name]*scalefactor])
    fieldnames=['Run','HLTpath','L1bit','Effective(/ub)']
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
    input:{run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),{hltpath:[l1name,l1prescale,hltprescale,efflumi]},bxdata,beamdata]}
    output:
    fieldnames=['Run','CMSLS','Delivered(/ub)','Recorded(/ub)','BX']
    '''
    result=[]
    assert(filename)
    fieldnames=['run','ls','delivered(/ub)','recorded(/ub)','bx']
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([run,'n/a','n/a','n/a','n/a'])
            continue
        for lsdata in rundata:
            cmslsnum=lsdata[1]
            deliveredlumi=lsdata[5]
            recordedlumi=lsdata[6]
            (bxidxlist,bxvaluelist,bxerrorlist)=lsdata[8]
            bxresult=[]
            if bxidxlist and bxvaluelist:
                bxinfo=CommonUtil.transposed([bxidxlist,bxvaluelist])
                bxresult=CommonUtil.flatten([run,cmslsnum,deliveredlumi*scalefactor,recordedlumi*scalefactor,bxinfo])
                result.append(bxresult)
            else:
                result.append([run,cmslsnum,deliveredlumi*scalefactor,recordedlumi*scalefactor])
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


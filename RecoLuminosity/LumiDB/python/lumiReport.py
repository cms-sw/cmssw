import os,sys
from RecoLuminosity.LumiDB import tablePrinter, csvReporter,CommonUtil
from RecoLuminosity.LumiDB.wordWrappers import wrap_always, wrap_onspace, wrap_onspace_strict

def toScreenTotDelivered(lumidata,isverbose):
    '''
    input:  {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    '''
    result=[]
    for run in sorted(lumidata):
        lsdata=lumidata[run]
        if lsdata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a'])
            if isverbose:
                result.extend(['n/a'])
            continue
        nls=len(lsdata)
        totlumi=sum([x[5] for x in lsdata])
        (totlumival,lumiunit)=CommonUtil.guessUnit(totlumi)
        beamenergyPerLS=[float(x[4]) for x in lsdata]
        avgbeamenergy=0.0
        if len(beamenergyPerLS):
            avgbeamenergy=sum(beamenergyPerLS)/len(beamenergyPerLS)
        runstarttime=lsdata[0][2]
        if isverbose:
            selectedls=[(x[0],x[1]) for x in lsdata]
            result.append([str(run),str(nls),'%.3f'%(totlumival)+' ('+lumiunit+')',runstarttime.strftime("%b %d %Y %H:%M:%S"),'%.3f'%(avgbeamenergy), str(selectedls)])
        else:
            result.append([str(run),str(nls),'%.3f'%(totlumival)+' ('+lumiunit+')',runstarttime.strftime("%b %d %Y %H:%M:%S"),'%.3f'%(avgbeamenergy)])
    if isverbose:
        labels = [('Run', 'Total LS', 'Delivered','Start Time','Beam Energy(GeV)','Selected LS')]
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    else:
        labels = [('Run', 'Total LS', 'Delivered','Start Time','Beam Energy(GeV)')]
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,40) )
def toCSVTotDelivered(lumidata,filename,isverbose):
    '''
    input:  {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    '''
    result=[]
    r=csvReporter.csvReporter(filename)
    fieldnames = ['Run', 'Total LS', 'Delivered(1/ub)','UTCTime','BeamEnergy(GeV)']
    if isverbose:
        fieldnames.append('Selected LS')
    for run in sorted(lumidata):
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
            result.append([run,nls,totlumival,runstarttime.strftime("%b %d %Y %H:%M:%S"),avgbeamenergy, str(selectedls)])
        else:
            result.append([run,nls,totlumival,runstarttime.strftime("%b %d %Y %H:%M:%S"),avgbeamenergy])
        r.writeRow(fieldnames)
        r.writeRows(result)
def toScreenLSDelivered(lumidata,isverbose):
    result=[]
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
        for lsdata in rundata:
            if lsdata is None or len(lsdata)==0:
                result.append([str(run),'n/a','n/a','n/a','n/a','n/a','n/a'])
                continue
            else:
                lumils=lsdata[0]
                cmsls=lsdata[1]
                lsts=lsdata[2]
                beamstatus=lsdata[3]
                beamenergy=lsdata[4]
                delivered=lsdata[5]
                result.append([str(run),str(lumils),str(cmsls),'%.3f'%delivered,lsts.strftime('%b %d %Y %H:%M:%S'),beamstatus,'%.3f'%beamenergy])
    labels = [('Run','lumils','cmsls','Delivered(1/ub)','UTCTime','Beam Status','Beam Energy(GeV)')]
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
         
def toCSVLSDelivered(lumidata,filename,isverbose):
    result=[]
    fieldnames=['Run','lumils','cmsls','Delivered(1/ub)','UTCTime','eamStatus','Beam Energy(GeV)']
    r=csvReporter.csvReporter(filename)
    for run in sorted(lumidata):
        rundata=lumidata[run]
        if rundata is None:
            result.append([run,'n/a','n/a','n/a','n/a','n/a','n/a'])
            continue
        for lsdata in rundata:
            if lsdata is None:
                result.append([run,'n/a','n/a','n/a','n/a','n/a','n/a'])
                continue
            else:
                lumils=lsdata[0]
                cmsls=lsdata[1]
                lsts=lsdata[2]
                beamstatus=lsdata[3]
                beamenergy=lsdata[4]
                delivered=lsdata[5]
                result.append([run,lumils,cmsls,delivered,lsts,beamstatus,beamenergy])
    r.writeRow(fieldnames)
    r.writeRows(result)

def toScreenOverview(lumidata,isverbose):
    '''
    input:  {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,recordedlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    '''
    result=[]
    labels = [('Run', 'Delivered LS', 'Delivered','Selected LS','Recorded')]
    totaltable=[]
    totalDeliveredLS = 0
    totalSelectedLS = 0
    totalDelivered = 0.0
    totalRecorded = 0.0
    for run in sorted(lumidata):
        lsdata=lumidata[run]
        if lsdata is None:
            result.append([str(run),'n/a','n/a','n/a','n/a'])
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
        selectedcmsls=[x[1] for x in lsdata if x[1]!=0]
        totalSelectedLS+=len(selectedcmsls)
        selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        result.append([str(run),str(nls),'%.3f'%(totdeliveredlumi)+' ('+deliveredlumiunit+')',selectedlsStr,'%.3f'%(totrecordedlumi)+' ('+recordedlumiunit+')'])
    print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    print ' ==  =  Total : '
    (totalDeliveredVal,totalDeliveredUni)=CommonUtil.guessUnit(totalDelivered)
    (totalRecordedVal,totalRecordedUni)=CommonUtil.guessUnit(totalRecorded)
    totrowlabels = [('Delivered LS','Delivered('+totalDeliveredUni+')','Selected LS','Recorded('+totalRecordedUni+')')]
    totaltable.append([str(totalDeliveredLS),'%.3f'%totalDeliveredVal,str(totalSelectedLS),'%.3f'%totalRecordedVal])
    print tablePrinter.indent (totrowlabels+totaltable, hasHeader = True, separateRows = False, prefix = '| ',
                               postfix = ' |', justify = 'right', delim = ' | ',
                               wrapfunc = lambda x: wrap_onspace (x, 20))
    
def toCSVOverview(lumidata,filename,isverbose):
    result=[]
    fieldnames = ['Run', 'DeliveredLS', 'Delivered(1/ub)','SelectedLS','Recorded(1/ub)']
    r=csvReporter.csvReporter(filename)
    for run in sorted(lumidata):
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
        selectedlsStr = CommonUtil.splitlistToRangeString(selectedcmsls)
        result.append([run,nls,totdeliveredlumi,selectedlsStr,totrecordedlumi])
    r.writeRow(fieldnames)
    r.writeRows(result)
def toCSVBXInfo(lumidata,filename,bxfield='bxlumi'):
    '''
    dump selected bxlumi or beam intensity as the last field
    '''
    if bxfields not in ['bxlumi','bxintensity']:
        print 'not recognized bxfield'
        return
    pass



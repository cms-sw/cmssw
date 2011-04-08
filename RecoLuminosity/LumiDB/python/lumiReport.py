import datetime
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
        print runstarttime
        if isverbose:
            selectedls=[(x[0],x[1]) for x in lsdata]
            result.append([str(run),str(nls),'%.3f'%(totlumival)+' ('+lumiunit+')',runstarttime.strftime("%b %d %Y %H:%M:%S"),'%.3f'%(avgbeamenergy), str(selectedls)])
        else:
            result.append([str(run),str(nls),'%.3f'%(totlumival)+' ('+lumiunit+')',runstarttime.strftime("%b %d %Y %H:%M:%S"),'%.3f'%(avgbeamenergy)])
    if isverbose:
        labels = [('Run', 'Total LS', 'Delivered Lumi','Start Time','Beam Energy(GeV)','Selected LS')]
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,20) )
    else:
        labels = [('Run', 'Total LS', 'Delivered Lumi','Start Time','Beam Energy (GeV)')]
        print tablePrinter.indent (labels+result, hasHeader = True, separateRows = False,
                               prefix = '| ', postfix = ' |', justify = 'right',
                               delim = ' | ', wrapfunc = lambda x: wrap_onspace (x,40) )
def toCSVTotDelivered(lumidata,filename,isverbose):
    '''
    input:  {run:[lumilsnum,cmslsnum,timestamp,beamstatus,beamenergy,deliveredlumi,calibratedlumierror,(bxidx,bxvalues,bxerrs),(bxidx,b1intensities,b2intensities)]}
    '''
    result=[]
    r=csvReporter.csvReporter(filename)
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
    if isverbose:
        fieldnames = ['Run', 'Total LS', 'Delivered Lumi(1/ub)','Start Time','Beam Energy(GeV)','Selected LS']
        r.writeRow(fieldnames)
        r.writeRows(result)
    else:
        fieldnames = ['Run', 'Total LS', 'Delivered Lumi','Start Time','Beam Energy (GeV)']
        r.writeRow(fieldnames)
        r.writeRows(result)
def toScreenLSDelivered(lumidata,isverbose):
    pass



from optparse import OptionParser
import sys
from dqmlayouts import *

genLists = {
    'shift': ('shift_ecal_layout', 'shift_ecal_T0_layout', 'shift_ecal_relval_layout'),
    'ecal': ('ecal-layouts', 'ecal_T0_layouts'),
    'overview': ('ecal_overview_layouts'),
    'online': ('ecal-layouts', 'shift_ecal_layout', 'ecal_overview_layouts'),
    'offline': ('ecal_T0_layouts', 'shift_ecal_T0_layout', 'ecal_overview_layouts'),
    'relval': ('ecal_relval-layouts', 'ecalmc_relval-layouts', 'shift_ecal_relval_layout'),
    'all': ('shift_ecal_layout', 'shift_ecal_T0_layout', 'shift_ecal_relval_layout',
        'ecal-layouts', 'ecal_T0_layouts', 'ecal_overview_layouts',
        'ecal_relval-layouts', 'ecalmc_relval-layouts', 'shift_ecal_relval_layout'),
    'priv': ('ecalpriv-layouts', 'ecal_overview_layouts')
}

optparser = OptionParser()
optparser.add_option('-l', '--list', dest = 'list', help = 'LIST=(shift|ecal|overview|online|offline|relval|all)', metavar = 'LIST', default = 'all')
optparser.add_option('-t', '--target-dir', dest = 'targetDir', help = '', metavar = '', default = '.')
optparser.add_option('-s', '--source-dir', dest = 'sourceDir', help = '', metavar = '')

(options, args) = optparser.parse_args()

if options.list not in genLists:
    optparser.print_usage()
    exit

if options.sourceDir == '':
    optparser.print_usage()
    exit

genList = genLists[options.list]
targetDir = options.targetDir
sourceDir = options.sourceDir

sys.path.append(sourceDir)

#### BEGIN path definitions / utility functions ####

from DQM.EcalBarrelMonitorTasks.ClusterTask_cfi import ecalClusterTask
from DQM.EcalBarrelMonitorTasks.EnergyTask_cfi import ecalEnergyTask
from DQM.EcalBarrelMonitorTasks.IntegrityTask_cfi import ecalIntegrityTask
from DQM.EcalBarrelMonitorTasks.LaserTask_cfi import ecalLaserTask
from DQM.EcalBarrelMonitorTasks.LedTask_cfi import ecalLedTask
from DQM.EcalBarrelMonitorTasks.OccupancyTask_cfi import ecalOccupancyTask
from DQM.EcalBarrelMonitorTasks.PedestalTask_cfi import ecalPedestalTask
from DQM.EcalBarrelMonitorTasks.PNDiodeTask_cfi import ecalPnDiodeTask
from DQM.EcalBarrelMonitorTasks.PresampleTask_cfi import ecalPresampleTask
from DQM.EcalBarrelMonitorTasks.RawDataTask_cfi import ecalRawDataTask
from DQM.EcalBarrelMonitorTasks.SelectiveReadoutTask_cfi import ecalSelectiveReadoutTask
from DQM.EcalBarrelMonitorTasks.TestPulseTask_cfi import ecalTestPulseTask
from DQM.EcalBarrelMonitorTasks.TimingTask_cfi import ecalTimingTask
from DQM.EcalBarrelMonitorTasks.TrigPrimTask_cfi import ecalTrigPrimTask

from DQM.EcalBarrelMonitorClient.IntegrityClient_cfi import ecalIntegrityClient
from DQM.EcalBarrelMonitorClient.LaserClient_cfi import ecalLaserClient
from DQM.EcalBarrelMonitorClient.LedClient_cfi import ecalLedClient
from DQM.EcalBarrelMonitorClient.OccupancyClient_cfi import ecalOccupancyClient
from DQM.EcalBarrelMonitorClient.PedestalClient_cfi import ecalPedestalClient
from DQM.EcalBarrelMonitorClient.PNIntegrityClient_cfi import ecalPnIntegrityClient
from DQM.EcalBarrelMonitorClient.PresampleClient_cfi import ecalPresampleClient
from DQM.EcalBarrelMonitorClient.RawDataClient_cfi import ecalRawDataClient
from DQM.EcalBarrelMonitorClient.SelectiveReadoutClient_cfi import ecalSelectiveReadoutClient
from DQM.EcalBarrelMonitorClient.SummaryClient_cfi import ecalSummaryClient
from DQM.EcalBarrelMonitorClient.TestPulseClient_cfi import ecalTestPulseClient
from DQM.EcalBarrelMonitorClient.TimingClient_cfi import ecalTimingClient
from DQM.EcalBarrelMonitorClient.TrigPrimClient_cfi import ecalTrigPrimClient
from DQM.EcalBarrelMonitorClient.CalibrationSummaryClient_cfi import ecalCalibrationSummaryClient

def extractPaths(worker) :
    paths = dict()
    for (me, params) in worker['MEs'].items():
        paths[me] = params['path']

    return paths

clusterTaskPaths = extractPaths(ecalClusterTask)
energyTaskPaths = extractPaths(ecalEnergyTask)
integrityTaskPaths = extractPaths(ecalIntegrityTask)
laserTaskPaths = extractPaths(ecalLaserTask)
ledTaskPaths = extractPaths(ecalLedTask)
occupancyTaskPaths = extractPaths(ecalOccupancyTask)
pedestalTaskPaths = extractPaths(ecalPedestalTask)
pnDiodeTaskPaths = extractPaths(ecalPnDiodeTask)
presampleTaskPaths = extractPaths(ecalPresampleTask)
rawDataTaskPaths = extractPaths(ecalRawDataTask)
selectiveReadoutTaskPaths = extractPaths(ecalSelectiveReadoutTask)
testPulseTaskPaths = extractPaths(ecalTestPulseTask)
timingTaskPaths = extractPaths(ecalTimingTask)
trigPrimTaskPaths = extractPaths(ecalTrigPrimTask)
integrityClientPaths = extractPaths(ecalIntegrityClient)
laserClientPaths = extractPaths(ecalLaserClient)
ledClientPaths = extractPaths(ecalLedClient)
occupancyClientPaths = extractPaths(ecalOccupancyClient)
pedestalClientPaths = extractPaths(ecalPedestalClient)
pnIntegrityClientPaths = extractPaths(ecalPnIntegrityClient)
presampleClientPaths = extractPaths(ecalPresampleClient)
rawDataClientPaths = extractPaths(ecalRawDataClient)
selectiveReadoutClientPaths = extractPaths(ecalSelectiveReadoutClient)
summaryClientPaths = extractPaths(ecalSummaryClient)
testPulseClientPaths = extractPaths(ecalTestPulseClient)
timingClientPaths = extractPaths(ecalTimingClient)
trigPrimClientPaths = extractPaths(ecalTrigPrimClient)
calibrationSummaryClientPaths = extractPaths(ecalCalibrationSummaryClient)

smNamesEE = [
    "EE-01", "EE-02", "EE-03", "EE-04", "EE-05", "EE-06", "EE-07", "EE-08", "EE-09",
    "EE+01", "EE+02", "EE+03", "EE+04", "EE+05", "EE+06", "EE+07", "EE+08", "EE+09"]

smNamesEB = [
    "EB-01", "EB-02", "EB-03", "EB-04", "EB-05", "EB-06", "EB-07", "EB-08", "EB-09",
    "EB-10", "EB-11", "EB-12", "EB-13", "EB-14", "EB-15", "EB-16", "EB-17", "EB-18",
    "EB+01", "EB+02", "EB+03", "EB+04", "EB+05", "EB+06", "EB+07", "EB+08", "EB+09",
    "EB+10", "EB+11", "EB+12", "EB+13", "EB+14", "EB+15", "EB+16", "EB+17", "EB+18"]

smMEMNamesEE = ["EE-02", "EE-03", "EE-07", "EE-08", "EE+02", "EE+03", "EE+07", "EE+08"]

laserWavelengths = ['1', '2', '3']

ledWavelengths = ['1', '2']

mgpaGains = ['12']

pnMGPAGains = ['16']

ebRep = {'subdet': 'EcalBarrel', 'prefix': 'EB', 'suffix': ''}
eeRep = {'subdet': 'EcalEndcap', 'prefix': 'EE'}
eemRep = {'subdet': 'EcalEndcap', 'prefix': 'EE', 'suffix': ' EE -'}
eepRep = {'subdet': 'EcalEndcap', 'prefix': 'EE', 'suffix': ' EE +'}

def ecal2P(name, path, description='', rep={}) :
    replacements = {}
    if len(rep) > 0:
        for key, value in rep.items():
            replacements[key] = '%(' + key + ')s'
            
    rows = []
    replacements.update(ebRep)
    rows.append([[path % replacements, description]])
    replacements.update(eeRep)
    rows.append([[path % replacements, description]])
    if len(rep) > 0:
        return LayoutElemSet(name, rows, rep)
    else:
        return LayoutElem(name, rows)

def ecal3P(name, path, description='', rep={}) :
    replacements = {}
    if len(rep) > 0:
        for key, value in rep.items():
            replacements[key] = '%(' + key + ')s'
    
    rows = []
    replacements.update(ebRep)
    rows.append([[path % replacements, description]])
    columns = []
    replacements.update(eemRep)
    columns.append([path % replacements, description])
    replacements.update(eepRep)
    columns.append([path % replacements, description])
    rows.append(columns)
    if len(rep) > 0:
        return LayoutElemSet(name, rows, rep)
    else:
        return LayoutElem(name, rows)

def ee2P(name, path, description='', rep={}) :
    replacements = {}
    if len(rep) > 0:
        for key, value in rep.items():
            replacements[key] = '%(' + key + ')s'
    
    columns = []
    replacements.update(eemRep)
    columns.append([path % replacements, description])
    replacements.update(eepRep)
    columns.append([path % replacements, description])
    if len(rep) > 0:
        return LayoutElemSet(name, [columns], rep)
    else:
        return LayoutElem(name, [columns])
    
def subdetSet_produce(subdet, paths) :
    rows = []
    rep = {'subdet': subdet[0], 'prefix': subdet[1]}
    if len(paths) <= 2:
        for p in paths:
            path = ''
            if type(p) is TupleType:
                path = p[0]
                rep.update(p[1])
            else:
                path = p

            for ph in findall(r'\%\(([^\)]+)\)s', path):
                if ph not in rep:
                    rep.update({ph: '%(' + ph + ')s'})
                        
            rows.append([[path % rep]])
    else:
        ncol = 0
        row = []
        for p in paths:
            path = ''
            if type(p) is TupleType:
                path = p[0]
                rep.update(p[1])
            else:
                path = p

            for ph in findall(r'\%\(([^\)]+)\)s', path):
                if ph not in rep:
                    rep.update({ph: '%(' + ph + ')s'})
                        
            row.append([path % rep])
            ncol += 1
            if ncol == 2:
                rows.append(row)
                ncol = 0
                row = []

    return rows

def eeSMSet(name, *paths) :
    rows = subdetSet_produce(('EcalEndcap', 'EE'), paths)
    return [LayoutElemSet(name + ' %(sm)s', rows, {'sm': smNamesEE}, addSerial = False)]

def eeSMMEMSet(name, *paths) :
    rows = subdetSet_produce(('EcalEndcap', 'EE'), paths)
    return [LayoutElemSet(name + ' %(sm)s', rows, {'sm': smMEMNamesEE}, addSerial = False)]

def ebSMSet(name, *paths) :
    rows = subdetSet_produce(('EcalBarrel', 'EB'), paths)
    return [LayoutElemSet(name + ' %(sm)s', rows, {'sm': smNamesEB}, addSerial = False)]

def smSet(name, *paths) :
    return eeSMSet(name, *paths) + ebSMSet(name, *paths)

def smMEMSet(name, *paths) :
    return eeSMMEMSet(name, *paths) + ebSMSet(name, *paths)

def subdetEtaPhi(name, path, patheta, pathphi) :
    elems = []
    elems.append(LayoutElem(name + " EB", [[[path % ebRep]], [[patheta % ebRep], [pathphi % ebRep]]]))
    for rep in [eemRep, eepRep]:
        elems.append(LayoutElem(name + rep['suffix'], [[[path % rep]], [[patheta % rep], [pathphi % rep]]]))

    return elems

#### END path definitions / utility functions ####

layouts = {}

#### BEGIN shift_ecal_layout / shift_ecal_T0_layout ####

layouts['shift_ecal_layout'] = LayoutDir("00 Shift/Ecal", [
    ecal3P('Summary', summaryClientPaths["QualitySummary"], "Combined summary of integrity, presample, timing, FE status, trigger primitives, and hot cell quality"),
    ecal3P('FE Status', rawDataClientPaths['QualitySummary'], "DCC front-end status quality summary."),
    ecal3P('Integrity', integrityClientPaths['QualitySummary'], "Quality summary checking that data for each crystal follows all the formatting rules and all the constraints which are dictated by the design of the electronics."),
    ecal3P('Occupancy', occupancyTaskPaths['DigiAll'], "Digi occupancy."),        
    ecal3P('Noise', presampleClientPaths['QualitySummary'], "Presample quality determined by mean and RMS."),
    ecal3P('RecHit Energy', energyTaskPaths['HitMapAll'], "RecHit energy profile."),            
    ecal3P('Timing', timingClientPaths['QualitySummary'], "Timing alignment of the Ecal reconstructed hit."),
    ecal3P('TriggerPrimitives', trigPrimClientPaths['EmulQualitySummary'], "Trigger primitive quality determined by emulator matching."),    
#    ecal3P('Hot Cells', occupancyClientPaths['QualitySummary'], "Phi-symmetry of the channel occupancy."),    
    ecal3P('Laser', laserClientPaths['QualitySummary'], 'Quality of the main laser (laser 3)', rep = {'wl': '3'}),
    ecal2P('Laser PN', laserClientPaths['PNQualitySummary'], 'Quality of the main laser signal on PN diodes', rep = {'wl': '3'}),
    ecal3P('Test Pulse', testPulseClientPaths['QualitySummary'], 'Quality of the test pulse injection', rep = {'gain': '12'}),
    ecal2P('Test Pulse PN', testPulseClientPaths['PNQualitySummary'], 'Quality of the test pulse injection on PN diodes', rep = {'pngain': '16'}),
    ee2P('Led', ledClientPaths['QualitySummary'], 'Quality of LED 1', rep = {'wl': '1'}),    
    LayoutElem("Synchronization Errors", [[[rawDataTaskPaths['TrendNSyncErrors'], 'Accumulated number of synchronization errors']]])
])

layouts['shift_ecal_T0_layout'] = layouts['shift_ecal_layout'].clone()
layouts['shift_ecal_T0_layout'].remove("Laser")
layouts['shift_ecal_T0_layout'].remove("Laser PN")
layouts['shift_ecal_T0_layout'].remove("Test Pulse")
layouts['shift_ecal_T0_layout'].remove("Test Pulse PN")
layouts['shift_ecal_T0_layout'].remove("Led")
layouts['shift_ecal_T0_layout'].remove("Synchronization Errors")

#### END shift_ecal_layout / shift_ecal_T0_layout ####

#### BEGIN ecal-layouts.py / ecal_T0_layouts.py / ecalpriv-layouts.py ####

layouts['ecal-layouts'] = LayoutDir("Ecal/Layouts", [
    ecal3P('Summary', summaryClientPaths["QualitySummary"], "Combined summary of integrity, presample, timing, FE status, trigger primitives, and hot cell quality"),
    ecal3P('Occupancy Summary', occupancyTaskPaths['DigiAll'], "Digi occupancy."),
    ecal3P('Calibration Summary', calibrationSummaryClientPaths['QualitySummary'], "Combined summary of calibration sequence data quality."),
    LayoutDir("Overview", []),
    LayoutDir("Electronics", []),
    LayoutDir("Noise", []),
    LayoutDir("Occupancy", []),
    LayoutDir("Energy", []),
    LayoutDir("Timing", []),
    LayoutDir("Trigger Primitives", []),
    LayoutDir("Selective Readout", []),
    LayoutDir("Laser", []),
    LayoutDir("Led", []),
    LayoutDir('Test Pulse', []),
    LayoutDir('Trend', []),
    LayoutDir("By SuperModule", [])
])

layouts['ecal-layouts'].get('Overview').append([
    ecal3P('Summary', summaryClientPaths["QualitySummary"], "Combined summary of integrity, presample, timing, FE status, trigger primitives, and hot cell quality"),
    ecal3P('FE Status', rawDataClientPaths['QualitySummary'], "DCC front-end status quality summary."),
    ecal3P('Integrity', integrityClientPaths['QualitySummary'], "Quality summary checking that data for each crystal follows all the formatting rules and all the constraints which are dictated by the design of the electronics."),
    ecal3P('Occupancy', occupancyTaskPaths['DigiAll'], "Digi occupancy."),
    ecal3P('Noise', presampleClientPaths['QualitySummary'], "Presample quality determined by mean and RMS."),
    ecal3P('RecHit Energy', energyTaskPaths['HitMapAll'], "RecHit energy profile."),        
    ecal3P('Timing', timingClientPaths['QualitySummary'], "Timing alignment of the Ecal reconstructed hit."),
    ecal3P('Trigger Primitives', trigPrimClientPaths['EmulQualitySummary'], "Trigger primitive quality determined by emulator matching."),
    ecal3P('Hot Cells', occupancyClientPaths['QualitySummary'], "Phi-symmetry of the channel occupancy."),
    ecal3P('Laser', laserClientPaths['QualitySummary'], 'Quality of the main laser (laser 3)', rep = {'wl': '3'}),
    ecal2P('Laser PN', laserClientPaths['PNQualitySummary'], 'Quality of the main laser signal on PN diodes', rep = {'wl': '3'}),
    ecal3P('Test Pulse', testPulseClientPaths['QualitySummary'], 'Quality of the test pulse injection', rep = {'gain': '12'}),
    ecal2P('Test Pulse PN', testPulseClientPaths['PNQualitySummary'], 'Quality of the test pulse injection on PN diodes', rep = {'pngain': '16'}),
    ee2P('Led', ledClientPaths['QualitySummary'], 'Quality of LED 1', rep = {'wl': '1'}),
    LayoutElem('Error Trends', [[[rawDataTaskPaths['TrendNSyncErrors'], 'Accumulated number of DCC-FE synchronization errors']], [[integrityTaskPaths['TrendNErrors'], 'Number of channel with integrity errors']]])
])

layouts['ecal-layouts'].get("Electronics").append([
    ecal3P('Integrity Summary', integrityClientPaths['QualitySummary']),
    ecal3P('Integrity Errors', integrityTaskPaths['Total']),
    ecal3P('FEStatus Summary', rawDataClientPaths['QualitySummary'], "DCC front-end status quality summary."),
    ecal3P('FE Sync Errors', rawDataTaskPaths['DesyncTotal']),
    ecal3P('Integrity Errors by Lumi', integrityTaskPaths['ByLumi']),
    ecal3P('Total Integrity Errors', integrityTaskPaths['Total'], 'Total number of integrity errors for each FED. For the list of channels with errors, please go to Ecal/Integrity/Gain etc.'),
    LayoutDir('IntegrityQuality', smSet('Integrity', integrityClientPaths['Quality'])),
    LayoutDir('FEStatus', smSet('FE Status Flags', rawDataTaskPaths['FEStatus']))
])

layouts['ecal-layouts'].get("Noise").append([
    ecal3P('Presample Quality', presampleClientPaths['QualitySummary']),
    ecal3P('RMS Map', presampleClientPaths['RMSMap']),
    LayoutDir('Quality', smSet('Quality', presampleClientPaths['Quality'])),        
    LayoutDir('Distributions', smSet('Distributions', presampleClientPaths['Mean'], presampleClientPaths['RMS']))
])

layouts['ecal-layouts'].get("Occupancy").append([
    ecal3P('Hot Cells', occupancyClientPaths['QualitySummary'])
])
layouts['ecal-layouts'].get("Occupancy").append(
    subdetEtaPhi("Digi", occupancyTaskPaths['DigiAll'], occupancyTaskPaths['DigiProjEta'], occupancyTaskPaths['DigiProjPhi']) +
    subdetEtaPhi("RecHit (Filtered)", occupancyTaskPaths['RecHitThrAll'], occupancyTaskPaths['RecHitThrProjEta'], occupancyTaskPaths['RecHitThrProjPhi']) +
    subdetEtaPhi("Trigger Primitive (Filtered)", occupancyTaskPaths['TPDigiThrAll'], occupancyTaskPaths['TPDigiThrProjEta'], occupancyTaskPaths['TPDigiThrProjPhi']) +
    subdetEtaPhi("Basic Cluster", clusterTaskPaths['BCOccupancy'], clusterTaskPaths['BCOccupancyProjEta'], clusterTaskPaths['BCOccupancyProjPhi'])
)
layouts['ecal-layouts'].get("Occupancy").append([
    ecal3P('Super Cluster Seed', clusterTaskPaths['SCSeedOccupancy']),
    ecal2P('Super Cluster Multiplicity', clusterTaskPaths['SCNum']),                
    ecal3P('Single Crystal Cluster', clusterTaskPaths['SingleCrystalCluster']),
    ecal3P('Laser3', laserTaskPaths['Occupancy'], rep = {'wl': '3'}),
    LayoutElem('Led', [
        [[ledTaskPaths['Occupancy'] % {'wl': '1', 'suffix': ' EE -'}], [ledTaskPaths['Occupancy'] % {'wl': '1', 'suffix': ' EE +'}]],
        [[ledTaskPaths['Occupancy'] % {'wl': '2', 'suffix': ' EE -'}], [ledTaskPaths['Occupancy'] % {'wl': '2', 'suffix': ' EE -'}]]
    ]),
    ecal3P('Test Pulse', testPulseTaskPaths['Occupancy'], rep = {'gain': '12'}),
    LayoutDir('Details', smSet('Occupancy', occupancyTaskPaths['Digi']))
])

layouts['ecal-layouts'].get("Energy").append([
    ecal3P('RecHit Energy', energyTaskPaths['HitMapAll']),
    ecal3P('RecHit Energy Spectrum', energyTaskPaths['HitAll'])
])
layouts['ecal-layouts'].get("Energy").append(
    subdetEtaPhi("Basic Cluster Energy", clusterTaskPaths['BCEMap'], clusterTaskPaths['BCEMapProjEta'], clusterTaskPaths['BCEMapProjPhi']) +
    subdetEtaPhi("Basic Cluster Size", clusterTaskPaths['BCSizeMap'], clusterTaskPaths['BCSizeMapProjEta'], clusterTaskPaths['BCSizeMapProjPhi'])
)
layouts['ecal-layouts'].get("Energy").append([
    ecal2P('Super Cluster Energy', clusterTaskPaths['SCE']),
    ecal2P('Super Cluster Energy Low', clusterTaskPaths['SCELow']),
    ecal2P('Super Cluster Seed Energy', clusterTaskPaths['SCSeedEnergy']),
    ecal2P('Super Cluster R9', clusterTaskPaths['SCR9']),
    LayoutElem('Super Cluster Size', [
        [[clusterTaskPaths['SCNBCs'] % ebRep], [clusterTaskPaths['SCNcrystals'] % ebRep]],
        [[clusterTaskPaths['SCNBCs'] % eeRep], [clusterTaskPaths['SCNcrystals'] % eeRep]]
    ]),
    LayoutDir('Details', smSet('RecHit', energyTaskPaths['HitMap'], energyTaskPaths['Hit'])),
##     LayoutDir('DiClusterMass', [
##         LayoutElem('Pi0', [[[clusterTaskPaths['Pi0']]]]),
##         LayoutElem('JPsi', [[[clusterTaskPaths['JPsi']]]]),
##         LayoutElem('Z', [[[clusterTaskPaths['Z']]]]),
##         LayoutElem('High Mass', [[[clusterTaskPaths['HighMass']]]])
##     ])
])

layouts['ecal-layouts'].get("Timing").append([
    ecal3P('Quality Summary', timingClientPaths['QualitySummary']),
    ecal3P('Mean', timingClientPaths['MeanAll']),
    ecal3P('RMS', timingClientPaths['RMSAll'])
])
layouts['ecal-layouts'].get("Timing").append(
    subdetEtaPhi("Map", timingTaskPaths['TimeAllMap'], timingClientPaths['ProjEta'], timingClientPaths['ProjPhi'])
)
layouts['ecal-layouts'].get("Timing").append([
    LayoutElem("Forward-Backward EB", [
        [[timingClientPaths['FwdBkwdDiff'] % ebRep]],
        [[timingClientPaths['FwdvBkwd'] % ebRep]]
    ]),
    LayoutElem("Forward-Backward EE", [
        [[timingClientPaths['FwdBkwdDiff'] % eeRep]],
        [[timingClientPaths['FwdvBkwd'] % eeRep]]
    ]),
    ecal3P('Single Event', timingTaskPaths['TimeAll']),
    ecal3P('Vs Amptlitude', timingTaskPaths['TimeAmpAll']),
    LayoutDir('Quality', smSet('Quality', timingClientPaths['Quality'])),
    LayoutDir('Details', [
        LayoutDir('Profile', smSet('Time', timingTaskPaths['TimeMap'])),
        LayoutDir('Mean', smSet('Mean', timingClientPaths['MeanSM'])),
        LayoutDir('RMS', smSet('RMS', timingClientPaths['RMSMap'])),
        LayoutDir('Vs Amplitude', smSet('Time vs Amplitude', timingTaskPaths['TimeAmp']))
    ]),
    LayoutDir('Laser Timing', smSet('Laser3 Timing', (laserTaskPaths['Timing'], {'wl': '3'})))
])

layouts['ecal-layouts'].get("Trigger Primitives").append([
    ecal3P('Emulation Quality', trigPrimClientPaths['EmulQualitySummary']),
    ecal3P('Et Spectrum', trigPrimTaskPaths['EtReal']),
    ecal3P('Emulation Et Spectrum', trigPrimTaskPaths['EtMaxEmul']),
    ecal3P('Et Map', trigPrimTaskPaths['EtSummary']),
    ecal3P('Occupancy', occupancyTaskPaths['TPDigiThrAll']),
    ecal3P('Timing', trigPrimClientPaths['TimingSummary']),
    ecal3P("Occupancy vs BX", trigPrimTaskPaths['OccVsBx']),
    ecal3P("Et vs BX", trigPrimTaskPaths['EtVsBx']),
    ecal3P('Emululation Timing', trigPrimTaskPaths['EmulMaxIndex']),
    LayoutDir('EmulMatching', smSet('Match', trigPrimTaskPaths['MatchedIndex'])),
    LayoutDir('Details', smSet('TP Et', trigPrimTaskPaths['EtRealMap']))
])

layouts['ecal-layouts'].get("Selective Readout").append([
    ecal2P('DCC Size', selectiveReadoutTaskPaths['DCCSize']),
    ecal3P('Event Size per DCC', selectiveReadoutTaskPaths['EventSize']),
    ecal3P("ZS Filter Output (High Int.)", selectiveReadoutTaskPaths['HighIntOutput']),
    ecal3P("ZS Filter Output (Low Int.)", selectiveReadoutTaskPaths['LowIntOutput']),
    ecal3P('Full Readout Flags', selectiveReadoutClientPaths['FR']),
    ecal3P('Zero Suppression Flags', selectiveReadoutClientPaths['ZS1']),
    ecal3P('Tower Size', selectiveReadoutTaskPaths['TowerSize']),
    ecal3P('TT Flags', trigPrimTaskPaths['TTFlags']),
    ecal3P('High Interest Occupancy', trigPrimTaskPaths['HighIntMap']),
    ecal3P('Medium Interest Occupancy', trigPrimTaskPaths['MedIntMap']),
    ecal3P('Low Interest Occupancy', trigPrimTaskPaths['LowIntMap'])
])

layouts['ecal-layouts'].get("Laser").append([
    ecal3P("Quality Summary L%(wl)s", laserClientPaths['QualitySummary'], rep = {'wl': laserWavelengths}),
    ecal3P('Amplitude L%(wl)s', laserTaskPaths['AmplitudeSummary'], rep = {'wl': laserWavelengths}),
    ecal2P('Amplitude RMS L%(wl)s', laserClientPaths['AmplitudeRMS'], rep = {'wl': laserWavelengths}),
    ecal3P('Occupancy', laserTaskPaths['Occupancy']),
    ecal2P('Timing Spread L%(wl)s', laserClientPaths['TimingRMSMap'], rep = {'wl': laserWavelengths}),
    ecal2P('PN Quality Summary L%(wl)s', laserClientPaths['PNQualitySummary'], rep = {'wl': laserWavelengths}),
    LayoutDirSet('Laser%(wl)s', [
        LayoutDir('Quality', smSet('Quality', laserClientPaths['Quality'])),
        LayoutDir('Amplitude', 
            smSet('Amplitude', laserTaskPaths['Amplitude']) +
            smSet('Distribution', laserClientPaths['AmplitudeMean'])
        ),
        LayoutDir('Timing',
            smSet('Timing', laserTaskPaths['Timing']) +
            smSet('Distributions', laserClientPaths['TimingMean'], laserClientPaths['TimingRMS'])
        ),
        LayoutDir('APD Over PN', smSet('APD Over PN', laserTaskPaths['AOverP'])),
        LayoutDir('Shape', smSet('Shape', laserTaskPaths['Shape'])),
        LayoutDir('PNAmplitude', smMEMSet('Amplitude', laserTaskPaths['PNAmplitude'])),
    ], {'wl': laserWavelengths}, addSerial = False)
])

layouts['ecal-layouts'].get("Led").append([
    ee2P("Quality Summary L%(wl)s", ledClientPaths['QualitySummary'], rep = {'wl': ledWavelengths}),
    ee2P('Amplitude L%(wl)s', ledTaskPaths['AmplitudeSummary'], rep = {'wl': ledWavelengths}),    
    LayoutElemSet('Amplitude RMS L%(wl)s', [[[ledClientPaths['AmplitudeRMS']]]], {'wl': ledWavelengths}),
    ee2P('Occupancy', ledTaskPaths['Occupancy']),
    LayoutElemSet('Timing Spread L%(wl)s', [[[ledClientPaths['TimingRMSMap']]]], {'wl': ledWavelengths}),
    LayoutElemSet('PN Quality Summary L%(wl)s', [[[ledClientPaths['PNQualitySummary']]]], {'wl': ledWavelengths}),
    LayoutDirSet('Led%(wl)s', [
        LayoutDir('Quality', eeSMSet('Quality', ledClientPaths['Quality'])),
        LayoutDir('Amplitude', 
            eeSMSet('Amplitude', ledTaskPaths['Amplitude']) +
            eeSMSet('Distribution', ledClientPaths['AmplitudeMean'])
        ),
        LayoutDir('Timing', 
            eeSMSet('Timing', ledTaskPaths['Timing']) +
            eeSMSet('Distributions', ledClientPaths['TimingMean'])
        ),
        LayoutDir('APD Over PN', eeSMSet('APD Over PN', ledTaskPaths['AOverP'])),
        LayoutDir('Shape', eeSMSet('Shape', ledTaskPaths['Shape'])),
        LayoutDir('PNAmplitude', eeSMMEMSet('Amplitude', ledTaskPaths['PNAmplitude'])),
    ], {'wl': ledWavelengths}, addSerial = False)
])

layouts['ecal-layouts'].get("Test Pulse").append([
    ecal3P('Quality Summary G%(gain)s', testPulseClientPaths['QualitySummary'], rep = {'gain': mgpaGains}),
    ecal3P('Occupancy', testPulseTaskPaths['Occupancy']),
    ecal2P('PN Quality Summary G%(pngain)s', testPulseClientPaths['PNQualitySummary'], rep = {'pngain': pnMGPAGains}),
    LayoutDirSet('Gain%(gain)s', [
        LayoutDir('Quality', smSet('Quality', testPulseClientPaths['Quality'])),
        LayoutDir('Amplitude', 
            smSet('Amplitude', testPulseTaskPaths['Amplitude']) +
            smSet('RMS', testPulseClientPaths['AmplitudeRMS'])
        ),
        LayoutDir('Shape', smSet('Shape', testPulseTaskPaths['Shape']))
    ], {'gain': mgpaGains}, addSerial = False),
    LayoutDirSet('PNGain%(pngain)s', smSet('Amplitude', testPulseTaskPaths['PNAmplitude']), {'pngain': pnMGPAGains}, addSerial = False)
])

layouts['ecal-layouts'].get("Trend").append([
    LayoutElem('Errors', [
        [[rawDataTaskPaths['TrendNSyncErrors']]],
        [[integrityTaskPaths['TrendNErrors']]]
    ]),
    ecal2P('Number of Digis', occupancyTaskPaths['TrendNDigi']),
    ecal2P('Number of RecHits', occupancyTaskPaths['TrendNRecHitThr']),
    ecal2P('Number of TPs', occupancyTaskPaths['TrendNTPDigi']),
    ecal2P('Presample Mean', presampleClientPaths['TrendMean']),
    ecal2P('Presample RMS', presampleClientPaths['TrendRMS']),
    LayoutElem('Basic Clusters', [
        [[clusterTaskPaths['TrendNBC'] % ebRep], [clusterTaskPaths['TrendBCSize'] % ebRep]],
        [[clusterTaskPaths['TrendNBC'] % eeRep], [clusterTaskPaths['TrendBCSize'] % eeRep]]
    ]),
    LayoutElem('Super Clusters', [
        [[clusterTaskPaths['TrendNSC'] % ebRep], [clusterTaskPaths['TrendSCSize'] % ebRep]],
        [[clusterTaskPaths['TrendNSC'] % eeRep], [clusterTaskPaths['TrendSCSize'] % eeRep]]
    ])
])

superModuleSet = [
    LayoutElem("Integrity", [
        [[integrityClientPaths['Quality']]]
    ]),
    LayoutElem("FEStatus", [
        [[rawDataTaskPaths['FEStatus']]]
    ]),
    LayoutElem("Digi Occupancy", [
        [[occupancyTaskPaths['Digi']]]
    ]),
    LayoutElem("Presample Quality", [
        [[presampleClientPaths['Quality']]]
    ]),
    LayoutElem("Presample Level", [
        [[presampleTaskPaths['Pedestal']]],
        [[presampleClientPaths['Mean']]]
    ]),
    LayoutElem("Noise", [
        [[presampleClientPaths['RMS']]]
    ]),
    LayoutElem("Energy", [
        [[energyTaskPaths['HitMap']]],
        [[energyTaskPaths['HitMap']]]
    ]),
    LayoutElem("Spectrum", [
        [[energyTaskPaths['Hit']]],
        [[energyTaskPaths['Hit']]]
    ]),
    LayoutElem("Timing Quality", [
        [[timingClientPaths['Quality']]]
    ]),
    LayoutElem("Timing", [
        [[timingTaskPaths['TimeMap']]],
        [[timingClientPaths['MeanSM']]]
    ]),
    LayoutElem("Jitter", [
        [[timingClientPaths['RMSMap']]]
    ]),
    LayoutElem("Timing Vs Amplitude", [
        [[timingTaskPaths['TimeAmp']]]
    ]),
    LayoutElem("Trigger Primitives", [
        [[trigPrimTaskPaths['EtRealMap']]],
        [[trigPrimTaskPaths['MatchedIndex']]]
    ]),
    LayoutDir('Laser', [
        LayoutElemSet('Quality L%(wl)s', [
            [[laserClientPaths['Quality']]]
        ], {'wl': laserWavelengths}),
        LayoutElemSet('Amplitude L%(wl)s', [
            [[laserTaskPaths['Amplitude']]],
            [[laserClientPaths['AmplitudeMean']]]
        ], {'wl': laserWavelengths}),
        LayoutElemSet('Timing L%(wl)s', [
            [[laserTaskPaths['Timing']]],
            [[laserClientPaths['TimingMean']], [laserClientPaths['TimingRMS']]]
        ], {'wl': laserWavelengths}),
        LayoutElemSet('APD Over PN L%(wl)s', [
            [[laserTaskPaths['AOverP']]]
        ], {'wl': laserWavelengths}),
        LayoutElemSet('Shape L%(wl)s', [
            [[laserTaskPaths['Shape']]]
        ], {'wl': laserWavelengths})
    ]),
    LayoutDir('Test Pulse', [
        LayoutElemSet('Quality G%(gain)s', [
            [[testPulseClientPaths['Quality']]]
        ], {'gain': mgpaGains}),
        LayoutElemSet('Amplitude G%(gain)s', [
            [[testPulseTaskPaths['Amplitude']]],
            [[testPulseClientPaths['AmplitudeRMS']]]
        ], {'gain': mgpaGains}),
        LayoutElemSet('Shape G%(gain)s', [
            [[testPulseTaskPaths['Shape']]]
        ], {'gain': mgpaGains})
    ])
]
ebSMRep = {'sm': smNamesEB}
ebSMRep.update(ebRep)
layouts['ecal-layouts'].get("By SuperModule").append(
    LayoutDirSet("%(sm)s", superModuleSet, ebSMRep, addSerial = False)
)
superModuleSet.append(
    LayoutDir('Led', [
        LayoutElemSet('Quality L%(wl)s', [
            [[ledClientPaths['Quality']]]
        ], {'wl': ledWavelengths}),
        LayoutElemSet('Amplitude L%(wl)s', [
            [[ledTaskPaths['Amplitude']]],
            [[ledClientPaths['AmplitudeMean']]]
        ], {'wl': ledWavelengths}),
        LayoutElemSet('Timing L%(wl)s', [
            [[ledTaskPaths['Timing']]],
            [[ledClientPaths['TimingMean']]]
        ], {'wl': ledWavelengths}),
        LayoutElemSet('APD Over PN L%(wl)s', [
            [[ledTaskPaths['AOverP']]]
        ], {'wl': ledWavelengths}),
        LayoutElemSet('Shape L%(wl)s', [
            [[ledTaskPaths['Shape']]]
        ], {'wl': ledWavelengths})
    ])
)
eeSMRep = {'sm': smNamesEE}
eeSMRep.update(eeRep)
layouts['ecal-layouts'].get("By SuperModule").append(
    LayoutDirSet('%(sm)s', superModuleSet, eeSMRep, addSerial = False)
)

layouts['ecal_T0_layouts'] = layouts['ecal-layouts'].clone()
layouts['ecal_T0_layouts'].remove('Calibration Summary')
layouts['ecal_T0_layouts'].remove('Selective Readout')
layouts['ecal_T0_layouts'].remove('Laser')
layouts['ecal_T0_layouts'].remove('Led')
layouts['ecal_T0_layouts'].remove('Test Pulse')
layouts['ecal_T0_layouts'].remove('Pedestal')
layouts['ecal_T0_layouts'].remove('Trend')
layouts['ecal_T0_layouts'].remove('Overview/Laser')
layouts['ecal_T0_layouts'].remove('Overview/Laser PN')
layouts['ecal_T0_layouts'].remove('Overview/Test Pulse')
layouts['ecal_T0_layouts'].remove('Overview/Test Pulse PN')
layouts['ecal_T0_layouts'].remove('Overview/Led')
layouts['ecal_T0_layouts'].remove('Overview/Error Trends')
layouts['ecal_T0_layouts'].remove('Occupancy/Laser3')
layouts['ecal_T0_layouts'].remove('Occupancy/Led')
layouts['ecal_T0_layouts'].remove('Occupancy/Test Pulse')
layouts['ecal_T0_layouts'].remove('By SuperModule/%(sm)s/Laser')
layouts['ecal_T0_layouts'].remove('By SuperModule/%(sm)s/Test Pulse')
layouts['ecal_T0_layouts'].remove('By SuperModule/%(sm)s/Pedestal')
layouts['ecal_T0_layouts'].remove('By SuperModule/%(sm)s/Led')

layouts['ecalpriv-layouts'] = layouts['ecal-layouts'].clone()
layouts['ecalpriv-layouts'].insert("By SuperModule",
    LayoutDir("Pedestal", [
        ecal3P('Quality Summary G%(gain)s', pedestalClientPaths['QualitySummary'], rep = {'gain': mgpaGains}),
        ecal3P('Occupancy G%(gain)s', pedestalTaskPaths['Occupancy'], rep = {'gain': mgpaGains}),
        ecal2P('PN Quality Summary G%(pngain)s', pedestalClientPaths['PNQualitySummary'], rep = {'pngain': pnMGPAGains}),
        LayoutDirSet('Gain%(gain)s', [
            LayoutDir('Quality', smSet('Quality', pedestalClientPaths['Quality'])),
            LayoutDir('Pedestal', 
                smSet('Map', pedestalTaskPaths['Pedestal']) +
                smSet('Distributions', pedestalClientPaths['Mean'], pedestalClientPaths['RMS'])
            )
        ], {'gain': mgpaGains}, addSerial = False),
        LayoutDirSet('PNGain%(pngain)s', 
            smSet('Mean', pedestalTaskPaths['PNPedestal']) +
            smSet('RMS', pedestalClientPaths['PNRMS']),
            {'pngain': pnMGPAGains}, addSerial = False)
    ])
)
layouts['ecalpriv-layouts'].get('By SuperModule/%(sm)s').append(
    LayoutDir('Pedestal', [
        LayoutElemSet('Quality G%(gain)s', [
            [[pedestalClientPaths['Quality']]]
        ], {'gain': mgpaGains}),
        LayoutElemSet('Pedestal G%(gain)s', [
            [[pedestalTaskPaths['Pedestal']]],
            [[pedestalClientPaths['Mean']], [pedestalClientPaths['RMS']]]
        ], {'gain': mgpaGains})
    ])
)

#### END ecal-layouts.py / ecal_T0_layouts.py / ecalpriv-layouts.py ####

#### BEGIN ecal_overview_layouts ####

layouts['ecal_overview_layouts'] = LayoutDir("Collisions/EcalFeedBack", [
    LayoutElem("Single Event Timing EB", [
        [[timingTaskPaths['TimeAll'] % ebRep]],
        [[timingClientPaths['FwdBkwdDiff'] % ebRep], [timingClientPaths['FwdvBkwd'] % ebRep]]
    ]),
    LayoutElem("Single Event Timing EE", [
        [[timingTaskPaths['TimeAll'] % eemRep], [timingTaskPaths['TimeAll'] % eepRep]],
        [[timingClientPaths['FwdBkwdDiff'] % eeRep], [timingClientPaths['FwdvBkwd'] % eeRep]]
    ])
])
layouts['ecal_overview_layouts'].append(
    subdetEtaPhi("Timing Map", timingTaskPaths['TimeAllMap'], timingClientPaths['ProjEta'], timingClientPaths['ProjPhi'])
)
layouts['ecal_overview_layouts'].append(
    LayoutElem("Timing ES", [
        [["EcalPreshower/ESTimingTask/ES Timing Z 1 P 1"], ["EcalPreshower/ESTimingTask/ES Timing Z -1 P 1"]],
        [["EcalPreshower/ESTimingTask/ES Timing Z 1 P 2"], ["EcalPreshower/ESTimingTask/ES Timing Z -1 P 2"]]
    ])
)
layouts['ecal_overview_layouts'].append(
    subdetEtaPhi("Occupancy", occupancyTaskPaths['RecHitThrAll'], occupancyTaskPaths['RecHitThrProjEta'], occupancyTaskPaths['RecHitThrProjPhi'])
)
layouts['ecal_overview_layouts'].append([
    LayoutElem("Occupancy ES", [
        [["EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z 1 P 1"], ["EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z -1 P 1"]],
        [["EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z 1 P 2"], ["EcalPreshower/ESOccupancyTask/ES Occupancy with selected hits Z -1 P 2"]]
    ]),
    LayoutElem("RecHit Energy EB", [
        [[energyTaskPaths['HitMapAll'] % ebRep]],
        [[energyTaskPaths['HitAll'] % ebRep]]
    ]),
    LayoutElem("RecHit Energy EE", [
        [[energyTaskPaths['HitMapAll'] % eemRep], [energyTaskPaths['HitMapAll'] % eepRep]],
        [[energyTaskPaths['HitAll'] % eemRep], [energyTaskPaths['HitAll'] % eepRep]]
    ]),
    LayoutElem("RecHit Energy ES", [
        [["EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z 1 P 1"], ["EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z -1 P 1"]],
        [["EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z 1 P 2"], ["EcalPreshower/ESOccupancyTask/ES Energy Density with selected hits Z -1 P 2"]]
    ])
])

#### END ecal_overview_layouts ####

#### BEGIN ecal_relval-layouts / ecalmc_relval-layouts ####

layouts['ecal_relval-layouts'] = LayoutDir("DataLayouts/Ecal", [
    ecal2P("Number of Ecal RecHits", occupancyTaskPaths['RecHitThr1D']),
    LayoutElem("Number of ES RecHits", [
        [["EcalPreshower/ESOccupancyTask/ES Num of RecHits Z 1 P 1"], ["EcalPreshower/ESOccupancyTask/ES Num of RecHits Z -1 P 1"]],
        [["EcalPreshower/ESOccupancyTask/ES Num of RecHits Z 1 P 2"], ["EcalPreshower/ESOccupancyTask/ES Num of RecHits Z -1 P 2"]]
    ]),
    LayoutElem("Ecal RecHit Occupancy Eta", [
        [[occupancyTaskPaths['RecHitThrProjEta'] % eemRep], [occupancyTaskPaths['RecHitThrProjEta'] % ebRep], [occupancyTaskPaths['RecHitThrProjEta'] % eepRep]]
    ]),
    LayoutElem("Ecal RecHit Occupancy Phi", [
        [[occupancyTaskPaths['RecHitThrProjPhi'] % eemRep]],
        [[occupancyTaskPaths['RecHitThrProjPhi'] % ebRep]],
        [[occupancyTaskPaths['RecHitThrProjPhi'] % eepRep]]
    ]),
    ecal3P("Ecal Spectrum", energyTaskPaths['HitAll']),
    LayoutElem("ES Spectrum", [
        [["EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 1"], ["EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 1"]],
        [["EcalPreshower/ESOccupancyTask/ES RecHit Energy Z 1 P 2"], ["EcalPreshower/ESOccupancyTask/ES RecHit Energy Z -1 P 2"]]
    ]),
    LayoutElem("Ecal Max Energy", [
        [["EcalBarrel/EBRecoSummary/recHits_EB_energyMax"]],
        [["EcalEndcap/EERecoSummary/recHits_EEP_energyMax"], ["EcalEndcap/EERecoSummary/recHits_EEM_energyMax"]]
    ]),
    LayoutElem("ES Max Energy", [
        [["EcalPreshower/ESRecoSummary/recHits_ES_energyMax"]]
    ]),
    LayoutElem("Ecal Timing", [
        [["EcalBarrel/EBRecoSummary/recHits_EB_time"]],
        [["EcalEndcap/EERecoSummary/recHits_EEP_time"], ["EcalEndcap/EERecoSummary/recHits_EEM_time"]]
    ]),
    LayoutElem("ES Timing", [
        [["EcalPreshower/ESRecoSummary/recHits_ES_time"]]
    ]),
    LayoutElem("Ecal Chi2", [
        [["EcalBarrel/EBRecoSummary/recHits_EB_Chi2"]],
        [["EcalEndcap/EERecoSummary/recHits_EEP_Chi2"], ["EcalEndcap/EERecoSummary/recHits_EEM_Chi2"]]
    ]),
    LayoutElem("EB SwissCross", [
        [["EcalBarrel/RecoSummary/recHits_EB_E1oE4"]]
    ]),
    LayoutElem("RecHit Flags", [
        [["EcalBarrel/RecoSummary/recHits_EB_recoFlag"]],
        [["EcalEndcap/EERecoSummary/recHits_EE_recoFlag"]]
    ]),
    LayoutElem("ReducedRecHit Flags", [
        [["EcalBarrel/RecoSummary/redRecHits_EB_recoFlag"]],
        [["EcalEndcap/EERecoSummary/redRecHits_EE_recoFlag"]]
    ]),
    LayoutElem("Basic Cluster RecHit Flags", [
        [["EcalBarrel/RecoSummary/basicClusters_recHits_EB_recoFlag"]],
        [["EcalEndcap/EERecoSummary/basicClusters_recHits_EE_recoFlag"]]
    ]),
    ecal2P("Number of Basic Clusters", clusterTaskPaths['BCNum']),
    ecal2P("Number of Super Clusters", clusterTaskPaths['SCNum']),
    ecal2P("Super Cluster Energy", clusterTaskPaths['SCE']),
    LayoutElem("Super Cluster Occupancy Eta", [
        [["EcalBarrel/RecoSummary/superClusters_EB_eta"]],
        [["EcalEndcap/EERecoSummary/superClusters_EE_eta"]]
    ]),
    LayoutElem("Super Cluster Occupancy Phi", [
        [["EcalBarrel/RecoSummary/superClusters_EB_phi"]],
        [["EcalEndcap/EERecoSummary/superClusters_EE_phi"]]
    ]),
    ecal2P("Super Cluster Size (Crystals)", clusterTaskPaths['SCNcrystals']),
    ecal2P("Super Cluster Size (Basic Clusters)", clusterTaskPaths['SCNBCs']),
    LayoutElem("Super Cluster Seed SwissCross", [
        [["EcalBarrel/RecoSummary/superClusters_EB_E1oE4"]]
    ]),
    LayoutElem("Preshower Planes Energy", [
        [["EcalPreshower/ESRecoSummary/esClusters_energy_plane1"], ["EcalPreshower/ESRecoSummary/esClusters_energy_plane2"]],
        [["EcalPreshower/ESRecoSummary/esClusters_energy_ratio"]]
    ])
])

layouts['ecalmc_relval-layouts'] = layouts['ecal_relval-layouts'].clone('MCLayouts/Ecal')

#### END ecal_relval-layouts / ecalmc_relval-layouts ####

#### BEGIN shift_ecal_relval_layout ####

layouts['shift_ecal_relval_layout'] = LayoutDir("00 Shift/Ecal", [
    ecal3P("RecHit Spectra", energyTaskPaths['HitAll']),
    ecal2P("Number of RecHits", occupancyTaskPaths['RecHitThr1D']),
    ecal3P("Mean Timing", timingClientPaths['MeanAll'])
])

#### END shift_ecal_relval_layout ####

for lo in genList:
    filename = lo
    if lo == 'ecalpriv-layouts' :
        filename = 'ecal-layouts'

    output = file(targetDir + '/' + filename + '.py', 'w')
    layouts[lo].expand(output)
    output.close()

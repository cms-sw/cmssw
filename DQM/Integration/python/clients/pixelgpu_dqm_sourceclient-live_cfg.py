#-------------------------------------
#	Pixel DQM Application using New DQM Sources/Clients
#-------------------------------------

#-------------------------------------
#	Standard Python Imports
#-------------------------------------
import os, sys, socket, string

#-------------------------------------
#	Standard CMSSW Imports/Definitions
#-------------------------------------
import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Run3_cff import Run3
process      = cms.Process('PIXELDQMLIVE', Run3)
subsystem    = 'Pixel'
cmssw        = os.getenv("CMSSW_VERSION").split("_")
debugstr     = "### PixelDQM::cfg::DEBUG: "
warnstr      = "### PixelDQM::cfg::WARN: "
errorstr     = "### PixelDQM::cfg::ERROR:"
useOfflineGT = False
useFileInput = False
useMap       = False
unitTest     = 'unitTest=True' in sys.argv

#-------------------------------------
#	Central DQM Stuff imports
#-------------------------------------
from DQM.Integration.config.online_customizations_cfi import *

if useOfflineGT:
    process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
    process.GlobalTag.globaltag = autoCond['run3_data_prompt']
else:
    process.load('DQM.Integration.config.FrontierCondition_GT_cfi')

if unitTest:
    process.load("DQM.Integration.config.unitteststreamerinputsource_cfi")
    from DQM.Integration.config.unitteststreamerinputsource_cfi import options
elif useFileInput:
    process.load("DQM.Integration.config.fileinputsource_cfi")
    from DQM.Integration.config.fileinputsource_cfi import options
else:
    process.load('DQM.Integration.config.inputsource_cfi')
    from DQM.Integration.config.inputsource_cfi import options

process.load('DQM.Integration.config.environment_cfi')

#-------------------------------------
#	Central DQM Customization
#-------------------------------------

if not useFileInput and not options.inputFiles:
    # stream label
    if process.runType.getRunType() == process.runType.hi_run:
        process.source.streamLabel = "streamHIDQMGPUvsCPU"
    else:
        process.source.streamLabel = "streamDQMGPUvsCPU"

process.dqmEnv.subSystemFolder = subsystem
process.dqmSaver.tag = 'PixelGPU'
process.dqmSaver.runNumber = options.runNumber
# process.dqmSaverPB.tag = 'PixelGPU'
# process.dqmSaverPB.runNumber = options.runNumber
process = customise(process)
process.DQMStore.verbose = 0
if not unitTest and not useFileInput and not options.inputFiles:
  if not options.BeamSplashRun :
    process.source.minEventsPerLumi = 100

#-------------------------------------
#	CMSSW/Pixel non-DQM Related Module import
#-------------------------------------
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('FWCore.MessageLogger.MessageLogger_cfi')

#-------------------------------------
#	CMSSW non-DQM Related Module Settings
#-------------------------------------
runType			= process.runType.getRunType()
runTypeName		= process.runType.getRunTypeName()
isCosmicRun		= runTypeName=="cosmic_run" or runTypeName=="cosmic_run_stage1"
isHeavyIon		= runTypeName=="hi_run"
cmssw			= os.getenv("CMSSW_VERSION").split("_")

#-------------------------------------
#	Pixel DQM Tasks and Harvesters import
#-------------------------------------
process.load('DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQM_FirstStep_cff')
process.load('DQM.SiPixelHeterogeneous.SiPixelHeterogenousDQMHarvesting_cff')
process.siPixelTrackComparisonHarvesterAlpaka.topFolderName = cms.string('SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU')

#-------------------------------------
#  User switches for what to monitor
#-------------------------------------
doRecHits  = False
doTracks   = True
doVertices = True

#-------------------------------------
#	Some Settings before Finishing up
#-------------------------------------
if process.runType.getRunType() == process.runType.hi_run:
    process.siPixelPhase1MonitorRawDataASerial.src = 'hltSiPixelDigiErrorsPPOnAASerialSync'
    process.siPixelPhase1MonitorRawDataADevice.src = 'hltSiPixelDigiErrorsPPOnAA'

    process.siPixelPhase1CompareDigiErrorsSoAAlpaka.pixelErrorSrcGPU = 'hltSiPixelDigiErrorsPPOnAA'
    process.siPixelPhase1CompareDigiErrorsSoAAlpaka.pixelErrorSrcCPU = 'hltSiPixelDigiErrorsPPOnAASerialSync'

    process.siPixelRecHitsSoAMonitorSerial.pixelHitsSrc = 'hltSiPixelRecHitsPPOnAASoASerialSync'
    process.siPixelRecHitsSoAMonitorSerial.TopFolderName = 'SiPixelHeterogeneous/PixelRecHitsCPU'

    process.siPixelRecHitsSoAMonitorDevice.pixelHitsSrc = 'hltSiPixelRecHitsPPOnAASoA'
    process.siPixelRecHitsSoAMonitorDevice.TopFolderName = 'SiPixelHeterogeneous/PixelRecHitsGPU'

    process.siPixelPhase1CompareRecHits.pixelHitsReferenceSoA = 'hltSiPixelRecHitsPPOnAASoASerialSync'
    process.siPixelPhase1CompareRecHits.pixelHitsTargetSoA  = 'hltSiPixelRecHitsPPOnAASoA'
    process.siPixelPhase1CompareRecHits.topFolderName = 'SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU'

    process.siPixelTrackSoAMonitorSerial.pixelTrackSrc = 'hltPixelTracksPPOnAASoASerialSync'
    process.siPixelTrackSoAMonitorSerial.topFolderName = 'SiPixelHeterogeneous/PixelTrackCPU'

    process.siPixelTrackSoAMonitorDevice.pixelTrackSrc = 'hltPixelTracksPPOnAASoA'
    process.siPixelTrackSoAMonitorDevice.topFolderName = 'SiPixelHeterogeneous/PixelTrackGPU'

    process.siPixelPhase1CompareTracks.pixelTrackReferenceSoA = 'hltPixelTracksPPOnAASoASerialSync'
    process.siPixelPhase1CompareTracks.pixelTrackTargetSoA = 'hltPixelTracksPPOnAASoA'
    process.siPixelPhase1CompareTracks.topFolderName = 'SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU'

    process.siPixelVertexSoAMonitorSerial.pixelVertexSrc = 'hltPixelVerticesPPOnAASoASerialSync'
    process.siPixelVertexSoAMonitorSerial.beamSpotSrc = 'hltOnlineBeamSpot'
    process.siPixelVertexSoAMonitorSerial.topFolderName = 'SiPixelHeterogeneous/PixelVertexCPU'

    process.siPixelVertexSoAMonitorDevice.pixelVertexSrc = 'hltPixelVerticesPPOnAASoA'
    process.siPixelVertexSoAMonitorDevice.beamSpotSrc = 'hltOnlineBeamSpot'
    process.siPixelVertexSoAMonitorDevice.topFolderName = 'SiPixelHeterogeneous/PixelVertexGPU'

    process.siPixelCompareVertices.pixelVertexReferenceSoA = 'hltPixelVerticesPPOnAASoASerialSync'
    process.siPixelCompareVertices.pixelVertexTargetSoA = 'hltPixelVerticesPPOnAASoA'
    process.siPixelCompareVertices.beamSpotSrc = 'hltOnlineBeamSpot'
    process.siPixelCompareVertices.topFolderName = 'SiPixelHeterogeneous/PixelVertexCompareGPUvsCPU'

else:
    process.siPixelPhase1MonitorRawDataASerial.src = 'hltSiPixelDigiErrorsSerialSync'
    process.siPixelPhase1MonitorRawDataADevice.src = 'hltSiPixelDigiErrors'
    
    process.siPixelPhase1CompareDigiErrorsSoAAlpaka.pixelErrorSrcGPU = 'hltSiPixelDigiErrors'
    process.siPixelPhase1CompareDigiErrorsSoAAlpaka.pixelErrorSrcCPU = 'hltSiPixelDigiErrorsSerialSync'
    
    process.siPixelRecHitsSoAMonitorSerial.pixelHitsSrc = 'hltSiPixelRecHitsSoASerialSync'
    process.siPixelRecHitsSoAMonitorSerial.TopFolderName = 'SiPixelHeterogeneous/PixelRecHitsCPU'
    
    process.siPixelRecHitsSoAMonitorDevice.pixelHitsSrc = 'hltSiPixelRecHitsSoA'
    process.siPixelRecHitsSoAMonitorDevice.TopFolderName = 'SiPixelHeterogeneous/PixelRecHitsGPU'
    
    process.siPixelPhase1CompareRecHits.pixelHitsReferenceSoA = 'hltSiPixelRecHitsSoASerialSync'
    process.siPixelPhase1CompareRecHits.pixelHitsTargetSoA  = 'hltSiPixelRecHitsSoA'
    process.siPixelPhase1CompareRecHits.topFolderName = 'SiPixelHeterogeneous/PixelRecHitsCompareGPUvsCPU'
    
    process.siPixelTrackSoAMonitorSerial.pixelTrackSrc = 'hltPixelTracksSoASerialSync'
    process.siPixelTrackSoAMonitorSerial.topFolderName = 'SiPixelHeterogeneous/PixelTrackCPU'

    process.siPixelTrackSoAMonitorDevice.pixelTrackSrc = 'hltPixelTracksSoA'
    process.siPixelTrackSoAMonitorDevice.topFolderName = 'SiPixelHeterogeneous/PixelTrackGPU'

    process.siPixelPhase1CompareTracks.pixelTrackReferenceSoA = 'hltPixelTracksSoASerialSync'
    process.siPixelPhase1CompareTracks.pixelTrackTargetSoA = 'hltPixelTracksSoA'
    process.siPixelPhase1CompareTracks.topFolderName = 'SiPixelHeterogeneous/PixelTrackCompareGPUvsCPU'
    
    process.siPixelVertexSoAMonitorSerial.pixelVertexSrc = 'hltPixelVerticesSoASerialSync'
    process.siPixelVertexSoAMonitorSerial.beamSpotSrc = 'hltOnlineBeamSpot'
    process.siPixelVertexSoAMonitorSerial.topFolderName = 'SiPixelHeterogeneous/PixelVertexCPU'
    
    process.siPixelVertexSoAMonitorDevice.pixelVertexSrc = 'hltPixelVerticesSoA'    
    process.siPixelVertexSoAMonitorDevice.beamSpotSrc = 'hltOnlineBeamSpot'
    process.siPixelVertexSoAMonitorDevice.topFolderName = 'SiPixelHeterogeneous/PixelVertexGPU'
    
    process.siPixelCompareVertices.pixelVertexReferenceSoA = 'hltPixelVerticesSoASerialSync'
    process.siPixelCompareVertices.pixelVertexTargetSoA = 'hltPixelVerticesSoA'
    process.siPixelCompareVertices.beamSpotSrc = 'hltOnlineBeamSpot'
    process.siPixelCompareVertices.topFolderName = 'SiPixelHeterogeneous/PixelVertexCompareGPUvsCPU'
    
#-------------------------------------
#       Some Debug
#-------------------------------------
process.dump = cms.EDAnalyzer("EventContentAnalyzer")
process.dumpPath = cms.Path(process.dump)

#-------------------------------------
#  Build the monitoring sequence based on flags
#-------------------------------------
monitoring_modules = []

# Mandatory pixel digi error modules
monitoring_modules.append(process.siPixelPhase1MonitorRawDataASerial)
monitoring_modules.append(process.siPixelPhase1MonitorRawDataADevice)
monitoring_modules.append(process.siPixelPhase1CompareDigiErrorsSoAAlpaka)

if doRecHits:
    monitoring_modules.append(process.siPixelRecHitsSoAMonitorDevice)
    monitoring_modules.append(process.siPixelRecHitsSoAMonitorSerial)
    monitoring_modules.append(process.siPixelPhase1CompareRecHits)

if doTracks:
    monitoring_modules.append(process.siPixelTrackSoAMonitorDevice)
    monitoring_modules.append(process.siPixelTrackSoAMonitorSerial)
    monitoring_modules.append(process.siPixelPhase1CompareTracks)

if doVertices:
    monitoring_modules.append(process.siPixelVertexSoAMonitorDevice)
    monitoring_modules.append(process.siPixelVertexSoAMonitorSerial)
    monitoring_modules.append(process.siPixelCompareVertices)

# Always add the comparison harvesting sequence as before
monitoring_modules.append(process.siPixelPhase1RawDataHarvesterSerial)
monitoring_modules.append(process.siPixelPhase1RawDataHarvesterDevice)

if doTracks:
    monitoring_modules.append(process.siPixelTrackComparisonHarvesterAlpaka)

# Now create the path with those modules
process.tasksPath = cms.Path()
for mod in monitoring_modules:
    process.tasksPath *= mod

print(process.tasksPath)
    
#-------------------------------------
#	Pixel DQM Tasks/Clients Sequences Definition
#-------------------------------------

#process.tasksPath = cms.Path(process.monitorpixelSoACompareSourceAlpaka *
#                             process.siPixelHeterogeneousDQMComparisonHarvestingAlpaka)

#-------------------------------------
#	Paths/Sequences Definitions
#-------------------------------------
process.dqmPath = cms.EndPath(process.dqmEnv)
process.dqmPath1 = cms.EndPath(process.dqmSaver)#*process.dqmSaverPB)
process.schedule = cms.Schedule(process.tasksPath,
                                #process.dumpPath,  # for debug
                                process.dqmPath,
                                process.dqmPath1)

#-------------------------------------
#	Scheduling and Process Customizations
#-------------------------------------
process.options = cms.untracked.PSet(
		Rethrow = cms.untracked.vstring(
			"ProductNotFound",
			"TooManyProducts",
			"TooFewProducts"
		)
)
process.options.wantSummary = True

# tracer
#process.Tracer = cms.Service("Tracer")
process = customise(process)
print("Global Tag used:", process.GlobalTag.globaltag.value())
print("Final Source settings:", process.source)

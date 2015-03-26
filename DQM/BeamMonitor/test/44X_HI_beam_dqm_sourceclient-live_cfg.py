import FWCore.ParameterSet.Config as cms

process = cms.Process("BeamMonitor")

#----------------------------
# Event Source
#-----------------------------
process.load("DQM.Integration.test.inputsource_cfi")
process.EventStreamHttpReader.SelectEvents =  cms.untracked.PSet(
    SelectEvents = cms.vstring(
        'HLT_HI*',
        #'HLT_HICentralityVeto',
        #'HLT_HIJet35U_Core',
        #'HLT_HIL1DoubleMuOpen_Core',
        #'HLT_HIMinBiasBSC_Core',
        #'HLT_HIPhoton15_Core',
    )
)

#--------------------------
# Filters
#--------------------------
# HLT Filter
process.load("HLTrigger.special.HLTTriggerTypeFilter_cfi")
# 0=random, 1=physics, 2=calibration, 3=technical
process.hltTriggerTypeFilter.SelectedTriggerType = 1

# L1 Trigger Bit Selection
#process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
#process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')
#process.hltLevel1GTSeed.L1TechTriggerSeeding = cms.bool(True)
#process.hltLevel1GTSeed.L1SeedsLogicalExpression = cms.string('NOT (36 OR 37 OR 38 OR 39)')

#----------------------------
# DQM Live Environment
#-----------------------------
process.load("DQM.Integration.test.environment_cfi")
process.dqmEnv.subSystemFolder = 'BeamMonitor'

import DQMServices.Components.DQMEnvironment_cfi
process.dqmEnvPixelLess = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
process.dqmEnvPixelLess.subSystemFolder = 'BeamMonitor_PixelLess'

#----------------------------
# BeamMonitor
#-----------------------------
process.load("DQM.BeamMonitor.BeamMonitor_cff")
process.load("DQM.BeamMonitor.BeamMonitorBx_cff")
process.load("DQM.BeamMonitor.BeamConditionsMonitor_cff")

process.dqmBeamMonitor.resetEveryNLumi = 5
process.dqmBeamMonitor.resetPVEveryNLumi = 5

# HI only has one bunch
process.dqmBeamMonitorBx.fitEveryNLumi = 50
process.dqmBeamMonitorBx.resetEveryNLumi = 50

#-------------------------------------------------
# GEOMETRY
#-------------------------------------------------
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

#-----------------------------
# Magnetic Field
#-----------------------------
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')

#--------------------------
# Calibration
#--------------------------
process.load("DQM.Integration.test.FrontierCondition_GT_cfi")

# Using offline alignments
#import commands
#from os import environ
#environ["http_proxy"]="http://cmsproxy.cms:3128"
#process.GlobalTag.connect = "frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/CMS_COND_31X_GLOBALTAG"
#dasinfo = eval(commands.getoutput("wget -qO- 'http://vocms115.cern.ch:8304/tier0/express_config?run=&stream=Express'"))
#process.GlobalTag.globaltag=dasinfo[0]['global_tag']
#process.GlobalTag.pfnPrefix=cms.untracked.string('frontier://(proxyurl=http://localhost:3128)(serverurl=http://localhost:8000/FrontierOnProd)(serverurl=http://localhost:8000/FrontierOnProd)(retrieve-ziplevel=0)(failovertoserver=no)/')
#del environ["http_proxy"]

#-----------------------
#  Reconstruction Modules
#-----------------------
## Collision Reconstruction
process.load("Configuration.StandardSequences.RawToDigi_Data_cff")

## Load Heavy Ion Sequence
process.load("Configuration.StandardSequences.ReconstructionHeavyIons_cff") ## HI sequences

# Select events based on the pixel cluster multiplicity
import  HLTrigger.special.hltPixelActivityFilter_cfi
process.multFilter = HLTrigger.special.hltPixelActivityFilter_cfi.hltPixelActivityFilter.clone(
    inputTag  = cms.InputTag('siPixelClusters'),
    minClusters = cms.uint32(150),
    maxClusters = cms.uint32(50000)
    )

process.filter_step = cms.Sequence( process.siPixelDigis
                                   *process.siPixelClusters
                                   #*process.multFilter
                                  )

process.HIRecoForDQM = cms.Sequence( process.siPixelDigis
                                    *process.siPixelClusters
                                    *process.siPixelRecHits
                                    *process.offlineBeamSpot
                                    *process.hiPixelVertices
                                    *process.hiPixel3PrimTracks
                                   )

# use HI pixel tracking and vertexing
process.dqmBeamMonitor.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
process.dqmBeamMonitorBx.BeamFitter.TrackCollection = cms.untracked.InputTag('hiPixel3PrimTracks')
process.dqmBeamMonitor.primaryVertex = cms.untracked.InputTag('hiSelectedVertex')
process.dqmBeamMonitor.PVFitter.VertexCollection = cms.untracked.InputTag('hiSelectedVertex')

# Beamspot DQM options
process.dqmBeamMonitor.OnlineMode = True                  ## in MC the LS are not ordered??
process.dqmBeamMonitor.BeamFitter.MinimumTotalLayers = 3   ## using pixel triplets
process.dqmBeamMonitorBx.BeamFitter.MinimumTotalLayers = 3   ## using pixel triplets

# make pixel vertexing less sensitive to incorrect beamspot
process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.originRadius = 0.2
process.hiPixel3ProtoTracks.RegionFactoryPSet.RegionPSet.fixedError = 0.5
process.hiSelectedProtoTracks.maxD0Significance = 100
process.hiPixelAdaptiveVertex.TkFilterParameters.maxD0Significance = 100
process.hiPixelAdaptiveVertex.useBeamConstraint = False
process.hiPixelAdaptiveVertex.PVSelParameters.maxDistanceToBeam = 1.0

# Change Beam Monitor variables
if process.dqmSaver.producer.value() is "Playback":
  process.dqmBeamMonitor.BeamFitter.WriteAscii = False
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmdev/BeamMonitorDQM/BeamFitResults.txt'
else:
  process.dqmBeamMonitor.BeamFitter.WriteAscii = True
  process.dqmBeamMonitor.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.txt'
  process.dqmBeamMonitor.BeamFitter.WriteDIPAscii = True
  process.dqmBeamMonitor.BeamFitter.DIPFileName = '/nfshome0/dqmpro/BeamMonitorDQM/BeamFitResults.txt'
#process.dqmBeamMonitor.BeamFitter.SaveFitResults = False
#process.dqmBeamMonitor.BeamFitter.OutputFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults.root'
  process.dqmBeamMonitorBx.BeamFitter.WriteAscii = True
  process.dqmBeamMonitorBx.BeamFitter.AsciiFileName = '/nfshome0/yumiceva/BeamMonitorDQM/BeamFitResults_Bx.txt'

#process.dqmBeamMonitor.BeamFitter.InputBeamWidth = 0.006
# Lower for HI
process.dqmBeamMonitor.PVFitter.minNrVerticesForFit = 20
process.dqmBeamMonitorBx.PVFitter.minNrVerticesForFit = 20

## TKStatus
process.dqmTKStatus = cms.EDAnalyzer("TKStatus",
        BeamFitter = cms.PSet(
        DIPFileName = process.dqmBeamMonitor.BeamFitter.DIPFileName
        )
)


#--------------------------
# Scheduling
#--------------------------
#process.phystrigger = cms.Sequence(process.hltTriggerTypeFilter*process.gtDigis*process.hltLevel1GTSeed)
process.dqmcommon = cms.Sequence(process.dqmEnv
                                 *process.dqmSaver)


process.monitor = cms.Sequence(process.dqmBeamMonitor
                               #+process.dqmBeamMonitorBx
                               )

process.hi = cms.Path(process.scalersRawToDigi
                     *process.dqmTKStatus
                     *process.hltTriggerTypeFilter
                     *process.filter_step
                     *process.HIRecoForDQM
                     *process.dqmcommon
                     *process.monitor)

#
# Last update: new version for python
#
#
import FWCore.ParameterSet.Config as cms

process = cms.Process("T")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
       
process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('TestPixTracks'),
    destinations = cms.untracked.vstring('cout'),
#    destinations = cms.untracked.vstring("log","cout"),
    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
        threshold = cms.untracked.string('ERROR')
    )
#    log = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG')
#    )
)

import HLTrigger.HLTfilters.hltHighLevel_cfi as hlt
# accept if 'path_1' succeeds
process.hltfilter = hlt.hltHighLevel.clone(
# Min-Bias
#   HLTPaths = ['HLT_Physics_v*'],
#    HLTPaths = ['HLT_Random_v*'],
    HLTPaths = ['HLT_ZeroBias*'],
#     HLTPaths = ['HLT_L1Tech54_ZeroBias*'],
#     HLTPaths = ['HLT_L1Tech53_MB*'],
# Commissioning:
#    HLTPaths = ['HLT_L1_Interbunch_BSC_v*'],
#    HLTPaths = ['HLT_L1_PreCollisions_v*'],
#    HLTPaths = ['HLT_BeamGas_BSC_v*'],
#    HLTPaths = ['HLT_BeamGas_HF_v*'],
#    HLTPaths = ['p*'],
#    HLTPaths = ['path_?'],
    andOr = True,  # False = and, True=or
    throw = False
    )


# to select PhysicsBit
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'


process.source = cms.Source("PoolSource",
 fileNames =  cms.untracked.vstring(
#    'file:../../../SimTracker/SiPixelDigitizer/test/tracks.root'
    'file:/afs/cern.ch/work/d/dkotlins/public/MC/mu/pt100/tracks/tracks1.root'
    )
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string('histo_tracks.root')
)

process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

# needed for global transformation
# process.load("Configuration.StandardSequences.FakeConditions_cff")

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")# Choose the global tag here:
process.GlobalTag.globaltag = "MC_70_V1::All"
# 2012
#process.GlobalTag.globaltag = "GR_P_V40::All"
#process.GlobalTag.globaltag = "GR_P_V28::All" # A&B
# 2011
# process.GlobalTag.globaltag = "GR_P_V20::All"
# process.GlobalTag.globaltag = "GR_R_311_V2::All"
# process.GlobalTag.globaltag = "GR_R_310_V2::All"
# for 2010
# process.GlobalTag.globaltag = 'GR10_P_V5::All'
# process.GlobalTag.globaltag = 'GR10_P_V4::All'
# OK for 2009 LHC data
# process.GlobalTag.globaltag = 'CRAFT09_R_V4::All'

process.d = cms.EDAnalyzer("TestWithTracks",
    Verbosity = cms.untracked.bool(False),
    src = cms.InputTag("generalTracks"),
#     PrimaryVertexLabel = cms.untracked.InputTag("offlinePrimaryVertices"),                             
#     trajectoryInput = cms.string("TrackRefitterP5")
#     trajectoryInput = cms.string('cosmictrackfinderP5')
)

#process.p = cms.Path(process.hltPhysicsDeclared*process.hltfilter*process.d)
# process.p = cms.Path(process.hltPhysicsDeclared*process.d)
process.p = cms.Path(process.d)




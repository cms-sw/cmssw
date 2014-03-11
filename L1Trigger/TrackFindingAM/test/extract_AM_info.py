#########
#
# Example script to run the extractor on events containing
# the L1 tracking info produced with the full geometry
# and official software
#
# This script extract the AM pattern reco output just the pattern 
# info or the full stuff if available
#
# Usage: cmsRun extract_AM_info.py
#
# More info:
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.HLLHCTuto620
#
# Look at part 6.2.1 of the tutorial
#
# Author: S.Viret (viret@in2p3.fr)
# Date  : 27/02/2014
#
# Script tested with release CMSSW_6_2_0_SLHC8
#
#########


import FWCore.ParameterSet.Config as cms

process = cms.Process("MIBextractor")

process.load('Configuration/StandardSequences/Services_cff')
process.load('Configuration/StandardSequences/EndOfProcess_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load('SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')


# Other statements

# Global tag for PromptReco
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:upgradePLS3', '')

process.options = cms.untracked.PSet(
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

# Number of events
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# The file you want to extract
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:AM_output.root'),
                            #fileNames = cms.untracked.vstring('file:AMFIT_output.root'),
                            duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)

# Load the extracto
process.load("Extractors.RecoExtractor.MIB_extractor_cff")

# Tune some options (see MIB_extractor_cfi.py for details)

#process.MIBextraction.doPixel          = True
process.MIBextraction.doMatch          = True
process.MIBextraction.doMC             = True

process.MIBextraction.doSTUB           = True
# You can choose to extract the info from filtered stubs only
#process.MIBextraction.STUB_container   = cms.string( "MergePROutput" )
#process.MIBextraction.STUB_name        = cms.string( "StubInPattern" )
process.MIBextraction.CLUS_container   = cms.string( "TTStubsFromPixelDigis")
process.MIBextraction.CLUS_name        = cms.string( "ClusterAccepted" )

process.MIBextraction.doL1TRK          = True
process.MIBextraction.L1pattern_tag    = cms.InputTag( "MergePROutput", "AML1Patterns")

# Choose the first line if you have only the patterns 
process.MIBextraction.L1track_tag      = cms.InputTag( "", "")
#process.MIBextraction.L1track_tag      = cms.InputTag( "MergeFITOutput", "AML1Tracks")

process.p = cms.Path(process.MIBextraction)



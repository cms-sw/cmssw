#########################
#
# Configuration file for L1 hough fit
# using a file with AMPR content 
#
# This script works on any official production sample
# (assuming that this sample contains a container of TTStubs,
# a container of TTClusters, and a container of TrackingParticles)
#
# And of course, a container of patterns.... (TTTracks) 
# Instruction to run this script are provided on this page:
#
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.HLLHCTuto
#
# Look at STEP VI
#
# Author: S.Viret (viret@in2p3.fr)
# Date        : 20/02/2014
#
# Script tested with release CMSSW_6_2_0_SLHC7
#
#########################

import FWCore.ParameterSet.Config as cms

process = cms.Process('AMFITBASE')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5DReco_cff')
process.load('Configuration.Geometry.GeometryExtendedPhase2TkBE5D_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('L1Trigger.TrackFindingAM.L1AMTrack_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('L1Trigger.TrackTrigger.TrackTrigger_cff')
process.load("Extractors.RecoExtractor.MIB_extractor_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(NEVTS)
)

# Input source
#
# You can use as input file the result of the script AMPR_test.py of part 5.2.2 of the tutorial
#
# Any other EDM file containing patterns and produced with CMSSW 620_SLHC7 should also work
#

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('INPUTFILENAME'),     
                            duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)

# Additional output definition
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'MYGLOBALTAG', '')

# The name of the stub container over which the association is done, please note that the filtered cluster container is
# not associated due to the lack of simPixelDigis in official samples

#process.TTStubAssociatorFromPixelDigis.TTStubs        = cms.VInputTag( cms.InputTag("MergeFITOutput", "StubInTrack"))
#process.TTStubAssociatorFromPixelDigis.TTClusterTruth = cms.VInputTag( cms.InputTag("TTClusterAssociatorFromPixelDigis","ClusterAccepted"))


process.RAWSIMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string('OUTPUTFILENAME'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM')
    )
)

# For the moment need to explicitely keep the following containers
# (not yet in the customizing scripts)

process.MIBextraction.doMatch          = True
process.MIBextraction.doMC             = True
process.MIBextraction.doSTUB           = True
process.MIBextraction.doL1TRK          = True

process.MIBextraction.L1pattern_tag    = cms.InputTag( "MergePROutput", "AML1Patterns")
process.MIBextraction.L1track_tag      = cms.InputTag( "MergeFITOutput", "AML1Tracks")
process.MIBextraction.CLUS_container   = cms.string( "TTStubsFromPixelDigis")
process.MIBextraction.CLUS_name        = cms.string( "ClusterAccepted" )

# Keep the PR output
process.RAWSIMoutput.outputCommands.append('keep  *_*_*_AMPRBASE')

# Keep the FIT output
process.RAWSIMoutput.outputCommands.append('keep  *_*_*_AMFITBASE')
process.RAWSIMoutput.outputCommands.append('drop *_TTTracksFromPattern_*_*')
process.RAWSIMoutput.outputCommands.append('keep  *_*_MergedTrackTruth_*')

# Path and EndPath definitions
process.L1AMFIT_step         = cms.Path(process.TTTracksFromPatternswStubs)
process.p                    = cms.Path(process.MIBextraction)
process.endjob_step          = cms.EndPath(process.endOfProcess)
process.RAWSIMoutput_step    = cms.EndPath(process.RAWSIMoutput)

process.schedule = cms.Schedule(process.L1AMFIT_step,process.p,process.endjob_step,process.RAWSIMoutput_step)

# Automatic addition of the customisation function

from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise as customiseBE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import l1EventContent as customise_ev_BE5D

process=customiseBE5D(process)
process=customise_ev_BE5D(process)

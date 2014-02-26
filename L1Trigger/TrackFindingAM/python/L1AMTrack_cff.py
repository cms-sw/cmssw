import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.MagneticField_cff import *
from L1Trigger.TrackFindingAM.L1AMTrack_cfi import *
from SimTracker.TrackTriggerAssociation.TTStubAssociation_cfi import * 
from SimTracker.TrackTriggerAssociation.TTClusterAssociation_cfi import *
import FWCore.ParameterSet.Config as cms


############################################
# STEP 1: AM-based pattern recognition 
############################################

# The simple sequence, creates only a pattern container
# used in principle only for multi-bank PR
#
# Indeed in this case we create the filtered stub container after merging all
# the pattern container (merging the filtered stub container is not possible
# due to persistency loss)

TTPatternsFromStubs   = cms.Sequence(TTPatternsFromStub)


# The complete sequence, creates the pattern container and the 
# container of filtered stubs/clusters, with corresponding
# associators containers

TTPatternsFromStubswStubs   = cms.Sequence(TTPatternsFromStub*MergePROutput*TTClusterAssociatorFromPixelDigis*TTStubAssociatorFromPixelDigis)



############################################
# STEP 2: Hough transform fit
############################################

# The simple sequence, creates only a track container
# used in principle only for debugging purposes
#
TTTracksFromPatterns  = cms.Sequence(TTTracksFromPattern)


# The sequence. Note that we call the Merge plugins because the filtered containers are created
# here. We just merge one branch...

TTTracksFromPatternswStubs   = cms.Sequence(TTTracksFromPattern*MergeFITOutput*TTClusterAssociatorFromPixelDigis*TTStubAssociatorFromPixelDigis)


############################################
# STEP 3: MERGE PR outputs
############################################

# This sequence is used mainly the multi-bank merging process

TTClusterAssociatorFromPixelDigis.TTClusters  = cms.VInputTag( cms.InputTag("MergePROutput", "ClusInPattern"))
TTStubAssociatorFromPixelDigis.TTStubs        = cms.VInputTag( cms.InputTag("MergePROutput", "StubInPattern"))
TTStubAssociatorFromPixelDigis.TTClusterTruth = cms.VInputTag( cms.InputTag("TTClusterAssociatorFromPixelDigis","ClusInPattern"))

MergePROutputs  = cms.Sequence(MergePROutput*TTClusterAssociatorFromPixelDigis*TTStubAssociatorFromPixelDigis)



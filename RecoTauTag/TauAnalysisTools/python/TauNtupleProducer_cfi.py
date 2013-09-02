import FWCore.ParameterSet.Config as cms

from RecoTauTag.TauAnalysisTools.truthTools_cfi import *
from RecoTauTag.TauAnalysisTools.tools.ntupleDefinitions import *

#buildTauNtuple = cms.Sequence(makeTruthCollections)

# Match to GenJets composing visible decay products of hadronic taus
protoMatcher = cms.EDProducer("TrivialDeltaRViewMatcher",
      src = cms.InputTag("shrinkingConePFTauProducer"),
      matched = cms.InputTag("trueHadronicTaus"),
      distMin = cms.double(0.1)
)

protoRecoTauNtupleProducer = cms.EDProducer(
    "TauNtupleProducer",
    # Input collection
    source = cms.InputTag("shrinkingConePFTauProducer"),

    # Prefix to append to the branch alias.  Fields are delimited by a hash
    #  i.e. Events->Scan('shrinking#pt')
    alias = cms.string("shrinking"),

    # Whether or not to match (or require unmatched to truth)
    matchingOption = cms.string("matched"), # 'none', 'matched', 'unmatched'

    # If matched or unmatched is specified
    matchingSource = cms.InputTag("protoMatcher"),
    matchedType = cms.string("Candidate"),

    # Iff option is 'matched', evaluate the following expressions on the truth 
    #  collection
    matched_expressions = cms.PSet(
        pt = cms.string("pt()"),
        eta = cms.string("eta()"),
        mass = cms.string("mass()"),
        charge = cms.string("charge()"),
    ),

    # The C++ class type
    dataType = cms.string("PFTau"),

    # StringFunction expressions to evaluate on each candidate
    #   see SWGuidePhysicsCutParser
    expressions = cms.PSet(
        pt       = cms.string("pt()"),
        nTrks  = cms.string("signalTracks().size()")
    ),

    # Make branches with discriminator results for PF/CaloTaus
    discriminators = cms.PSet(
        ByLeadTrackPt    = cms.InputTag("shrinkingConePFTauDiscriminationByLeadingTrackPtCut"),
    ),

    # Make branches using the associated decay modes.  Will only run when
    # 'source' parameter is present
    decayModeExpressions = cms.PSet(
        #source = cms.InputTag("shrinkingConePFTauDecayModeProducer"),
        mass = cms.string("mass()")
    )
)


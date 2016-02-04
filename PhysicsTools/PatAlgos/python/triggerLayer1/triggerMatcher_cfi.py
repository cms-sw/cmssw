import FWCore.ParameterSet.Config as cms

# Examples for configurations of the trigger match for various physics objects
#
# A detailed description is given in
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#PATTriggerMatcher
# Cuts on the parameters
# - 'maxDPtRel' and
# - 'maxDeltaR'
# are NOT tuned (using old values from TQAF MC match, January 2008)


## Default example matches ##

# firing trigger objects used in succeeding HLT path 'HLT_Mu20'
cleanMuonTriggerMatchHLTMu20 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Mu20_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_DoubleMu6'
cleanMuonTriggerMatchHLTDoubleMu6 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleMu6_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Photon26_IsoVL_Photon18'
cleanPhotonTriggerMatchHLTPhoton26IsoVLPhoton18 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatPhotons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Photon26_IsoVL_Photon18_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT'
cleanElectronTriggerMatchHLTEle27CaloIdVTCaloIsoTTrkIdTTrkIsoT = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatElectrons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Ele27_CaloIdVT_CaloIsoT_TrkIdT_TrkIsoT_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_DoubleIsoPFTau20_Trk5'
cleanTauTriggerMatchHLTDoubleIsoPFTau20Trk5 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatTaus" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleIsoPFTau20_Trk5_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Jet240'
cleanJetTriggerMatchHLTJet240 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'cleanPatJets' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Jet240_v*" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_MET100'
metTriggerMatchHLTMET100 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'patMETs' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_MET100_v*" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)


triggerMatchingDefaultSequence = cms.Sequence(
  cleanMuonTriggerMatchHLTMu20
+ cleanMuonTriggerMatchHLTDoubleMu6
+ cleanPhotonTriggerMatchHLTPhoton26IsoVLPhoton18
+ cleanElectronTriggerMatchHLTEle27CaloIdVTCaloIsoTTrkIdTTrkIsoT
+ cleanTauTriggerMatchHLTDoubleIsoPFTau20Trk5
+ cleanJetTriggerMatchHLTJet240
+ metTriggerMatchHLTMET100
)


## Further examples ##

# L1 e/gammas by original collection
cleanElectronTriggerMatchL1EGammaCollection = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                  # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'cleanPatElectrons' )
, matched = cms.InputTag( 'patTrigger' )        # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'coll( "l1extraParticles:NonIsolated" ) || coll( "l1extraParticles:Isolated" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )      # only one match per trigger object
, resolveByMatchQuality = cms.bool( False )     # take first match found per reco object
)

# L1 and HLT muons by ID
cleanMuonTriggerMatchTriggerMuon = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"           # match by DeltaR and DeltaPt, best match by DeltaR
, src     = cms.InputTag( 'cleanPatMuons' )
, matched = cms.InputTag( 'patTrigger' )    # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'type( "TriggerL1Mu" ) || type( "TriggerMuon" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )  # only one match per trigger object
, resolveByMatchQuality = cms.bool( False ) # take first match found per reco object
)

# firing trigger objects used in succeeding HLT paths of PD /SingleMu
cleanMuonTriggerMatchPDSingleMu = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR and DeltaPt, best match by DeltaR
, src     = cms.InputTag( 'cleanPatMuons' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_IsoMu12_v*" ) || path( "HLT_IsoMu15_v*" ) || path( "HLT_IsoMu17_v*" ) || path( "HLT_IsoMu24_v*" ) || path( "HLT_IsoMu30_v*" ) || path( "HLT_L1SingleMu10_v*" ) || path( "HLT_L1SingleMu20_v*" ) || path( "HLT_L2Mu10_v*" ) || path( "HLT_L2Mu20_v*" ) || path( "HLT_Mu3_v*" ) || path( "HLT_Mu5_v*" ) || path( "HLT_Mu8_v*" ) || path( "HLT_Mu12_v*" ) || path( "HLT_Mu15_v*" ) || path( "HLT_Mu20_v*" ) || path( "HLT_Mu24_v*" ) || path( "HLT_Mu30_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# all trigger objects used in HLT path 'HLT_Mu20' (fake MET)
metTriggerMatchHLTMu20 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'patMETs' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Mu20_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

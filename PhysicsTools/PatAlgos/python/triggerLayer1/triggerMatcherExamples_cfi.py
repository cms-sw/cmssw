import FWCore.ParameterSet.Config as cms

# Examples for configurations of the trigger match for various physics objects
#
# A detailed description is given in
# https://twiki.cern.ch/twiki/bin/view/CMS/SWGuidePATTrigger#PATTriggerMatcher
# Cuts on the parameters
# - 'maxDPtRel' and
# - 'maxDeltaR'
# are NOT tuned (using old values from TQAF MC match, January 2008)


## Example matches ##

# firing trigger objects used in succeeding HLT path 'HLT_Mu17'
somePatMuonTriggerMatchHLTMu17 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Mu17_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_DoubleMu5_IsoMu5'
somePatMuonTriggerMatchHLTDoubleMu5IsoMu5 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleMu5_IsoMu5_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Photon26_Photon18'
somePatPhotonTriggerMatchHLTPhoton26Photon18 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatPhotons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Photon26_Photon18_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL'
somePatElectronTriggerMatchHLTEle17CaloIdTCaloIsoVLTrkIdVLTrkIsoVL = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatElectrons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Ele17_CaloIdT_CaloIsoVL_TrkIdVL_TrkIsoVL_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1'
somePatTauTriggerMatchHLTDoubleMediumIsoPFTau30Trk1eta2p1 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatTaus" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleMediumIsoPFTau30_Trk1_eta2p1_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_PFJet40'
somePatJetTriggerMatchHLTPFJet40 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'selectedPatJets' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_PFJet40_v*" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_MET120'
somePatMetTriggerMatchHLTMET120 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'patMETs' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_MET120_v*" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Mu8_DiJet30' (x-trigger)
somePatMuonTriggerMatchHLTMu8DiJet30 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'type( "TriggerMuon" ) && path( "HLT_Mu8_DiJet30_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)
somePatJetTriggerMatchHLTMu8DiJet30 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "selectedPatJets" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'type( "TriggerJet" ) && path( "HLT_Mu8_DiJet30_v*" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)


_exampleTriggerMatchers = [ 'somePatMuonTriggerMatchHLTMu17'
                          , 'somePatMuonTriggerMatchHLTDoubleMu5IsoMu5'
                          , 'somePatPhotonTriggerMatchHLTPhoton26Photon18'
                          , 'somePatElectronTriggerMatchHLTEle17CaloIdTCaloIsoVLTrkIdVLTrkIsoVL'
                          , 'somePatTauTriggerMatchHLTDoubleMediumIsoPFTau30Trk1eta2p1'
                          , 'somePatJetTriggerMatchHLTPFJet40'
                          , 'somePatMetTriggerMatchHLTMET120'
                          , 'somePatMuonTriggerMatchHLTMu8DiJet30'
                          , 'somePatJetTriggerMatchHLTMu8DiJet30'
                          ]


## Further examples ##

# L1 e/gammas by original collection
somePatElectronTriggerMatchL1EGammaCollection = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                  # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'selectedPatElectrons' )
, matched = cms.InputTag( 'patTrigger' )        # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'coll( "l1extraParticles:NonIsolated" ) || coll( "l1extraParticles:Isolated" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )      # only one match per trigger object
, resolveByMatchQuality = cms.bool( False )     # take first match found per reco object
)

# L1 and HLT muons by ID
somePatMuonTriggerMatchTriggerMuon = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"           # match by DeltaR and DeltaPt, best match by DeltaR
, src     = cms.InputTag( 'selectedPatMuons' )
, matched = cms.InputTag( 'patTrigger' )    # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'type( "TriggerL1Mu" ) || type( "TriggerMuon" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )  # only one match per trigger object
, resolveByMatchQuality = cms.bool( False ) # take first match found per reco object
)

# firing trigger objects used in succeeding HLT paths of PD /SingleMu
somePatMuonTriggerMatchPDSingleMu = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR and DeltaPt, best match by DeltaR
, src     = cms.InputTag( 'selectedPatMuons' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_RelIso1p0Mu5_v*" ) || path( "HLT_RelIso1p0Mu20_v*" ) || path( "HLT_Mu5_v*" ) || path( "HLT_Mu50_eta2p1_v*" ) || path( "HLT_Mu40_v*" ) || path( "HLT_Mu40_eta2p1_v*" ) || path( "HLT_Mu40_eta2p1_Track60_dEdx3p7_v*" ) || path( "HLT_Mu40_eta2p1_Track50_dEdx3p6_v*" ) || path( "HLT_Mu30_v*" ) || path( "HLT_Mu30_eta2p1_v*" ) || path( "HLT_Mu24_v*" ) || path( "HLT_Mu24_eta2p1_v*" ) || path( "HLT_Mu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v*" ) || path( "HLT_Mu24_CentralPFJet30_CentralPFJet25_v*" ) || path( "HLT_Mu24_CentralPFJet30_CentralPFJet25_v*" ) || path( "HLT_Mu17_eta2p1_TriCentralPFNoPUJet45_35_25_v*" ) || path( "HLT_Mu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v*" ) || path( "HLT_Mu15_eta2p1_v*" ) || path( "HLT_Mu15_eta2p1_TriCentral_40_20_20_v*" ) || path( "HLT_Mu15_eta2p1_TriCentral_40_20_20_DiBTagIP3D1stTrack_v*" ) || path( "HLT_Mu15_eta2p1_TriCentral_40_20_20_BTagIP3D1stTrack_v*" ) || path( "HLT_Mu15_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v*" ) || path( "HLT_Mu12_v*" ) || path( "HLT_Mu12_eta2p1_L1Mu10erJetC12WdEtaPhi1DiJetsC_v*" ) || path( "HLT_Mu12_eta2p1_DiCentral_40_20_v*" ) || path( "HLT_Mu12_eta2p1_DiCentral_40_20_DiBTagIP3D1stTrack_v*" ) || path( "HLT_Mu12_eta2p1_DiCentral_20_v*" ) || path( "HLT_L2Mu70_2Cha_eta2p1_PFMET60_v*" ) || path( "HLT_L2Mu70_2Cha_eta2p1_PFMET55_v*" ) || path( "HLT_IsoMu40_eta2p1_v*" ) || path( "HLT_IsoMu34_eta2p1_v*" ) || path( "HLT_IsoMu30_v*" ) || path( "HLT_IsoMu30_eta2p1_v*" ) || path( "HLT_IsoMu24_v*" ) || path( "HLT_IsoMu24_eta2p1_v*" ) || path( "HLT_IsoMu24_PFJet30_PFJet25_Deta3_CentralPFJet25_v*" ) || path( "HLT_IsoMu24_CentralPFJet30_CentralPFJet25_v*" ) || path( "HLT_IsoMu24_CentralPFJet30_CentralPFJet25_PFMET20_v*" ) || path( "HLT_IsoMu20_eta2p1_v*" ) || path( "HLT_IsoMu20_eta2p1_CentralPFJet80_v*" ) || path( "HLT_IsoMu20_WCandPt80_v*" ) || path( "HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet45_35_25_v*" ) || path( "HLT_IsoMu17_eta2p1_TriCentralPFNoPUJet30_v*" ) || path( "HLT_IsoMu17_eta2p1_DiCentralPFNoPUJet30_v*" ) || path( "HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_v*" ) || path( "HLT_IsoMu17_eta2p1_CentralPFNoPUJet30_BTagIPIter_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# all trigger objects used in HLT path 'HLT_Mu17' (fake MET)
somePatMetTriggerMatchHLTMu17 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'patMETs' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Mu17_v*" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

triggerMatcherExamplesTask = cms.Task(
    somePatMuonTriggerMatchHLTMu17,
    somePatMuonTriggerMatchHLTDoubleMu5IsoMu5,
    somePatPhotonTriggerMatchHLTPhoton26Photon18,
    somePatElectronTriggerMatchHLTEle17CaloIdTCaloIsoVLTrkIdVLTrkIsoVL,
    somePatTauTriggerMatchHLTDoubleMediumIsoPFTau30Trk1eta2p1,
    somePatJetTriggerMatchHLTPFJet40,
    somePatMetTriggerMatchHLTMET120,
    somePatMuonTriggerMatchHLTMu8DiJet30,
    somePatJetTriggerMatchHLTMu8DiJet30,
    somePatElectronTriggerMatchL1EGammaCollection,
    somePatMuonTriggerMatchTriggerMuon,
    somePatMuonTriggerMatchPDSingleMu,
    somePatMetTriggerMatchHLTMu17
)

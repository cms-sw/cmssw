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

# firing trigger objects used in succeeding HLT path 'HLT_Mu9'
cleanMuonTriggerMatchHLTMu9 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Mu9" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_DoubleMu3'
cleanMuonTriggerMatchHLTDoubleIsoMu3 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatMuons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleMu3" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Photon20_Cleaned_L1R'
cleanPhotonTriggerMatchHLTPhoton20CleanedL1R = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatPhotons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Photon20_Cleaned_L1R" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Ele20_SW_L1R'
cleanElectronTriggerMatchHLTEle20SWL1R = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatElectrons" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Ele20_SW_L1R" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_DoubleLooseIsoTau15'
cleanTauTriggerMatchHLTDoubleLooseIsoTau15 = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( "cleanPatTaus" )
, matched = cms.InputTag( "patTrigger" )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleLooseIsoTau15" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_Jet15U'
cleanJetTriggerMatchHLTJet15U = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'cleanPatJets' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Jet15U" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# firing trigger objects used in succeeding HLT path 'HLT_MET45'
metTriggerMatchHLTMET45 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'patMETs' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_MET45" )' )
, maxDPtRel = cms.double( 3.0 )
, maxDeltaR = cms.double( 0.4 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)


triggerMatchingDefaultSequence = cms.Sequence(
  cleanMuonTriggerMatchHLTMu9
+ cleanMuonTriggerMatchHLTDoubleIsoMu3
+ cleanPhotonTriggerMatchHLTPhoton20CleanedL1R
+ cleanElectronTriggerMatchHLTEle20SWL1R
+ cleanTauTriggerMatchHLTDoubleLooseIsoTau15
+ cleanJetTriggerMatchHLTJet15U
+ metTriggerMatchHLTMET45
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
, matchedCuts = cms.string( 'type( "TriggerL1Mu" ) || type( "TriggerMu" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )  # only one match per trigger object
, resolveByMatchQuality = cms.bool( False ) # take first match found per reco object
)

# firing trigger objects used in succeeding HLT paths of PD /Mu
cleanMuonTriggerMatchPDMu = cms.EDProducer(
  "PATTriggerMatcherDRDPtLessByR"                 # match by DeltaR and DeltaPt, best match by DeltaR
, src     = cms.InputTag( 'cleanPatMuons' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_DoubleMu3" ) || path( "HLT_IsoMu3" ) || path( "HLT_L1Mu14_L1ETM30" ) || path( "HLT_L1Mu14_L1SingleEG10" ) || path( "HLT_L1Mu14_L1SingleJet6U" ) || path( "HLT_L1Mu20" ) || path( "HLT_L2Mu11" ) || path( "HLT_L2Mu9" ) || path( "HLT_Mu3" ) || path( "HLT_Mu5" ) || path( "HLT_Mu9" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

# all trigger objects used in HLT path 'HLT_Mu3' (fake MET)
metTriggerMatchHLTMu3 = cms.EDProducer(
  "PATTriggerMatcherDRLessByR"                    # match by DeltaR only, best match by DeltaR
, src     = cms.InputTag( 'patMETs' )
, matched = cms.InputTag( 'patTrigger' )          # default producer label as defined in PhysicsTools/PatAlgos/python/triggerLayer1/triggerProducer_cfi.py
, matchedCuts = cms.string( 'path( "HLT_Mu3" )' )
, maxDPtRel = cms.double( 0.5 )
, maxDeltaR = cms.double( 0.5 )
, resolveAmbiguities    = cms.bool( True )        # only one match per trigger object
, resolveByMatchQuality = cms.bool( True )        # take best match found per reco object: by DeltaR here (s. above)
)

import FWCore.ParameterSet.Config as cms


#RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
#        mix = cms.PSet(initialSeed = cms.untracked.uint32(12345),
#                       engineName = cms.untracked.string('HepJamesRandom')
#        ),
#        restoreStateLabel = cms.untracked.string("randomEngineStateProducer"),
#)

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from Validation.Configuration.globalValidation_cff import *

from HLTriggerOffline.Common.HLTValidation_cff import *


from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.RecoJets.JetValidation_cff import *
from JetMETCorrections.Type1MET.pfMETCorrectionType0_cfi import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
from Validation.RecoMuon.muonValidation_cff import *
from Validation.MuonIsolation.MuIsoVal_cff import *
from Validation.MuonIdentification.muonIdVal_cff import *
from Validation.RecoMuon.muonValidationHLT_cff import *
from Validation.EventGenerator.BasicGenValidation_cff import *
# miniAOD
from Validation.RecoParticleFlow.miniAODValidation_cff import *
from Validation.RecoEgamma.photonMiniAODValidationSequence_cff import *
from Validation.RecoEgamma.egammaValidationMiniAOD_cff import *

prevalidation = cms.Sequence( globalPrevalidation * hltassociation * metPreValidSeq * jetPreValidSeq * cms.SequencePlaceholder("mix") )
prevalidationLiteTracking = cms.Sequence( prevalidation )
prevalidationLiteTracking.replace(globalPrevalidation,globalPrevalidationLiteTracking)
prevalidationMiniAOD = cms.Sequence( genParticles1 * miniAODValidationSequence * photonMiniAODValidationSequence * egammaValidationMiniAOD)


validation = cms.Sequence(
                         genvalid_all
                         *globaldigisanalyze
                         *globalhitsanalyze
                         *globalrechitsanalyze
                         *globalValidation
                         *hltvalidation)

_validation_fastsim = validation.copy()
for _entry in [globaldigisanalyze,globalhitsanalyze,globalrechitsanalyze]:
    _validation_fastsim.remove(_entry)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(validation,_validation_fastsim)

validationLiteTracking = cms.Sequence( validation )
validationLiteTracking.replace(globalValidation,globalValidationLiteTracking)
validationLiteTracking.remove(condDataValidation)

validationMiniAOD = cms.Sequence(type0PFMEtCorrectionPFCandToVertexAssociationForValidationMiniAOD * JetValidationMiniAOD * METValidationMiniAOD)

prevalidation_preprod = cms.Sequence( preprodPrevalidation )

validation_preprod = cms.Sequence(
                          genvalid_all
                          +trackingTruthValid
                          +tracksValidation
                          +METRelValSequence
                          +recoMuonValidation
                          +muIsoVal_seq
                          +muonIdValDQMSeq
                          +hltvalidation_preprod
                          )

validation.remove(condDataValidation)
validation_prod = cms.Sequence(
             genvalid_all
            +hltvalidation_prod
            )


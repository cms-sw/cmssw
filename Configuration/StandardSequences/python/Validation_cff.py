import FWCore.ParameterSet.Config as cms


RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
        mix = cms.PSet(initialSeed = cms.untracked.uint32(12345),
                       engineName = cms.untracked.string('HepJamesRandom')
        ),
        restoreStateLabel = cms.untracked.string("randomEngineStateProducer"),
)

from Validation.GlobalDigis.globaldigis_analyze_cfi import *
from Validation.GlobalRecHits.globalrechits_analyze_cfi import *
from Validation.GlobalHits.globalhits_analyze_cfi import *
from Validation.Configuration.globalValidation_cff import *

from HLTriggerOffline.Common.HLTValidation_cff import *


# for ValidationProd
from SimGeneral.TrackingAnalysis.trackingParticles_cfi import *
from Validation.RecoMET.METRelValForDQM_cff import *
from Validation.TrackingMCTruth.trackingTruthValidation_cfi import *
from Validation.RecoTrack.TrackValidation_cff import *
from Validation.RecoMuon.muonValidation_cff import *
from Validation.MuonIsolation.MuIsoVal_cff import *
from Validation.RecoMuon.muonValidationHLT_cff import *
from Validation.RecoB.bTagAnalysis_cfi import *
bTagValidation.jetMCSrc = 'IC5byValAlgo'
bTagValidation.etaRanges = cms.vdouble(0.0, 1.1, 2.4)
from RecoBTag.SoftLepton.softElectronBJetTags_cfi import  *

# to be changed
from PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi import * 


validation = cms.Sequence(cms.SequencePlaceholder("mix")+globaldigisanalyze*globalhitsanalyze*globalrechitsanalyze*globalValidation*hltvalidation)
validation_prod = cms.Sequence(trackingTruthValid
                          +tracksValidation
                          +METRelValSequence
                          +recoMuonValidation
                          +muIsoVal_seq
                          +myPartons
                          +iterativeCone5Flavour
                          +softElectronBJetTags 
                          +bTagValidation
                          +hltvalidation_prod
                          )

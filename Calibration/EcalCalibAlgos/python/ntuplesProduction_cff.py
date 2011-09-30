import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.metTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *    

def addSimpleNtuple(process,isMC,runFromAlca,recalib):
    
    #--------------------------
    #Define PAT sequence
    #--------------------------
    # Standard PAT Configuration File

    ## remove MC matching from the default sequence
    ## remove tau from the default sequence
    process.load("PhysicsTools.PatAlgos.patSequences_cff")
    process.load("PhysicsTools.PatAlgos.tools.pfTools")
    if (not isMC):
        removeMCMatching(process, ['All'])
        
    if (not runFromAlca):    
        removeSpecificPATObjects( process, ['Taus'] )
        process.patDefaultSequence.remove( process.patTaus )
        # Add tcMET and pfMET
        addTcMET(process, 'TC')
        addPfMET(process, 'PF')
        # Jet energy corrections to use:
        if (not isMC):
            inputJetCorrLabel = ('AK5PF', ['L1FastJet', 'L2Relative', 'L3Absolute', 'L2L3Residual'])
        else:
            inputJetCorrLabel = ('AK5PF', ['L1Fastjet', 'L2Relative', 'L3Absolute'])
            
            process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
            process.load('RecoJets.Configuration.RecoPFJets_cff')
            
            #    from RecoJets.JetProducers.kt4PFJets_cfi import *
            #    process.patJetCorrFactors.rho = cms.InputTag("kt6PFJets","rho")
            
            # Add PF jets
            switchJetCollection(process,
                                cms.InputTag('ak5PFJets'),
                                doJTA        = True,
                                doBTagging   = True,
                                jetCorrLabel = inputJetCorrLabel,
                                doType1MET   = True,
                                genJetCollection=cms.InputTag("ak5GenJets"),
                                doJetID      = True
                                )
            
            process.patJets.addTagInfos = True
            process.patJets.tagInfoSources  = cms.VInputTag( cms.InputTag("secondaryVertexTagInfosAOD") )
            
            # PAT selection layer
            
            process.selectedPatPhotons.cut   = cms.string("pt > 15.")
            process.selectedPatMuons.cut     = cms.string("pt >  5.")
            process.selectedPatJets.cut      = cms.string("pt > 15.")
    else:
        removeAllPATObjectsBut(process, ['Electrons','METs'])
        addPfMET(process, 'PF')
    
    process.selectedPatElectrons.cut = cms.string("pt > 15.")
    
    #--------------------------
    # Ntuple
    #--------------------------
    
    process.load("Calibration/EcalCalibNtuple/simpleNtuple_cfi")
    process.simpleNtuple.useTriggerEvent = cms.untracked.bool(False)
    process.simpleNtuple.saveEle       = cms.untracked.bool(True)
    process.simpleNtuple.saveEleShape  = cms.untracked.bool(True)
    
    process.simpleNtuple.dataFlag      = cms.untracked.bool(True)
    process.simpleNtuple.saveL1        = cms.untracked.bool(True)
    process.simpleNtuple.saveHLT       = cms.untracked.bool(True)

    process.simpleNtuple.savePho       = cms.untracked.bool(True)

    process.simpleNtuple.saveMu        = cms.untracked.bool(True)
    process.simpleNtuple.saveJet       = cms.untracked.bool(True)
    process.simpleNtuple.saveCALOMet   = cms.untracked.bool(True)
    process.simpleNtuple.saveTCMet     = cms.untracked.bool(True)
    process.simpleNtuple.savePFMet     = cms.untracked.bool(True)
    process.simpleNtuple.saveMCPU      = cms.untracked.bool(False)
    process.simpleNtuple.verbosity_    = cms.untracked.bool(False)

    
    #--------------------------
    # paths
    #--------------------------
    
    process.simpleNtuple_step = cms.Sequence(
        process.patDefaultSequence *
        process.simpleNtuple
        )


def addSimpleNtupleEoverP(process,recalib):
    process.load("Calibration/EcalCalibNtuple/simpleNtupleEoverP_cfi")
    process.simpleNtupleEoverP.recHitCollection_EB = cms.InputTag("alCaIsolatedElectrons","alcaBarrelHits")
    process.simpleNtupleEoverP.recHitCollection_EE = cms.InputTag("alCaIsolatedElectrons","alcaEndcapHits")
    if (recalib):
        process.simpleNtupleEoverP.EleTag = cms.InputTag("electronRecalibSCAssociator")
    else:
        process.simpleNtupleEoverP.EleTag = cms.InputTag("gsfElectrons")
        
    process.simpleNtupleEoverP.CALOMetTag = cms.InputTag("met")
        
    process.simpleNtupleEoverP_step = cms.Sequence(
        process.simpleNtupleEoverP
        )

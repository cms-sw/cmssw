import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.tools.tauTools import *

def adaptPFMuons(process,module):
    module.useParticleFlow = True
    print "Temporarily switching off isolation & isoDeposits for PF Muons"
    module.isolation   = cms.PSet()
    module.isoDeposits = cms.PSet()
    pass
def adaptPFElectrons(process,module):
    module.useParticleFlow = True
    print "Temporarily switching off isolation & isoDeposits for PF Electrons"
    module.isolation   = cms.PSet()
    module.isoDeposits = cms.PSet()
    print "Temporarily switching off electron ID for PF Electrons"
    module.isolation   = cms.PSet()
    module.addElectronID = False
    if module.embedTrack.value(): 
        module.embedTrack = False
        print "Temporarily switching off electron track embedding"
    if module.embedGsfTrack.value(): 
        module.embedGsfTrack = False
        print "Temporarily switching off electron gsf track embedding"
    if module.embedSuperCluster.value(): 
        module.embedSuperCluster = False
        print "Temporarily switching off electron supercluster embedding"
def adaptPFPhotons(process,module):
    raise RuntimeError, "Photons are not supported yet"
def adaptPFJets(process,module):
    module.embedCaloTowers   = False

def addPFCandidates(process,src,patLabel='PFParticles',cut=""):
    from PhysicsTools.PatAlgos.producersLayer1.pfParticleProducer_cfi import allLayer1PFParticles
    # make modules
    producer = allLayer1PFParticles.clone(pfCandidateSource = src)
    filter   = cms.EDFilter("PATPFParticleSelector", 
                    src = cms.InputTag('allLayer1' + patLabel), 
                    cut = cms.string(cut))
    counter  = cms.EDFilter("PATCandViewCountFilter",
                    minNumber = cms.uint32(0),
                    maxNumber = cms.uint32(999999),
                    src       = cms.InputTag('selectedLayer1' + patLabel))
    # add modules to process
    setattr(process, 'allLayer1'      + patLabel, producer)
    setattr(process, 'selectedLayer1' + patLabel, filter)
    setattr(process, 'countLayer1'    + patLabel, counter)
    # insert into sequence
    process.allLayer1Objects.replace(process.allLayer1Summary, producer +  process.allLayer1Summary)
    process.selectedLayer1Objects.replace(process.selectedLayer1Summary, filter +  process.selectedLayer1Summary)
    process.countLayer1Objects    += counter
    # summary tables
    process.aodSummary.candidates.append(src)
    process.allLayer1Summary.candidates.append(cms.InputTag('allLayer1' + patLabel))
    process.selectedLayer1Summary.candidates.append(cms.InputTag('selectedLayer1' + patLabel))

def switchToPFMET(process,input=cms.InputTag('pfMET')):
    oldMETSource = process.layer1METs.metSource
    switchMCAndTriggerMatch(process,oldMETSource,input)
    process.layer1METs.metSource = input
    process.layer1METs.addMuonCorrections = False
    #process.patJetMETCorrections.remove(process.patMETCorrections)
    process.patAODExtraReco.remove(process.patMETCorrections)

def usePF2PAT(process,runPF2PAT=True):
    """Switch PAT to use PF2PAT instead of AOD sources. if 'runPF2PAT' is true, we'll also add PF2PAT in front of the PAT sequence"""
    # -------- CORE ---------------
    if runPF2PAT:
        process.load("PhysicsTools.PFCandProducer.PF2PAT_cff")
        process.patAODCoreReco += process.PF2PAT
        # note: I can't just replace it, because other people could have added stuff here (e.g. btagging)
    removeCleaning(process)
    process.aodSummary.candidates = cms.VInputTag();
    
    # -------- OBJECTS ------------
    # Muons
    process.allLayer1Muons.pfMuonSource = cms.InputTag("pfMuons")
    adaptPFMuons(process,process.allLayer1Muons)
    switchMCAndTriggerMatch(process,process.allLayer1Muons.muonSource,process.allLayer1Muons.pfMuonSource)
    process.aodSummary.candidates.append(process.allLayer1Muons.pfMuonSource)
    
    # Electrons
#    process.allLayer1Electrons.pfElectronSource = cms.InputTag("pfElectrons")
#    adaptPFElectrons(process,process.allLayer1Electrons)
#    switchMCAndTriggerMatch(process,process.allLayer1Electrons.electronSource,process.allLayer1Electrons.pfElectronSource)
#    process.aodSummary.candidates.append(process.allLayer1Electrons.pfElectronSource)
#    process.patAODCoreReco.remove(process.electronsNoDuplicates)
#    process.patAODExtraReco.remove(process.patElectronId)
#    process.patAODExtraReco.remove(process.patElectronIsolation)
    print "Temporarily switching off electrons completely"
    removeSpecificPATObject(process,'Electrons')
    process.patAODCoreReco.remove(process.electronsNoDuplicates)
    process.patAODExtraReco.remove(process.patElectronId)
    process.patAODExtraReco.remove(process.patElectronIsolation)
    #process.countLayer1Leptons.countElectrons = False
    
    # Photons
    print "Temporarily switching off photons completely"
    removeSpecificPATObject(process,'Photons')
    process.patAODExtraReco.remove(process.patPhotonIsolation)
    
    # Jets
    switchJetCollection(process, cms.InputTag('pfTopProjection','PFJets'),
        doJTA=True,
        doBTagging=True,
        jetCorrLabel=None, # You may want to apply jet energy corrections
        doType1MET=False)  # You don't want CaloMET with PFJets, do you?
    adaptPFJets(process, process.allLayer1Jets)
    process.aodSummary.candidates.append(process.allLayer1Jets.jetSource)

    # Taus
    oldTaus = process.allLayer1Taus.tauSource
    process.allLayer1Taus.tauSource = cms.InputTag("allLayer0Taus")
    switchMCAndTriggerMatch(process, oldTaus, process.allLayer1Taus.tauSource)
    redoPFTauDiscriminators(process, oldTaus, process.allLayer1Taus.tauSource)
    process.aodSummary.candidates.append(process.allLayer1Taus.tauSource)
    
    # MET
    switchToPFMET(process, cms.InputTag('pfMET'))
    
    # Unmasked PFCandidates
    addPFCandidates(process,cms.InputTag('pfTopProjection','PFCandidates'),patLabel='PFParticles',cut="")

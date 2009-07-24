import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.tools.tauTools import *

def warningIsolation():
    print "WARNING: particle based isolation must be studied"

def adaptPFMuons(process,module):

    
    print "Adapting PF Muons "
    print "***************** "
    warningIsolation()
    print 
    module.useParticleFlow = True
    module.isolation   = cms.PSet()
    module.isoDeposits = cms.PSet(
        pfChargedHadrons = cms.InputTag("isoMuonWithCharged"),
        pfNeutralHadrons = cms.InputTag("isoMuonWithNeutral"),
        pfPhotons = cms.InputTag("isoMuonWithPhotons")
        )
    module.isolationValues = cms.PSet(
        pfChargedHadrons = cms.InputTag("pfMuonIsolationFromDepositsChargedHadrons"),
        pfNeutralHadrons = cms.InputTag("pfMuonIsolationFromDepositsNeutralHadrons"),
        pfPhotons = cms.InputTag("pfMuonIsolationFromDepositsPhotons")
        )
    # matching the pfMuons, not the standard muons.
    switchMCMatch(process,module.muonSource,module.pfMuonSource)
    process.aodSummary.candidates.append(module.pfMuonSource)
    print " muon source:", module.pfMuonSource
    print " isolation  :",
    print module.isolationValues
    print " isodeposits: "
    print module.isoDeposits
    print 
    

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
    print 'MET: using ', input
    oldMETSource = process.layer1METs.metSource
    switchMCMatch(process,oldMETSource,input)
    process.layer1METs.metSource = input
    process.layer1METs.addMuonCorrections = False
    #process.patJetMETCorrections.remove(process.patMETCorrections)
    process.patAODExtraReco.remove(process.patMETCorrections)

def switchToPFJets(process,input=cms.InputTag('pfNoTau')):
    print 'Jets: using ', input
    switchJetCollection(process,
                        input,
                        doJTA=True,
                        doBTagging=True,
                        jetCorrLabel=None, 
                        doType1MET=False)  
    adaptPFJets(process, process.allLayer1Jets)
    process.aodSummary.candidates.append(process.allLayer1Jets.jetSource)

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
    adaptPFMuons(process,process.allLayer1Muons)

    
    # Electrons
#    process.allLayer1Electrons.pfElectronSource = cms.InputTag("isolatedElectrons")
#    adaptPFElectrons(process,process.allLayer1Electrons)
#    switchMCMatch(process,process.allLayer1Electrons.electronSource,process.allLayer1Electrons.pfElectronSource)
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
    switchToPFJets( process, 'pfNoTau' )
    
    # Taus
    #COLIN try to re-enable taus
    removeSpecificPATObject(process,'Taus')
#    oldTaus = process.allLayer1Taus.tauSource
#    process.allLayer1Taus.tauSource = cms.InputTag("allLayer0Taus")
#    switchMCMatch(process, oldTaus, process.allLayer1Taus.tauSource)
#    redoPFTauDiscriminators(process, oldTaus, process.allLayer1Taus.tauSource)
#    process.aodSummary.candidates.append(process.allLayer1Taus.tauSource)
    
    # MET
    switchToPFMET(process, cms.InputTag('pfMET'))
    
    # Unmasked PFCandidates
    addPFCandidates(process,cms.InputTag('pfNoJet'),patLabel='PFParticles',cut="")

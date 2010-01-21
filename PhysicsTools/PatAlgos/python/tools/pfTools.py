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
    module.userIsolation   = cms.PSet()
    module.isoDeposits = cms.PSet(
        pfChargedHadrons = cms.InputTag("isoDepMuonWithCharged"),
        pfNeutralHadrons = cms.InputTag("isoDepMuonWithNeutral"),
        pfPhotons = cms.InputTag("isoDepMuonWithPhotons")
        )
    module.isolationValues = cms.PSet(
        pfChargedHadrons = cms.InputTag("isoValMuonWithCharged"),
        pfNeutralHadrons = cms.InputTag("isoValMuonWithNeutral"),
        pfPhotons = cms.InputTag("isoValMuonWithPhotons")
        )
    # matching the pfMuons, not the standard muons.
    process.muonMatch.src = module.pfMuonSource

    print " muon source:", module.pfMuonSource
    print " isolation  :",
    print module.isolationValues
    print " isodeposits: "
    print module.isoDeposits
    print 
    

def adaptPFElectrons(process,module):
    # module.useParticleFlow = True

    print "Adapting PF Electrons "
    print "********************* "
    warningIsolation()
    print 
    module.useParticleFlow = True
    module.userIsolation   = cms.PSet()
    module.isoDeposits = cms.PSet(
        pfChargedHadrons = cms.InputTag("isoDepElectronWithCharged"),
        pfNeutralHadrons = cms.InputTag("isoDepElectronWithNeutral"),
        pfPhotons = cms.InputTag("isoDepElectronWithPhotons")
        )
    module.isolationValues = cms.PSet(
        pfChargedHadrons = cms.InputTag("isoValElectronWithCharged"),
        pfNeutralHadrons = cms.InputTag("isoValElectronWithNeutral"),
        pfPhotons = cms.InputTag("isoValElectronWithPhotons")
        )

    # COLIN: since we take the egamma momentum for pat Electrons, we must
    # match the egamma electron to the gen electrons, and not the PFElectron.  
    # -> do not uncomment the line below.
    # process.electronMatch.src = module.pfElectronSource
    # COLIN: how do we depend on this matching choice? 

    print " PF electron source:", module.pfElectronSource
    print " isolation  :"
    print module.isolationValues
    print " isodeposits: "
    print module.isoDeposits
    print 
    
    print "removing traditional isolation"
    process.patDefaultSequence.remove(getattr(process, 'patElectronIsolation'))

    


##     print "Temporarily switching off isolation & isoDeposits for PF Electrons"
##     module.isolation   = cms.PSet()
##     module.isoDeposits = cms.PSet()
##     print "Temporarily switching off electron ID for PF Electrons"
##     module.isolation   = cms.PSet()
##     module.addElectronID = False
##     if module.embedTrack.value(): 
##         module.embedTrack = False
##         print "Temporarily switching off electron track embedding"
##     if module.embedGsfTrack.value(): 
##         module.embedGsfTrack = False
##         print "Temporarily switching off electron gsf track embedding"
##     if module.embedSuperCluster.value(): 
##         module.embedSuperCluster = False
##         print "Temporarily switching off electron supercluster embedding"

def adaptPFPhotons(process,module):
    raise RuntimeError, "Photons are not supported yet"

def adaptPFJets(process,module):
    module.embedCaloTowers   = False

from RecoTauTag.RecoTau.TauDiscriminatorTools import adaptTauDiscriminator, producerIsTauTypeMapper 

def reconfigureLayer0Taus(process, 
      tauType='shrinkingConePFTau', 
      layer0Selection=["DiscriminationByIsolation", "DiscriminationByLeadingPionPtCut"],
      selectionDependsOn=["DiscriminationByLeadingTrackFinding"],
      producerFromType=lambda producer: producer+"Producer"):
   print "Layer1Taus will be produced from taus of type: %s that pass %s" \
	 % (tauType, layer0Selection)
 
   # Get the prototype of tau producer to make, i.e. fixedConePFTauProducer
   producerName = producerFromType(tauType)
   # Set as the source for the all layer0 taus selector
   process.allLayer0Taus.src = producerName
   # Start our layer0 base sequence
   process.allLayer0TausBaseSequence = cms.Sequence(getattr(process,
      producerName))
   # Get our prediscriminants
   for predisc in selectionDependsOn:
      # Get the prototype
      originalName = tauType+predisc # i.e. fixedConePFTauProducerDiscriminationByLeadingTrackFinding
      clonedName = "allLayer0TausBase"+predisc
      clonedDisc = getattr(process, originalName).clone()
      # Register in our process
      setattr(process, clonedName, clonedDisc)
      process.allLayer0TausBaseSequence += getattr(process, clonedName)
      # Adapt this discriminator for the cloned prediscriminators 
      adaptTauDiscriminator(clonedDisc, newTauProducer="allLayer0TausBase", 
	    newTauTypeMapper=producerIsTauTypeMapper,
	    preservePFTauProducer=True)
   # Reconfigure the layer0 PFTau selector discrimination sources
   process.allLayer0Taus.discriminators = cms.VPSet()
   for selection in layer0Selection:
      # Get our discriminator that will be used to select layer0Taus
      originalName = tauType+selection
      clonedName = "allLayer0TausBase"+selection
      clonedDisc = getattr(process, originalName).clone()
      # Register in our process
      setattr(process, clonedName, clonedDisc)
      # Adapt our cloned discriminator to the new prediscriminants
      adaptTauDiscriminator(clonedDisc, newTauProducer="allLayer0TausBase",
	    newTauTypeMapper=producerIsTauTypeMapper, preservePFTauProducer=True)
      process.allLayer0TausBaseSequence += clonedDisc
      # Add this selection to our layer0Tau selectors
      process.allLayer0Taus.discriminators.append(cms.PSet(
         discriminator=cms.InputTag(clonedName), selectionCut=cms.double(0.5)))


def adaptPFTaus(process,tauType = 'shrinkingConePFTau' ):
    oldTaus = process.allLayer0Taus.src
    # Set up the collection used as a preselection to use this tau type
    reconfigureLayer0Taus(process, tauType)
    process.allLayer1Taus.tauSource = cms.InputTag("allLayer0Taus")
    redoPFTauDiscriminators(process, cms.InputTag(tauType+'Producer'),
                            process.allLayer1Taus.tauSource,
                            tauType)
    #switchToAnyPFTau(process, oldTaus, process.allLayer1Taus.tauSource, tauType)
    switchToPFTauByType(process, pfTauType=tauType,
	  pfTauLabelNew=process.allLayer1Taus.tauSource,
	  pfTauLabelOld=oldTaus)

    process.makeAllLayer1Taus.remove(process.patPFCandidateIsoDepositSelection)

#helper function for PAT on PF2PAT sample
def tauTypeInPF2PAT(process,tauType='shrinkingConePFTau'): 
    process.load("PhysicsTools.PFCandProducer.pfTaus_cff")
    process.allLayer0Taus.src = cms.InputTag(tauType+'Producer')
            

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
    process.allLayer1Summary.candidates.append(cms.InputTag('allLayer1' + patLabel))
    process.selectedLayer1Summary.candidates.append(cms.InputTag('selectedLayer1' + patLabel))

def switchToPFMET(process,input=cms.InputTag('pfMET')):
    print 'MET: using ', input
    oldMETSource = process.layer1METs.metSource
    process.layer1METs.metSource = input
    process.layer1METs.addMuonCorrections = False
    process.patDefaultSequence.remove(getattr(process, 'makeLayer1METs'))


def switchToPFJets(process,input=cms.InputTag('pfNoTau')):
    print 'Jets: using ', input
    switchJetCollection(process,
                        input,
                        doJTA=True,
                        doBTagging=True,
                        jetCorrLabel=('IC5','PF'), 
                        doType1MET=False)  
    adaptPFJets(process, process.allLayer1Jets)

def usePF2PAT(process,runPF2PAT=True):

    # PLEASE DO NOT CLOBBER THIS FUNCTION WITH CODE SPECIFIC TO A GIVEN PHYSICS OBJECT.
    # CREATE ADDITIONAL FUNCTIONS IF NEEDED. 


    """Switch PAT to use PF2PAT instead of AOD sources. if 'runPF2PAT' is true, we'll also add PF2PAT in front of the PAT sequence"""

    # -------- CORE ---------------
    if runPF2PAT:
        process.load("PhysicsTools.PFCandProducer.PF2PAT_cff")

#        process.dump = cms.EDAnalyzer("EventContentAnalyzer")
        process.patDefaultSequence.replace(process.allLayer1Objects,
                                           process.PF2PAT +
                                           process.allLayer1Objects
                                           )

    removeCleaning(process)
    
    # -------- OBJECTS ------------
    # Muons
    adaptPFMuons(process,process.allLayer1Muons)

    
    # Electrons
    adaptPFElectrons(process,process.allLayer1Electrons)

    # Photons
    print "Temporarily switching off photons completely"
    removeSpecificPATObjects(process,['Photons'])
    process.patDefaultSequence.remove(process.patPhotonIsolation)
    
    # Jets
    switchToPFJets( process, cms.InputTag('pfNoTau') )
    
    # Taus
    #adaptPFTaus( process ) #default (i.e. shrinkingConePFTau)
    adaptPFTaus( process, tauType='fixedConePFTau' )
    
    # MET
    switchToPFMET(process, cms.InputTag('pfMET'))
    
    # Unmasked PFCandidates
    addPFCandidates(process,cms.InputTag('pfNoJet'),patLabel='PFParticles',cut="")

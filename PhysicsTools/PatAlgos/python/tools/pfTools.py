import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.tools.coreTools import *
from PhysicsTools.PatAlgos.tools.jetTools import *
from PhysicsTools.PatAlgos.tools.tauTools import *

def warningIsolation():
    print "WARNING: particle based isolation must be studied"

def adaptPFMuons(process,module ):    
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
    mcMuons = getattr(process,module.genParticleMatch.moduleLabel)
    mcMuons.src = module.pfMuonSource


    

    print " muon source:", module.pfMuonSource
    print " isolation  :",
    print module.isolationValues
    print " isodeposits: "
    print module.isoDeposits
    print 
    

def adaptPFElectrons(process,module,l1Collection=cms.InputTag("allLayer1Electrons")):
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

    if (l1Collection.moduleLabel=="allLayer1Electrons"):
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

from RecoTauTag.RecoTau.TauDiscriminatorTools import adaptTauDiscriminator, producerIsTauTypeMapper 

def reconfigureLayer0Taus(process,moduleL0,
      tauType='shrinkingConePFTau', 
      layer0Selection=["DiscriminationByIsolation", "DiscriminationByLeadingPionPtCut"],
      selectionDependsOn=["DiscriminationByLeadingTrackFinding"],
      producerFromType=lambda producer: producer+"Producer"):
   print "Layer1Taus will be produced from taus of type: %s that pass %s" \
	 % (tauType, layer0Selection)

   # Get the prototype of tau producer to make, i.e. fixedConePFTauProducer
   producerName = producerFromType(tauType)
   # Set as the source for the all layer0 taus selector
   moduleL0.src = producerName
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
   moduleL0.discriminators = cms.VPSet()
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
      moduleL0.discriminators.append(cms.PSet(
         discriminator=cms.InputTag(clonedName), selectionCut=cms.double(0.5)))


def adaptPFTaus(process,module,tauType = 'shrinkingConePFTau',
                l0tauColl=cms.InputTag("allLayer0Taus") ):

    moduleL0 =  getattr(process,l0tauColl.moduleLabel)
    oldTaus = moduleL0.src

    # Set up the collection used as a preselection to use this tau type    
    reconfigureLayer0Taus(process,moduleL0, tauType)
    module.tauSource = l0tauColl
    
    redoPFTauDiscriminators(process, 
                            cms.InputTag(tauType+'Producer'),
                            module.tauSource,
                            tauType,
                            l0tauCollection=l0tauColl)
    
    #switchToAnyPFTau(process, oldTaus, process.allLayer1Taus.tauSource, tauType)
    switchToPFTauByType(process,module, pfTauType=tauType,
                        pfTauLabelNew=module.tauSource,
                        pfTauLabelOld=oldTaus)

    if (l0tauColl.moduleLabel=="allLayer0Taus"):
        process.makeAllLayer1Taus.remove(process.patPFCandidateIsoDepositSelection)
    if (l0tauColl.moduleLabel=="pfLayer0Taus"):
        process.PF2PAT.remove(process.patPFCandidateIsoDepositSelection)

#helper function for PAT on PF2PAT sample
def tauTypeInPF2PAT(process,tauType='shrinkingConePFTau'): 
    process.load("PhysicsTools.PFCandProducer.pfTaus_cff")
    process.allLayer0Taus.src = cms.InputTag(tauType+'Producer')
            

def addPFCandidates(process,src,patLabel='PFParticles',cut="",
                    layer='allLayer1',selected='selectedLayer1',
                    counted='countLayer1'):
    from PhysicsTools.PatAlgos.producersLayer1.pfParticleProducer_cfi import allLayer1PFParticles
    # make modules
    producer = allLayer1PFParticles.clone(pfCandidateSource = src)
    filter   = cms.EDFilter("PATPFParticleSelector", 
                    src = cms.InputTag(layer + patLabel), 
                    cut = cms.string(cut))
    counter  = cms.EDFilter("PATCandViewCountFilter",
                    minNumber = cms.uint32(0),
                    maxNumber = cms.uint32(999999),
                    src       = cms.InputTag(selected + patLabel))
    # add modules to process
    setattr(process, layer      + patLabel, producer)
    setattr(process, selected + patLabel, filter)
    setattr(process, counted    + patLabel, counter)
    # insert into sequence
    if (layer=='allLayer1'):
        process.allLayer1Objects.replace(process.allLayer1Summary, producer +  process.allLayer1Summary)
        process.selectedLayer1Objects.replace(process.selectedLayer1Summary, filter +  process.selectedLayer1Summary)
        process.countLayer1Objects    += counter
        # summary tables
        process.allLayer1Summary.candidates.append(cms.InputTag('allLayer1' + patLabel))
        process.selectedLayer1Summary.candidates.append(cms.InputTag('selectedLayer1' + patLabel))
    if (layer=='pfLayer1'): 
        process.pfLayer1Objects.replace(process.pfLayer1Summary, producer +  process.pfLayer1Summary)
        process.pfSelectedObjects.replace(process.pfselectedLayer1Summary, filter +  process.pfselectedLayer1Summary)
        process.pfCountObjects    += counter

        
def switchToPFMET(process,input=cms.InputTag('pfMET'),metColl=cms.InputTag('layer1METs')):
    print 'MET: using ', input
    module =  getattr(process,metColl.moduleLabel)
    oldMETSource = module.metSource
    module.metSource = input
    module.addMuonCorrections = False
    if (metColl.moduleLabel=='layer1METs'):
        process.patDefaultSequence.remove(process.patMETCorrections)


def switchToPFJets(process,
                   input=cms.InputTag('pfNoTau'), algo='IC5',
                   l1jetColl  = cms.InputTag("allLayer1Jets")
                   ):

    print "Switching to PFJets,  ", algo
    print "************************ "

    if( algo == 'IC5' ):
        genJetCollectionName = 'iterativeCone5GenJetsNoNu'
    elif algo == 'AK5':
        genJetCollectionName = 'ak5GenJetsNoNu'
    else:
        print 'bad jet algorithm:', algo, '! for now, only IC5 and AK5 are allowed. If you need other algorithms, please contact Colin'
        sys.exit(1)
        
    # changing the jet collection in PF2PAT:
    from PhysicsTools.PFCandProducer.Tools.jetTools import jetAlgo
    process.allPfJets = jetAlgo( algo );    
   
    switchJetCollection(process,
                        input,
                        doJTA=True,
                        doBTagging=True,
                        jetCorrLabel=( algo, 'PF' ), 
                        genJetCollection = genJetCollectionName,
                        doType1MET=False,
                        l1jetCollection=l1jetColl
                        )  
    l1jets   = getattr(process,l1jetColl.moduleLabel)
    l1jets.embedCaloTowers   = False
#    l1jets.embedPFCandidates   = True


def usePF2PAT(process,runPF2PAT=True, jetAlgo='IC5'):

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
    switchToPFJets( process, cms.InputTag('pfNoTau'), jetAlgo )
    
    # Taus
    #adaptPFTaus( process ) #default (i.e. shrinkingConePFTau)
    adaptPFTaus( process,process.allLayer1Taus, tauType='fixedConePFTau' )
    
    # MET
    switchToPFMET(process, cms.InputTag('pfMET'))
    
    # Unmasked PFCandidates
    addPFCandidates(process,cms.InputTag('pfNoJet'),patLabel='PFParticles',cut="")


def usePATandPF2PAT(process,runPATandPF2PAT=True, jetAlgo='IC5'):
 if runPATandPF2PAT:
     process.load("PhysicsTools.PFCandProducer.PF2PAT_cff")
     
     #LAYER1
     #  #ELECTRONS
     import PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi
     process.pfLayer1Electrons=PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.allLayer1Electrons.clone()
     adaptPFElectrons(process,process.pfLayer1Electrons, cms.InputTag("pfLayer1Electrons"))

     #  #MUONS
     import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
     process.pfLayer1Muons=PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.allLayer1Muons.clone()
     import PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi
     process.pfMuonMatch=PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi.muonMatch.clone()
     process.pfLayer1Muons.genParticleMatch=cms.InputTag("pfMuonMatch")
     adaptPFMuons(process,process.pfLayer1Muons)

     #  #TAUS
     from PhysicsTools.PFCandProducer.pfTaus_cff import allLayer0Taus
     process.pfLayer0Taus=allLayer0Taus.clone()
     process.pfTauSequence.replace(process.allLayer0Taus,
                                   process.pfLayer0Taus)
     process.pfNoTau.topCollection=cms.InputTag("pfLayer0Taus")
     import PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi     
     process.pfLayer1Taus=PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi.allLayer1Taus.clone()
 

     import  PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi
     process.pfTauMatch =PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauMatch.clone()

     import PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi
     process.pfTauGenJetMatch =PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauGenJetMatch.clone()

     from PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff import *
     process.pfTauIsoDepositPFCandidates=PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFCandidates.clone()
     process.pfTauIsoDepositPFChargedHadrons = PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFChargedHadrons.clone()
     process.pfTauIsoDepositPFNeutralHadrons = PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFNeutralHadrons.clone()
     process.pfTauIsoDepositPFGammas = PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFGammas.clone()

     process.pfLayer1Taus.isoDeposits.pfAllParticles = cms.InputTag("pfTauIsoDepositPFCandidates")
     process.pfLayer1Taus.isoDeposits.pfChargedHadron = cms.InputTag("pfTauIsoDepositPFChargedHadrons")
     process.pfLayer1Taus.isoDeposits.pfNeutralHadron = cms.InputTag("pfTauIsoDepositPFNeutralHadrons")
     process.pfLayer1Taus.isoDeposits.pfGamma = cms.InputTag("pfTauIsoDepositPFGammas")
     process.pfLayer1Taus.userIsolation.pfAllParticles.src = cms.InputTag("pfTauIsoDepositPFCandidates")
     process.pfLayer1Taus.userIsolation.pfChargedHadron.src = cms.InputTag("pfTauIsoDepositPFChargedHadrons")
     process.pfLayer1Taus.userIsolation.pfNeutralHadron.src = cms.InputTag("pfTauIsoDepositPFNeutralHadrons")
     process.pfLayer1Taus.userIsolation.pfGamma.src = cms.InputTag("pfTauIsoDepositPFGammas")
     process.pfLayer1Taus.genParticleMatch = cms.InputTag("pfTauMatch")
     process.pfLayer1Taus.genJetMatch      = cms.InputTag("pfTauGenJetMatch")
     adaptPFTaus( process,process.pfLayer1Taus,tauType='fixedConePFTau', l0tauColl=cms.InputTag("pfLayer0Taus"))

     #  #JETS
     import PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi
     process.pfLayer1Jets=PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi.allLayer1Jets.clone()
     import PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi
     process.pfJetPartonMatch =  PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi.jetPartonMatch.clone()
     process.pfJetGenJetMatch =  PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi.jetGenJetMatch.clone()
     process.pfLayer1Jets.genPartonMatch = cms.InputTag("pfJetPartonMatch")
     process.pfLayer1Jets.genJetMatch    = cms.InputTag("pfJetGenJetMatch")  
     import PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff
     process.pfJetPartonAssociation  = PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff.jetPartonAssociation.clone()
     process.pfJetFlavourAssociation = PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff.jetFlavourAssociation.clone()
     process.pfJetFlavourAssociation.srcByReference = cms.InputTag("pfJetPartonAssociation")
     process.pfLayer1Jets.JetPartonMapSource = cms.InputTag("pfJetFlavourAssociation")
     import PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi
     process.pfJetCorrFactors=PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi.jetCorrFactors.clone()
     process.pfLayer1Jets.jetCorrFactorsSource = cms.VInputTag(cms.InputTag("pfJetCorrFactors") )
     import RecoJets.JetAssociationProducers.ak5JTA_cff
     process.jetTracksAssociatorAtVertexPF = RecoJets.JetAssociationProducers.ak5JTA_cff.ak5JetTracksAssociatorAtVertex.clone()
     import PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff
     process.pfJetCharge = PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff.patJetCharge.clone()
     process.pfJetCharge.src = cms.InputTag("jetTracksAssociatorAtVertexPF")
     
     process.pfLayer1Jets.trackAssociationSource = cms.InputTag("jetTracksAssociatorAtVertexPF")
     process.pfLayer1Jets.jetChargeSource = cms.InputTag("pfJetCharge")
     
     #  #MET
     import PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi
     process.pfLayer1METs = PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi.layer1METs.clone()
     switchToPFMET(process, cms.InputTag('pfMET'),metColl=cms.InputTag('pfLayer1METs'))

     #  #SUMMARY
     process.pfLayer1Summary = cms.EDAnalyzer("CandidateSummaryTable",
                                              logName = cms.untracked.string("pfLayer1Objects|PATSummaryTables"),
                                              candidates = cms.VInputTag(cms.InputTag("pfLayer1Electrons"),
                                                                         cms.InputTag("pfLayer1Muons"),
                                                                         cms.InputTag("pfLayer1Taus"),
                                                                         cms.InputTag("pfLayer1Photons"),
                                                                         cms.InputTag("pfLayer1Jets"),
                                                                         cms.InputTag("pfLayer1METs")
                                                                         )
                                              )

     #  #SEQUENCE

     process.makepflayer1Muons=cms.Sequence(process.pfMuonMatch+
                                            process.pfLayer1Muons)

     process.makepflayer1Taus=cms.Sequence(process.pfTauMatch+
                                           process.pfTauGenJetMatch+
                                           process.pfTauIsoDepositPFCandidates+                                          
                                           process.pfTauIsoDepositPFChargedHadrons+
                                           process.pfTauIsoDepositPFNeutralHadrons+
                                           process.pfTauIsoDepositPFGammas+
                                           process.pfLayer1Taus)

     process.makepflayer1Jets=cms.Sequence(process.pfJetPartonMatch+
                                           process.pfJetGenJetMatch+
                                           process.pfJetPartonAssociation+
                                           process.pfJetFlavourAssociation+
                                           process.pfJetCorrFactors+
                                           process.jetTracksAssociatorAtVertexPF+
                                           process.pfJetCharge+
                                           process.pfLayer1Jets)
     switchToPFJets( process,cms.InputTag('pfNoTau'), jetAlgo, l1jetColl  = cms.InputTag("pfLayer1Jets") )
     
     process.pfLayer1Objects=cms.Sequence(process.pfLayer1Electrons+
                                          process.makepflayer1Muons+
                                          process.makepflayer1Taus+
                                          process.makepflayer1Jets+
                                          process.pfLayer1METs+
                                          process.pfLayer1Summary)

     
     #SELECTED
     #  #ELECTRONS
     import PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi
     process.pfselectedLayer1Electrons = PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi.selectedLayer1Electrons.clone()
     process.pfselectedLayer1Electrons.src=cms.InputTag("pfLayer1Electrons")
     #  #MUONS
     import PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi
     process.pfselectedLayer1Muons = PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi.selectedLayer1Muons.clone()
     process.pfselectedLayer1Muons.src=cms.InputTag("pfLayer1Muons")     
     #  #TAUS
     import PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi
     process.pfselectedLayer1Taus = PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi.selectedLayer1Taus.clone()
     process.pfselectedLayer1Taus.src=cms.InputTag("pfLayer1Taus")     
     #  #JETS
     import PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi
     process.pfselectedLayer1Jets = PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi.selectedLayer1Jets.clone()
     process.pfselectedLayer1Jets.src=cms.InputTag("pfLayer1Jets")
     #  #SUMMARY
     process.pfselectedLayer1Summary = cms.EDAnalyzer(
         "CandidateSummaryTable",
         logName = cms.untracked.string("pfselectedLayer1Objects|PATSummaryTables"),
         candidates = cms.VInputTag(cms.InputTag("pfselectedLayer1Electrons"),
                                    cms.InputTag("pfselectedLayer1Muons"),
                                    cms.InputTag("pfselectedLayer1Taus"),
                                    cms.InputTag("pfselectedLayer1Photons"),
                                    cms.InputTag("pfselectedLayer1Jets"),
                                    cms.InputTag("pfLayer1METs")
                                    )
         )
     #  #SEQUENCE
     process.pfSelectedObjects=cms.Sequence(process.pfselectedLayer1Electrons+
                                            process.pfselectedLayer1Muons+
                                            process.pfselectedLayer1Taus+
                                            process.pfselectedLayer1Jets+
                                            process.pfselectedLayer1Summary)

     
     #COUNT
     #  #ELECTRONS
     import PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi 
     process.pfcountLayer1Electrons = PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi.countLayer1Electrons.clone()
     process.pfcountLayer1Electrons.src=cms.InputTag("pfselectedLayer1Electrons")
     #  #MUONS
     import PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi 
     process.pfcountLayer1Muons = PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi.countLayer1Muons.clone()
     process.pfcountLayer1Muons.src=cms.InputTag("pfselectedLayer1Muons")     
     #  #TAUS
     import PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi 
     process.pfcountLayer1Taus = PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi.countLayer1Taus.clone()
     process.pfcountLayer1Taus.src=cms.InputTag("pfselectedLayer1Taus")    
     #  #JETS
     import PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi 
     process.pfcountLayer1Jets = PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi.countLayer1Jets.clone()
     process.pfcountLayer1Jets.src=cms.InputTag("pfselectedLayer1Jets")         
     #  #SEQUENCE
     process.pfCountObjects=cms.Sequence(process.pfcountLayer1Electrons+
                                         process.pfcountLayer1Muons+
                                         process.pfcountLayer1Taus+
                                         process.pfcountLayer1Jets)
     
     #FINAL SEQUENCE
     addPFCandidates(process,cms.InputTag('pfNoJet'),patLabel='PFParticles',cut="",
                     layer='pfLayer1',selected='pfselectedLayer1',
                     counted='pfcountLayer1')
     process.PFPATafterPAT =cms.Sequence(process.PF2PAT+
                                         process.pfLayer1Objects+
                                         process.pfSelectedObjects+
                                         process.pfCountObjects)




def removeMCDependencedorPF( process ):
    #-- Remove MC dependence ------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.coreTools import removeMCMatching
    process.PF2PAT.remove(process.genParticlesForMETAllVisible)
    process.PF2PAT.remove(process.genMetTrue)
    process.PF2PAT.remove(process.genParticlesForJets)
    process.PF2PAT.remove(process.ak5GenJetsNoNu)
    process.PF2PAT.remove(process.iterativeCone5GenJetsNoNu)
    removeMCMatching(process, 'PFAll')
    

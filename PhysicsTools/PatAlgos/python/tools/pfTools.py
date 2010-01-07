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
    

def adaptPFElectrons(process,module,l1Collection=cms.InputTag("patElectrons")):
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

    if (l1Collection.moduleLabel=="patElectrons"):
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
   print "patTaus will be produced from taus of type: %s that pass %s" \
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
    
    #switchToAnyPFTau(process, oldTaus, process.patTaus.tauSource, tauType)
    switchToPFTauByType(process,module, pfTauType=tauType,
                        pfTauLabelNew=module.tauSource,
                        pfTauLabelOld=oldTaus)

    if (l0tauColl.moduleLabel=="allLayer0Taus"):
        process.makePatTaus.remove(process.patPFCandidateIsoDepositSelection)
    if (l0tauColl.moduleLabel=="pfLayer0Taus"):
        process.PF2PAT.remove(process.patPFCandidateIsoDepositSelection)

#helper function for PAT on PF2PAT sample
def tauTypeInPF2PAT(process,tauType='shrinkingConePFTau'): 
    process.load("PhysicsTools.PFCandProducer.pfTaus_cff")
    process.allLayer0Taus.src = cms.InputTag(tauType+'Producer')
            

def addPFCandidates(process,src,patLabel='PFParticles',cut="",
                    layer='pat',selected='selectedPat',
                    counted='countPat'):
    from PhysicsTools.PatAlgos.producersLayer1.pfParticleProducer_cfi import patPFParticles
    # make modules
    producer = patPFParticles.clone(pfCandidateSource = src)
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
    if (layer=='pat'):
        process.patCandidates.replace(process.patCandidateSummary, producer+process.patCandidateSummary)
        process.selectedPatCandidates.replace(process.selectedPatCandidateSummary, filter + process.selectedPatCandidateSummary)
        process.countPatCandidates += counter
        # summary tables
        process.patCandidateSummary.candidates.append(cms.InputTag('pat' + patLabel))
        process.selectedPatCandidateSummary.candidates.append(cms.InputTag('selectedPat' + patLabel))
    if (layer=='pfPat'): 
        process.pfPatCandidates.replace(process.pfPatCandidateSummary, producer +  process.pfPatCandidateSummary)
        process.pfSelectedPatCandidates.replace(process.pfSelectedPatCandidateSummary, filter +  process.pfSelectedPatCandidateSummary)
        process.pfCountPatCandidates += counter

        
def switchToPFMET(process,input=cms.InputTag('pfMET'),metColl=cms.InputTag('patAK5CaloMETs')):
    print 'MET: using ', input
    module =  getattr(process,metColl.moduleLabel)
    oldMETSource = module.metSource
    module.metSource = input
    module.addMuonCorrections = False
    if (metColl.moduleLabel=='patAK5CaloMETs'):
        process.patDefaultSequence.remove(process.patMETCorrections)


def switchToPFJets(process,
                   input=cms.InputTag('pfNoTau'), algo='IC5',
                   l1jetColl  = cms.InputTag(jetCollectionString())
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
                        algo,
                        'PFlow',
                        doJTA=True,
                        doBTagging=True,
                        jetCorrLabel=( algo, 'PF' ), 
                        doType1MET=False,
                        doL1Cleaning = False,                     
                        doL1Counters = False,   
                        genJetCollection = genJetCollectionName,
                        doJetID =True
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
       #process.dump = cms.EDAnalyzer("EventContentAnalyzer")
        process.patDefaultSequence.replace(process.patCandidates, process.PF2PAT+process.patCandidates)

    removeCleaning(process)
    
    # -------- OBJECTS ------------
    # Muons
    adaptPFMuons(process,process.patMuons)

    
    # Electrons
    adaptPFElectrons(process,process.patElectrons)

    # Photons
    print "Temporarily switching off photons completely"
    removeSpecificPATObjects(process,['Photons'])
    process.patDefaultSequence.remove(process.patPhotonIsolation)
    
    # Jets
    switchToPFJets( process, cms.InputTag('pfNoTau'), jetAlgo )
    
    # Taus
    #adaptPFTaus( process ) #default (i.e. shrinkingConePFTau)
    adaptPFTaus( process,process.patTaus, tauType='fixedConePFTau' )
    
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
     process.pfPatElectrons=PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi.patElectrons.clone()
     adaptPFElectrons(process,process.pfPatElectrons, cms.InputTag("pfPatElectrons"))

     #  #MUONS
     import PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi
     process.pfPatMuons=PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi.patMuons.clone()
     import PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi
     process.pfMuonMatch=PhysicsTools.PatAlgos.mcMatchLayer0.muonMatch_cfi.muonMatch.clone()
     process.pfPatMuons.genParticleMatch=cms.InputTag("pfMuonMatch")
     adaptPFMuons(process,process.pfPatMuons)

     #  #TAUS
     from PhysicsTools.PFCandProducer.pfTaus_cff import allLayer0Taus
     process.pfLayer0Taus=allLayer0Taus.clone()
     process.pfTauSequence.replace(process.allLayer0Taus,
                                   process.pfLayer0Taus)
     process.pfNoTau.topCollection=cms.InputTag("pfLayer0Taus")
     import PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi     
     process.pfPatTaus=PhysicsTools.PatAlgos.producersLayer1.tauProducer_cfi.patTaus.clone()
 

     import  PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi
     process.pfTauMatch =PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauMatch.clone()

     import PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi
     process.pfTauGenJetMatch =PhysicsTools.PatAlgos.mcMatchLayer0.tauMatch_cfi.tauGenJetMatch.clone()

     from PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff import *
     process.pfTauIsoDepositPFCandidates=PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFCandidates.clone()
     process.pfTauIsoDepositPFChargedHadrons = PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFChargedHadrons.clone()
     process.pfTauIsoDepositPFNeutralHadrons = PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFNeutralHadrons.clone()
     process.pfTauIsoDepositPFGammas = PhysicsTools.PatAlgos.recoLayer0.tauIsolation_cff.tauIsoDepositPFGammas.clone()

     process.pfPatTaus.isoDeposits.pfAllParticles = cms.InputTag("pfTauIsoDepositPFCandidates")
     process.pfPatTaus.isoDeposits.pfChargedHadron = cms.InputTag("pfTauIsoDepositPFChargedHadrons")
     process.pfPatTaus.isoDeposits.pfNeutralHadron = cms.InputTag("pfTauIsoDepositPFNeutralHadrons")
     process.pfPatTaus.isoDeposits.pfGamma = cms.InputTag("pfTauIsoDepositPFGammas")
     process.pfPatTaus.userIsolation.pfAllParticles.src = cms.InputTag("pfTauIsoDepositPFCandidates")
     process.pfPatTaus.userIsolation.pfChargedHadron.src = cms.InputTag("pfTauIsoDepositPFChargedHadrons")
     process.pfPatTaus.userIsolation.pfNeutralHadron.src = cms.InputTag("pfTauIsoDepositPFNeutralHadrons")
     process.pfPatTaus.userIsolation.pfGamma.src = cms.InputTag("pfTauIsoDepositPFGammas")
     process.pfPatTaus.genParticleMatch = cms.InputTag("pfTauMatch")
     process.pfPatTaus.genJetMatch      = cms.InputTag("pfTauGenJetMatch")
     adaptPFTaus( process,process.pfPatTaus,tauType='fixedConePFTau', l0tauColl=cms.InputTag("pfLayer0Taus"))

     #  #JETS
     import PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi
     process.pfPatJets=PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi.patAK5CaloJets.clone()
     import PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi
     process.pfJetPartonMatch =  PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi.jetPartonMatch.clone()
     process.pfJetGenJetMatch =  PhysicsTools.PatAlgos.mcMatchLayer0.jetMatch_cfi.jetGenJetMatch.clone()
     process.pfPatJets.genPartonMatch = cms.InputTag("pfJetPartonMatch")
     process.pfPatJets.genJetMatch    = cms.InputTag("pfJetGenJetMatch")  
     import PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff
     process.pfJetPartonAssociation  = PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff.jetPartonAssociation.clone()
     process.pfJetFlavourAssociation = PhysicsTools.PatAlgos.mcMatchLayer0.jetFlavourId_cff.jetFlavourAssociation.clone()
     process.pfJetFlavourAssociation.srcByReference = cms.InputTag("pfJetPartonAssociation")
     process.pfPatJets.JetPartonMapSource = cms.InputTag("pfJetFlavourAssociation")
     import PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi
     process.pfJetCorrFactors=PhysicsTools.PatAlgos.recoLayer0.jetCorrFactors_cfi.jetCorrFactors.clone()
     process.pfPatJets.jetCorrFactorsSource = cms.VInputTag(cms.InputTag("pfJetCorrFactors") )
     import RecoJets.JetAssociationProducers.ak5JTA_cff
     process.jetTracksAssociatorAtVertexPF = RecoJets.JetAssociationProducers.ak5JTA_cff.ak5JetTracksAssociatorAtVertex.clone()
     import PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff
     process.pfJetCharge = PhysicsTools.PatAlgos.recoLayer0.jetTracksCharge_cff.patJetCharge.clone()
     process.pfJetCharge.src = cms.InputTag("jetTracksAssociatorAtVertexPF")
     
     process.pfPatJets.trackAssociationSource = cms.InputTag("jetTracksAssociatorAtVertexPF")
     process.pfPatJets.jetChargeSource = cms.InputTag("pfJetCharge")
     
     #  #MET
     import PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi
     process.pfPatMETs = PhysicsTools.PatAlgos.producersLayer1.metProducer_cfi.patMETs.clone()
     switchToPFMET(process, cms.InputTag('pfMET'),metColl=cms.InputTag('pfPatMETs'))

     #  #SUMMARY
     process.pfPatCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
                                              logName = cms.untracked.string("pfPatCandidates|PATSummaryTables"),
                                              candidates = cms.VInputTag(cms.InputTag("pfPatElectrons"),
                                                                         cms.InputTag("pfPatMuons"),
                                                                         cms.InputTag("pfPatTaus"),
                                                                         cms.InputTag("pfPatPhotons"),
                                                                         cms.InputTag("pfPatJets"),
                                                                         cms.InputTag("pfPatMETs")
                                                                         )
                                              )

     #  #SEQUENCE

     process.makepflayer1Muons=cms.Sequence(process.pfMuonMatch+
                                            process.pfPatMuons)

     process.makepflayer1Taus=cms.Sequence(process.pfTauMatch+
                                           process.pfTauGenJetMatch+
                                           process.pfTauIsoDepositPFCandidates+                                          
                                           process.pfTauIsoDepositPFChargedHadrons+
                                           process.pfTauIsoDepositPFNeutralHadrons+
                                           process.pfTauIsoDepositPFGammas+
                                           process.pfPatTaus)

     process.makepflayer1Jets=cms.Sequence(process.pfJetPartonMatch+
                                           process.pfJetGenJetMatch+
                                           process.pfJetPartonAssociation+
                                           process.pfJetFlavourAssociation+
                                           process.pfJetCorrFactors+
                                           process.jetTracksAssociatorAtVertexPF+
                                           process.pfJetCharge+
                                           process.pfPatJets)
     switchToPFJets( process,cms.InputTag('pfNoTau'), jetAlgo, l1jetColl  = cms.InputTag("pfPatJets") )
     
     process.pfPatCandidates=cms.Sequence(process.pfPatElectrons+
                                          process.makepflayer1Muons+
                                          process.makepflayer1Taus+
                                          process.makepflayer1Jets+
                                          process.pfPatMETs+
                                          process.pfPatCandidateSummary)

     
     #SELECTED
     #  #ELECTRONS
     import PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi
     process.pfSelectedPatElectrons = PhysicsTools.PatAlgos.selectionLayer1.electronSelector_cfi.selectedPatElectrons.clone()
     process.pfSelectedPatElectrons.src=cms.InputTag("pfPatElectrons")
     #  #MUONS
     import PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi
     process.pfSelectedPatMuons = PhysicsTools.PatAlgos.selectionLayer1.muonSelector_cfi.selectedPatMuons.clone()
     process.pfSelectedPatMuons.src=cms.InputTag("pfPatMuons")     
     #  #TAUS
     import PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi
     process.pfSelectedPatTaus = PhysicsTools.PatAlgos.selectionLayer1.tauSelector_cfi.selectedPatTaus.clone()
     process.pfSelectedPatTaus.src=cms.InputTag("pfPatTaus")     
     #  #JETS
     import PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi
     process.pfSelectedPatJets = PhysicsTools.PatAlgos.selectionLayer1.jetSelector_cfi.selectedPatJets.clone()
     process.pfSelectedPatJets.src=cms.InputTag("pfPatJets")
     #  #SUMMARY
     process.pfSelectedPatCandidateSummary = cms.EDAnalyzer("CandidateSummaryTable",
         logName = cms.untracked.string("pfSelectedPatCandidates|PATSummaryTables"),
         candidates = cms.VInputTag(cms.InputTag("pfSelectedPatElectrons"),
                                    cms.InputTag("pfSelectedPatMuons"),
                                    cms.InputTag("pfSelectedPatTaus"),
                                    cms.InputTag("pfSelectedPatPhotons"),
                                    cms.InputTag("pfSelectedPatJets"),
                                    cms.InputTag("pfPatMETs")
                                    )
         )
     #  #SEQUENCE
     process.pfSelectedPatCandidates=cms.Sequence(process.pfSelectedPatElectrons+
                                                  process.pfSelectedPatMuons+
                                                  process.pfSelectedPatTaus+
                                                  process.pfSelectedPatJets+
                                                  process.pfSelectedPatCandidateSummary
                                                  )

     
     #COUNT
     #  #ELECTRONS
     import PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi 
     process.pfCountPatElectrons = PhysicsTools.PatAlgos.selectionLayer1.electronCountFilter_cfi.countPatElectrons.clone()
     process.pfCountPatElectrons.src=cms.InputTag("pfSelectedPatElectrons")
     #  #MUONS
     import PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi 
     process.pfCountPatMuons = PhysicsTools.PatAlgos.selectionLayer1.muonCountFilter_cfi.countPatMuons.clone()
     process.pfCountPatMuons.src=cms.InputTag("pfSelectedPatMuons")     
     #  #TAUS
     import PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi 
     process.pfCountPatTaus = PhysicsTools.PatAlgos.selectionLayer1.tauCountFilter_cfi.countPatTaus.clone()
     process.pfCountPatTaus.src=cms.InputTag("pfSelectedPatTaus")    
     #  #JETS
     import PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi 
     process.pfCountPatJets = PhysicsTools.PatAlgos.selectionLayer1.jetCountFilter_cfi.countPatJets.clone()
     process.pfCountPatJets.src=cms.InputTag("pfSelectedPatJets")         
     #  #SEQUENCE
     process.pfCountPatCandidates=cms.Sequence(process.pfCountPatElectrons+
                                               process.pfCountPatMuons+
                                               process.pfCountPatTaus+
                                               process.pfCountPatJets
                                               )
     
     #FINAL SEQUENCE
     addPFCandidates(process,cms.InputTag('pfNoJet'),patLabel='PFParticles',cut="",
                     layer='pfPat',selected='pfselectedPat',
                     counted='pfcountPat')
     process.PFPATafterPAT =cms.Sequence(process.PF2PAT+
                                         process.pfPatCandidates+
                                         process.pfSelectedPatCandidates+
                                         process.pfCountPatCandidates
                                         )


def removeMCDependencedorPF( process ):
    #-- Remove MC dependence ------------------------------------------------------
    from PhysicsTools.PatAlgos.tools.coreTools import removeMCMatching
    process.patDefaultSequence.remove(process.genParticlesForMETAllVisible)
    process.patDefaultSequence.remove(process.genMetTrue)
    process.patDefaultSequence.remove(process.genParticlesForJets)
    process.patDefaultSequence.remove(process.ak5GenJetsNoNu)
    process.patDefaultSequence.remove(process.iterativeCone5GenJetsNoNu)
    removeMCMatching(process, ['PFAll'])
    

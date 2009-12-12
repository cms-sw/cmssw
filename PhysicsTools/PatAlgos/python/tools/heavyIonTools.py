import FWCore.ParameterSet.Config as cms


def configureHeavyIons(process):
   """
   ------------------------------------------------------------------
   configure all defaults for heavy ions
   
   process : process
   ------------------------------------------------------------------    
   """
   productionDefaults(process)
   selectionDefaults(process)


def productionDefaults(process):
   """
   ------------------------------------------------------------------
   configure all relevant layer1 candidates for heavy ions
   
   process : process
   ------------------------------------------------------------------    
   """
   ## adapt jet defaults
   jetCors  = getattr(process, 'jetCorrFactors')
   jetCors.jetSource = cms.InputTag("iterativeConePu5CaloJets")

   jetMatch = getattr(process, 'jetGenJetMatch')
   jetMatch.src     = cms.InputTag("iterativeConePu5CaloJets")
   jetMatch.matched = cms.InputTag("hiCleanedGenJets")
   
   patJets = getattr(process, 'allLayer1Jets')
   patJets.addBTagInfo         = False
   patJets.addTagInfos         = False
   patJets.addDiscriminators   = False
   patJets.addAssociatedTracks = False
   patJets.addJetCharge        = False
   patJets.addJetID            = False
   patJets.getJetMCFlavour     = False
   patJets.addGenPartonMatch   = False
   patJets.addGenJetMatch      = True
   patJets.jetSource  = cms.InputTag("iterativeConePu5CaloJets")

   ## adapt muon defaults
   muonMatch = getattr(process, 'muonMatch')
   muonMatch.matched = cms.InputTag("hiGenParticles")
   patMuons  = getattr(process, 'allLayer1Muons')
   patMuons.embedGenMatch = cms.bool(True)

   ## adapt photon defaults
   photonMatch = getattr(process, 'photonMatch')
   photonMatch.matched = cms.InputTag("hiGenParticles")
   patPhotons  = getattr(process, 'allLayer1Photons')
   patPhotons.addPhotonID   = cms.bool(True)
   patPhotons.addGenMatch   = cms.bool(True)
   patPhotons.embedGenMatch = cms.bool(True)
   patPhotons.userData.userFloats.src  = cms.VInputTag(
      cms.InputTag( "isoCC1"),cms.InputTag( "isoCC2"),cms.InputTag( "isoCC3"),cms.InputTag( "isoCC4"),cms.InputTag("isoCC5"),
      cms.InputTag( "isoCR1"),cms.InputTag( "isoCR2"),cms.InputTag( "isoCR3"),cms.InputTag( "isoCR4"),cms.InputTag("isoCR5"),
      cms.InputTag( "isoT11"),cms.InputTag( "isoT12"),cms.InputTag( "isoT13"),cms.InputTag( "isoT14"),  
      cms.InputTag( "isoT21"),cms.InputTag( "isoT22"),cms.InputTag( "isoT23"),cms.InputTag( "isoT24"),  
      cms.InputTag( "isoT31"),cms.InputTag( "isoT32"),cms.InputTag( "isoT33"),cms.InputTag( "isoT34"),  
      cms.InputTag( "isoT41"),cms.InputTag( "isoT42"),cms.InputTag( "isoT43"),cms.InputTag( "isoT44"),  
      cms.InputTag("isoDR11"),cms.InputTag("isoDR12"),cms.InputTag("isoDR13"),cms.InputTag("isoDR14"),  
      cms.InputTag("isoDR21"),cms.InputTag("isoDR22"),cms.InputTag("isoDR23"),cms.InputTag("isoDR24"),  
      cms.InputTag("isoDR31"),cms.InputTag("isoDR32"),cms.InputTag("isoDR33"),cms.InputTag("isoDR34"),  
      cms.InputTag("isoDR41"),cms.InputTag("isoDR42"),cms.InputTag("isoDR43"),cms.InputTag("isoDR44")
      )
   patPhotons.photonIDSource = cms.InputTag("PhotonIDProd","PhotonCutBasedIDLoose")


def selectionDefaults(process):
   """
   ------------------------------------------------------------------
   configure all relevant selected layer1 candidates for heavy ions
   
   process : process
   ------------------------------------------------------------------    
   """
   selectedJets = getattr(process, 'selectedLayer1Jets')
   selectedJets.cut = cms.string('pt > 20.')
   selectedMuons = getattr(process, 'selectedLayer1Muons')
   selectedMuons.cut = cms.string('pt > 0. & abs(eta) < 12.')
   selectedPhotons = getattr(process, 'selectedLayer1Photons')
   selectedPhotons.cut = cms.string('pt > 0. & abs(eta) < 12.')

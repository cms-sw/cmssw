import FWCore.ParameterSet.Config as cms

process = cms.Process("ALL")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

#
# This runs over a file that already contains the L1Tracks.
#


from SLHCUpgradeSimulations.L1TrackTrigger.minBiasFiles_p1_cfi import *

process.source = cms.Source("PoolSource",
     fileNames = minBiasFiles_p1
)


# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')


process.load('Configuration.Geometry.GeometryExtended2023TTIReco_cff')
process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
process.load('Configuration.Geometry.GeometryExtended2023TTI_cff')

process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')

# ---------------------------------------------------------------------------
#
# --- Recreate the L1Tracks to benefit from the latest updates, and create
#     the collection of special tracks for electrons
#

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TrackingSequence_cfi")
#process.pTracking = cms.Path( process.DefaultTrackingSequence )
process.pTracking = cms.Path( process.FullTrackingSequence )


# ---------------------------------------------------------------------------



# --- Muons
process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkMuonSequence_cfi")
process.pMuons = cms.Path( process.L1TkMuons )


# ---------------------------------------------------------------------------

#
# --- run the L1Calo simulation
#

process.load('Configuration/StandardSequences/L1HwVal_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_forTTI_cff")

# bug fix for missing HCAL TPs in MC RAW
from SimCalorimetry.HcalTrigPrimProducers.hcaltpdigi_cff import HcalTPGCoderULUT
HcalTPGCoderULUT.LUTGenerationMode = cms.bool(True)
process.valRctDigis.hcalDigis             = cms.VInputTag(cms.InputTag('valHcalTriggerPrimitiveDigis'))
process.L1CaloTowerProducer.HCALDigis =  cms.InputTag("valHcalTriggerPrimitiveDigis")

process.slhccalo = cms.Path( process.RawToDigi + process.valHcalTriggerPrimitiveDigis+process.SLHCCaloTrigger)

        # To produce L1EG objects corresponding
        # to the Run-1 L1EG algorithm, one just needs to run
        # L1Reco. The (Run-1) L1EG algorithm has already been
        # run in the DIGI step of the production. 
process.load('Configuration.StandardSequences.L1Reco_cff')
process.L1Reco = cms.Path( process.l1extraParticles )


# ----    Produce the L1EGCrystal clusters (code of Sasha Savin & Nick Smith)

# This is now included in L1TkEmTauSequence_cfi.py, which is run later

        # first you need the ECAL RecHIts :
#process.load('Configuration.StandardSequences.Reconstruction_cff')
#process.reconstruction_step = cms.Path( process.calolocalreco )
#
#
#process.L1EGammaCrystalsProducer = cms.EDProducer("L1EGCrystalClusterProducer",
#   EtminForStore = cms.double( 4. ),
#   debug = cms.untracked.bool(False),
#   useECalEndcap = cms.bool(True)
#)
#process.pSasha = cms.Path( process.L1EGammaCrystalsProducer )


	# needed because the calo stuff above clashes with the DTs 
process.es_prefer_dt = cms.ESPrefer("DTConfigTrivialProducer","L1DTConfig")



# ---------------------------------------------------------------------------

# Now we produce L1TkEmParticles and L1TkElectrons

# ----  "photons" isolated w.r.t. L1Tracks :
        
process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEmParticleProducer_cfi")
process.pL1TkPhotons = cms.Path( process.L1TkPhotons )
        
# ---- "photons", tighter isolation working point  -  e.g. for SinglePhoton trigger
process.pL1TkPhotonsTightIsol = cms.Path( process.L1TkPhotonsTightIsol )

# ----  "electrons" from L1Tracks. Inclusive electrons :

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkElectronTrackProducer_cfi")
process.pElectrons = cms.Path( process.L1TkElectrons )

# ----  L1TkElectrons that are isolated w.r.t. L1Tracks :

process.L1TkIsoElectrons = process.L1TkElectrons.clone()
process.L1TkIsoElectrons.IsoCut = cms.double(0.1)
process.pElectronsTkIso = cms.Path( process.L1TkIsoElectrons )

# ---- "electrons" from L1Tracks, Inclusive electrons : dedicated low PT sequence
process.pElectronsLoose = cms.Path( process.L1TkElectronsLoose)

# ---- L1TkElectrons that are isolated w.r.t. L1Tracks : dedicated low PT sequence
process.L1TkIsoElectronsLoose = process.L1TkElectronsLoose.clone()
process.L1TkIsoElectronsLoose.IsoCut = cms.double(0.1)
process.pElectronsTkIsoLoose = cms.Path( process.L1TkIsoElectronsLoose )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# --- L1TkMET

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkPrimaryVertexProducer_cfi")
process.pL1TkPrimaryVertex = cms.Path( process.L1TkPrimaryVertex )

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEtMissProducer_cfi")
process.pL1TrkMET = cms.Path( process.L1TkEtMiss )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# --- TkTaus

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkEmTauSequence_cfi")
process.pTaus = cms.Path( process.TkEmTauSequence )

#process.L1TkTauFromL1Track = cms.EDProducer("L1TkTauFromL1TrackProducer",
#                                            L1TrackInputTag = cms.InputTag("TTTracksFromPixelDigis","Level1TTTracks"),
#                                            ZMAX = cms.double( 25. ),# in cm
#                                            CHI2MAX = cms.double( 100. ),
#                                            PTMINTRA = cms.double( 2. ),# in GeV
#                                            DRmax = cms.double( 0.5 ),
#                                            nStubsmin = cms.int32( 5 )        # minimum number of stubs
#                                            )
#process.pTaus = cms.Path( process.L1TkTauFromL1Track )

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# --- jets, HT and MHT

process.load("SLHCUpgradeSimulations.L1TrackTrigger.L1TkCaloSequence_cff")

# -- Produce L1TkJets, HT and MHT from the L1Jets :       
process.L1TkCaloL1Jets = cms.Path( process.L1TkCaloSequence )

# -- Produce the HLT JI Jets and L1TkJets, HT and MHT from  these jets :
process.L1TkCaloHIJets = cms.Path( process.L1TkCaloSequenceHI )
# ---------------------------------------------------------------------------


process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "FileOut.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring( 'drop *')
)


        # the final collection of L1TkMuons
process.Out.outputCommands.append('keep *_L1TkMuonsMerge*_*_*')
        # intermediate collections :
#process.Out.outputCommands.append('keep *_L1TkMuons*_*_*')
process.Out.outputCommands.append('keep *_L1TkMuonsDT_DTMatchInwardsTTTrackFullReso_*')
process.Out.outputCommands.append('keep *_l1extraMuExtended_*_*')
process.Out.outputCommands.append('keep *_l1TkMuonsExt*_*_*')


        # the L1EG objects
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_*_*' )
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticlesNewClustering_*_*')
process.Out.outputCommands.append('keep l1extraL1Em*_l1extraParticles_*_*')
                # for crystal-level granularity :
process.Out.outputCommands.append('keep *_L1EGammaCrystalsProducer_*_*')
#process.Out.outputCommands.append('keep *_l1ExtraCrystalProducer_*_*')
        # the L1TkEmParticles
process.Out.outputCommands.append('keep *_L1TkPhotons_*_*')
        # the L1TkElectrons
process.Out.outputCommands.append('keep *_L1TkElectrons_*_*')
process.Out.outputCommands.append('keep *_L1TkIsoElectrons_*_*')
process.Out.outputCommands.append('keep *_L1TkElectronsLoose_*_*')
process.Out.outputCommands.append('keep *_L1TkIsoElectronsLoose_*_*')


        # the L1TkPrimaryVertex
process.Out.outputCommands.append('keep *_L1TkPrimaryVertex_*_*')
        # the TrkMET 
process.Out.outputCommands.append('keep *_L1TkEtMiss*_*_*')
        # the calo-based L1MET 
process.Out.outputCommands.append('keep *_l1extraParticles_MET_*')

	# TkTaus
process.Out.outputCommands.append('keep *_L1TkEmTauProducer_*_*')
process.Out.outputCommands.append('keep *_SLHCL1ExtraParticles_Taus_*')

	# jets, HT, MHT
# Collections of L1TkJets :
process.Out.outputCommands.append('keep *_L1TkJets_*_*')        #  L1TkJets from the L1Jets
process.Out.outputCommands.append('keep *_L1TkJetsHI_*_*')      #  L1TkJets from the HLT Heavy Ion jets
# intermediate products:
process.Out.outputCommands.append('keep *_iterativeConePu5CaloJets_*_*')        # HLT HI jets
process.Out.outputCommands.append('keep *_L1CalibFilterTowerJetProducer_CalibratedTowerJets_*')         # L1Jets
process.Out.outputCommands.append('keep *_L1CalibFilterTowerJetProducer_UncalibratedTowerJets_*')       # L1Jets
# Collections of HT and MHT variables :
        # -- made from the L1Jets :
process.Out.outputCommands.append('keep *_L1TkHTMissCalo_*_*')          # from L1Jets, calo only
process.Out.outputCommands.append('keep *_L1TkHTMissVtx_*_*')           # from L1Jets, with vertex constraint
        # -- made from the HLT HI jets:
process.Out.outputCommands.append('keep *_L1TkHTMissCaloHI_*_*')        # from HLT HI jets, calo only
process.Out.outputCommands.append('keep *_L1TkHTMissVtxHI_*_*')         # from HLT HI jets, with vertex constraint

	# keep the (rebuilt) tracks
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_ClusterAccepted_ALL')
#process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_ClusterAccepted_ALL')
#process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_StubAccepted_ALL')
#process.Out.outputCommands.append('keep *_TTStubsFromPixelDigis_StubAccepted_ALL')

process.Out.outputCommands.append('keep *_TTTracksFromPixelDigis_Level1TTTracks_ALL')
process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigisLargerPhi_Level1TTTracks_ALL')
#process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_Level1TTTracks_ALL')




	# gen level information
##process.Out.outputCommands.append('keep *_generator_*_*')
##process.Out.outputCommands.append('keep *_*gen*_*_*')
##process.Out.outputCommands.append('keep *_*Gen*_*_*')
#process.Out.outputCommands.append('keep *_genParticles_*_*')

        # the gen-level MET
#process.Out.outputCommands.append('keep *_genMetTrue_*_*')

# --- to browse the genParticles, one needs to keep the collections of associators below:
#process.Out.outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')
#process.Out.outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
#process.Out.outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')


process.FEVToutput_step = cms.EndPath(process.Out)








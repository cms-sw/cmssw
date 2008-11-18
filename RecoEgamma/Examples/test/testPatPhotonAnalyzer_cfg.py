import FWCore.ParameterSet.Config as cms

process = cms.Process("PhotonIDProc")

# Physics Analysis Tools (PAT)
process.load("PhysicsTools.PatAlgos.patLayer0_cff")
process.load("PhysicsTools.PatAlgos.patLayer1_cff")
# PhotonID
process.load("RecoEgamma.PhotonIdentification.photonId_cff")
# Standard
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("Configuration.StandardSequences.MagneticField_cff")


##############################################################
# Input files
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/3652B809-B585-DD11-A8D9-000423D9939C.root',
        '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/48AFC4AC-B485-DD11-A63C-000423D94C68.root',
        '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0000/4E76DF7F-B385-DD11-955C-000423D6A6F4.root',
        '/store/relval/CMSSW_2_1_9/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v2/0001/725453E4-0487-DD11-A22C-000423D94494.root'
)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
##############################################################


##############################################################
# Set Pat "cleaning" variables.  These cut
# photons that do not pass these.
# The following variables here are the same as in 
#  CMSSW/PhysicsTools/PatAlgos/python/cleaningLayer0/photonCleaner_cfi.py
# but I have copied and pasted them here for simplicity.

process.allLayer0Photons.removeDuplicates = cms.string('none')
process.allLayer0Photons.removeElectrons  = cms.string('none')
process.allLayer0Photons.saveAll          = cms.string('all')

process.allLayer0Photons.isolation = cms.PSet(
        tracker = cms.PSet(
            # source
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositTk"),
            # parameters
            deltaR = cms.double(0.4),               # Cone radius
            vetos  = cms.vstring('0.04',            # Inner veto cone radius
                                 'Threshold(1.0)'), # Pt threshold
            skipDefaultVeto = cms.bool(True),
            # cut value
            cut = cms.double(50.0),
        ),
        ecal = cms.PSet(
            # source
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositEcalFromClusts"),
            # parameters
            deltaR          = cms.double(0.4),
            vetos           = cms.vstring('EcalBarrel:0.045', 'EcalEndcaps:0.070'),
            skipDefaultVeto = cms.bool(True),
            # cut value
            cut = cms.double(50.0),
        ),
        hcal = cms.PSet(
            # source 
            src = cms.InputTag("patAODPhotonIsolations","gamIsoDepositHcalFromTowers"),
            # parameters
            deltaR          = cms.double(0.4),
            skipDefaultVeto = cms.bool(True),
            # cut value
            cut = cms.double(50.0)
        ),
        user = cms.VPSet(),
)
##############################################################


##############################################################
# Variables for our analyzer
process.photonIDAna = cms.EDAnalyzer("PatPhotonSimpleAnalyzer",
    outputFile  = cms.string('PatPhotonHists.root'),

    # Some extra cuts you might wish to make
    #  before histograms/TTrees are filled.
    # Minimum Et
    minPhotonEt     = cms.double(10.0),
    # Minimum and max abs(eta)
    minPhotonAbsEta = cms.double(0.0),
    maxPhotonAbsEta = cms.double(3.0),
    # Minimum R9 = E(3x3) / E(SuperCluster)
    minPhotonR9     = cms.double(0.3),
    # Maximum HCAL / ECAL Energy
    maxPhotonHoverE = cms.double(0.2),

    # Optionally produce a TTree of photons (set to False or True).
    createPhotonTTree  = cms.bool(True)
)
##############################################################


# Run PhotonID, PAT Layer0, PAT Layer1, Our Analyzer
process.p = cms.Path(process.photonIDSequence*process.patLayer0*process.patLayer1*process.photonIDAna)


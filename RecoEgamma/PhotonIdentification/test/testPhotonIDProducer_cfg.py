import FWCore.ParameterSet.Config as cms

process = cms.Process("PhotonIDProc")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("RecoEgamma.PhotonIdentification.photonId_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
#process.load("MagneticField.Engine.volumeBasedMagneticField_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("RecoEgamma.EgammaPhotonProducers.photonSequence_cff")

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213986417-IDEAL_V2-2nd/0004/443BCAED-CB40-DD11-AB37-000423D6B48C.root')
    #fileNames = cms.untracked.vstring('dcap://cmsdcap.hep.wisc.edu:22125/pnfs/hep.wisc.edu/store/user/mbanderson/PhotonJet20-200/PhotonJet20-200-0000.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.Out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring('drop *', 
        'keep edmHepMCProduct_*_*_*', 
        'keep recoBasicClusters_*_*_*', 
        'keep recoSuperClusters_*_*_*', 
        'keep *_PhotonIDProd_*_*', 
        'keep *_PhotonIDProd_*_*', 
        'keep recoPhotons_*_*_*'),
    fileName = cms.untracked.string('Photest.root')
)

process.photonIDAna = cms.EDAnalyzer("PhotonIDSimpleAnalyzer",
    outputFile  = cms.string('PhoIDHists.root'),

    # Variables that must be passed before a photon candidate (a SuperCluster)
    #  gets placed into the histograms.  Basic, simple cuts.
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
    # This slows down the analyzer, and if running
    # over 100,000+ events, this can create a large ROOT file
    createPhotonTTree  = cms.bool(False)
)

#process.p = cms.Path(process.photonSequence*process.photonIDSequence*process.photonIDAna)
process.p = cms.Path(process.photonSequence*process.photonIDSequence*process.photonIDAna)
process.e = cms.EndPath(process.Out)

process.PoolSource.fileNames = ['dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_7.root',
'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_13.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_14.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_4.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_10.root','dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_5.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_08.08_on_June_10-2008/gamjet_pt40-60/reco_20.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_38.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_12.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_29.root','dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_7.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_40.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_31.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_22.root', 'dcache:/pnfs/cms/WAX/resilient/oleksiy/temp/CMSSW_2_1_0_pre5/Tuesday_at_09.25_on_June_10-2008/gamjet_pt40-60/reco_8.root']

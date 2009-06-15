import FWCore.ParameterSet.Config as cms

process = cms.Process('SkimH2Gam')

# Complete Preselection Sequence for 2e2mu analysis

process.load('Configuration/StandardSequences/Services_cff')
process.load('FWCore/MessageService/MessageLogger_cfi')

process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')

#new for photonid
process.load("RecoEgamma.PhotonIdentification.photonId_cff")
process.load("Geometry.CaloEventSetup.CaloGeometry_cfi")
process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")


#process.GlobalTag.globaltag = 'IDEAL_V11::All'
process.GlobalTag.globaltag = 'STARTUP_V11::All'

# Complete Skim analysis
process.load('HiggsAnalysis/Skimming/higgsTo2Gamma_Sequences_cff')

process.eca = cms.EDAnalyzer("EventContentAnalyzer")

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
                                         createPhotonTTree  = cms.bool(True)
                                     )

process.PhotonIDProd.EcalRecHitInnerRadius = cms.double(0.08)
process.PhotonIDProd.EcalRecHitEtaSlice = cms.double(0.05)
process.PhotonIDProd.EcalRecThresh = cms.double(0.1)
process.PhotonIDProd.isolationtrackThreshold = cms.double(1.5)

#process.p = cms.Path(process.photonIDSequence*process.photonIDAna)

#process.hTo2GammaSkimPath = cms.Path(process.higgsTo2GammaSequence+process.eca)
process.hTo2GammaSkimPath = cms.Path(process.photonIDSequence*process.photonIDAna+process.higgsTo2GammaSequence)


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('hTo2Gamma_Skim_isol.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('higgsTo2Gamma_Sequence')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('hTo2GammaSkimPath')
    )
                               
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(500) )

process.source = cms.Source("PoolSource",
                            debugFlag = cms.untracked.bool(True),
                            debugVebosity = cms.untracked.uint32(10),
                            # fileNames = cms.untracked.vstring('file:/home/llr/cms/ndefilip/RAW2DIGI_RECO_IDEAL_21_2e2mu.root'                            
                            #fileNames = cms.untracked.vstring('/store/user/ndefilip/comphep-bbll/CMSSW_2_2_3-bkg_RAW2DIGI_RECO_IDEAL/6e3420323cbcc78f83bfe627dc999a04/RAW2DIGI_RECO_IDEAL_998.root'
                            # )

                            fileNames = cms.untracked.vstring(
                    '/store/relval/CMSSW_2_2_10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_V11_v1/0003/4A892733-043E-DE11-B7A6-001D09F24448.root',
                                    '/store/relval/CMSSW_2_2_10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_V11_v1/0002/98518B50-903D-DE11-A8C6-001D09F24024.root',
                                    '/store/relval/CMSSW_2_2_10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_V11_v1/0002/34DB16FC-8D3D-DE11-A97C-001D09F2A465.root',
                                    '/store/relval/CMSSW_2_2_10/RelValH130GGgluonfusion/GEN-SIM-RECO/STARTUP_V11_v1/0002/1AF6C5B2-8B3D-DE11-BC0D-001D09F291D2.root'
                    )
                            
                            )

#Loose cuts (this is the default)
#process.higgsTo2GammaFilter.photonLooseMinPt = cms.double(15.0)
#process.higgsTo2GammaFilter.photonTightMinPt = cms.double(25.0)
#process.higgsTo2GammaFilter.photonLooseMaxEta = cms.double(3.1)
#process.higgsTo2GammaFilter.photonTightMaxEta = cms.double(2.6)
#process.higgsTo2GammaFilter.photonLooseMaxHoE = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHoE = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonLooseMaxHIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHIsol = cms.double(20.)
#process.higgsTo2GammaFilter.photonLooseMaxEIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxEIsol = cms.double(20.)
#process.higgsTo2GammaFilter.photonLooseMaxTIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxTIsol = cms.double(10.)
#process.higgsTo2GammaFilter.nPhotonLooseMin = cms.int32(2)
#process.higgsTo2GammaFilter.nPhotonTightMin = cms.int32(1)
#process.higgsTo2GammaFilter.DebugHiggsTo2GammaSkim = cms.bool(False)

#Tight cuts (uncomment the following) 
#process.higgsTo2GammaFilter.photonLooseMinPt = cms.double(20.0)
#process.higgsTo2GammaFilter.photonTightMinPt = cms.double(30.0)
#process.higgsTo2GammaFilter.photonLooseMaxEta = cms.double(3.1)
#process.higgsTo2GammaFilter.photonTightMaxEta = cms.double(2.6)
#process.higgsTo2GammaFilter.photonLooseMaxHoE = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHoE = cms.double(0.2)
#process.higgsTo2GammaFilter.photonLooseMaxHIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxHIsol = cms.double(15.)
#process.higgsTo2GammaFilter.photonLooseMaxEIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxEIsol = cms.double(10.)
#process.higgsTo2GammaFilter.photonLooseMaxTIsol = cms.double(-1.0)
#process.higgsTo2GammaFilter.photonTightMaxTIsol = cms.double(5.)
#process.higgsTo2GammaFilter.nPhotonLooseMin = cms.int32(2)
#process.higgsTo2GammaFilter.nPhotonTightMin = cms.int32(1)
#process.higgsTo2GammaFilter.DebugHiggsTo2GammaSkim = cms.bool(False)

process.higgsTo2GammaFilter.DebugHiggsTo2GammaSkim = cms.bool(True)

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)



# Endpath
# process.o = cms.EndPath ( process.output )




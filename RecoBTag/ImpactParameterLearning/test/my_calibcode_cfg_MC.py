
##*****************************************************
##*****************************************************
##******** for DATA
##*****************************************************
##*****************************************************
import FWCore.ParameterSet.Config as cms

from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
from RecoBTag.ImpactParameter.impactParameter_cfi import *

process = cms.Process("Calib")

process.source = cms.Source(
  "PoolSource",
  # replace 'myfile.root' with the source file you want to use
  fileNames = cms.untracked.vstring(   
  # data from /Jet/Run2012A-PromptReco-v1/AOD
  #        'file:FE2E21F0-0583-E111-8DA9-001D09F241F0.root'
  # data from /BTag/Run2012A-PromptReco-v1/AOD
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/F88F3DF3-53E7-E111-91BE-003048D4DCD8.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/000F649B-49E7-E111-9434-00237DE0BED6.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0030E782-40E7-E111-A87C-003048F0E830.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/005FC58F-5DE7-E111-883D-0025904B1446.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0066EAF2-64E7-E111-8B4D-0025904B578E.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/020C40F4-6AE7-E111-9E75-00266CFFA780.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/04D8EE0A-5AE7-E111-A08B-002481E0E450.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/064DF4E6-5BE7-E111-9EB7-0030487F171B.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/06A9A98B-5FE7-E111-9F82-002590494C92.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/082FC482-75E7-E111-9618-003048F02D36.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0896105C-66E7-E111-93A9-00266CFFA2D0.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/08DDFF81-50E7-E111-A859-003048D3C7BC.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0AA179BC-53E7-E111-A7CB-003048D4DF90.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0ACE5C03-61E7-E111-BE6B-002481E76052.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0C419424-4DE7-E111-9E39-0030487F0EDF.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0CB0FEA1-46E7-E111-A8C3-00266CF32930.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0CE72B6F-60E7-E111-9F05-002481E76052.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0E2755E3-5DE7-E111-982B-002481E0D500.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/0E36C61E-66E7-E111-9EED-00237DE0BED6.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/102300AF-66E7-E111-8243-003048C68AA6.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/102D7ED4-63E7-E111-B36D-003048C692FE.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1219B9DA-61E7-E111-B6D4-0025904B1428.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/124AA395-31E7-E111-A043-0030487D710F.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/143865A9-4DE7-E111-8FBE-0030487F0EDF.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/146FAC44-52E7-E111-BCE8-003048C6903A.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/14E81D7D-52E7-E111-962A-003048D439BE.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/160E1D82-35E7-E111-94C5-0030487F92B5.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/16501EA4-48E7-E111-B40C-003048D4397E.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1684525F-57E7-E111-A4C5-0025904B144A.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/18E2D0DD-31E7-E111-8D8F-0030487F1A67.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/18F50273-52E7-E111-A5F8-0025901D4B22.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/18FBC0D1-61E7-E111-B83D-002481E7628E.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1A3EB849-5DE7-E111-9130-003048C692FE.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1A8EBFCD-63E7-E111-9A61-0025904B578E.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1ADE7EA3-51E7-E111-A022-003048D3C7BC.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1C0EF983-47E7-E111-A43A-00266CFFA604.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1C8BB143-7FE7-E111-9DA2-002481E0E912.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1CE771D7-65E7-E111-BDE9-003048C68A9E.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1E2D5345-5BE7-E111-B5B6-0030487F171B.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1EB5B5EA-66E7-E111-A765-0030487D83B9.root',
  'rfio:/dpm/in2p3.fr/home/cms/phedex/store/mc/Summer12_DR53X/QCD_Pt-120to170_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/00000/1EF528D3-61E7-E111-BD9B-003048F010F4.root'

 )
  )

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
  )


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = "GR_R_52_V7::All"
process.GlobalTag.globaltag = "START53_V7A::All"
#process.GlobalTag.globaltag = "GR_R_53_V2B::All"


###-------------------- Import the JEC services -----------------------
#process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
##$$
###-------------------- Import the Jet RECO modules -----------------------
#process.load('RecoJets.Configuration.RecoPFJets_cff')
###-------------------- Turn-on the FastJet density calculation -----------------------
#process.kt6PFJets.doRhoFastjet = True
###-------------------- Turn-on the FastJet jet area calculation for your favorite algorithm -----------------------
#process.ak5PFJets.doAreaFastjet = True
#$$

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.load("SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")

process.load("RecoBTau.JetTagComputer.jetTagRecord_cfi")

process.load("SimTracker.TrackHistory.TrackHistory_cff")

# for Impact Parameter based taggers
process.load("RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityComputer_cfi")
process.load("RecoBTag.ImpactParameter.negativeOnlyJetProbabilityComputer_cfi")
process.load("RecoBTag.ImpactParameter.positiveOnlyJetProbabilityComputer_cfi")
process.load("RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityComputer_cfi")
process.load("RecoBTag.ImpactParameter.negativeTrackCounting3D2ndComputer_cfi")
process.load("RecoBTag.ImpactParameter.negativeTrackCounting3D3rdComputer_cfi")
process.load("RecoBTag.Configuration.RecoBTag_cff")
process.load("RecoJets.JetAssociationProducers.ak5JTA_cff")
process.load("RecoBTag.ImpactParameter.negativeOnlyJetBProbabilityJetTags_cfi")
process.load("RecoBTag.ImpactParameter.negativeOnlyJetProbabilityJetTags_cfi")
process.load("RecoBTag.ImpactParameter.positiveOnlyJetProbabilityJetTags_cfi")
process.load("RecoBTag.ImpactParameter.positiveOnlyJetBProbabilityJetTags_cfi")
process.load("RecoBTag.ImpactParameter.negativeTrackCountingHighPur_cfi")
process.load("RecoBTag.ImpactParameter.negativeTrackCountingHighEffJetTags_cfi")
process.load("RecoBTag.ImpactParameter.jetProbabilityBJetTags_cfi")
process.load("RecoBTag.ImpactParameter.jetBProbabilityBJetTags_cfi")

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")

# for Secondary Vertex taggers
process.load("RecoBTag.SecondaryVertex.secondaryVertexTagInfos_cfi")
process.load("RecoBTag.SecondaryVertex.secondaryVertexNegativeTagInfos_cfi")
process.load("RecoBTag.SecondaryVertex.simpleSecondaryVertexHighEffBJetTags_cfi")
process.load("RecoBTag.SecondaryVertex.simpleSecondaryVertexHighPurBJetTags_cfi")
process.load("RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeHighEffBJetTags_cfi")
process.load("RecoBTag.SecondaryVertex.simpleSecondaryVertexNegativeHighPurBJetTags_cfi")
process.load("RecoBTag.SecondaryVertex.combinedSecondaryVertexNegativeBJetTags_cfi")
process.load("RecoBTag.SecondaryVertex.combinedSecondaryVertexNegativeES_cfi")
process.load("RecoBTag.SecondaryVertex.combinedSecondaryVertexPositiveBJetTags_cfi")
process.load("RecoBTag.SecondaryVertex.combinedSecondaryVertexPositiveES_cfi")

# for Soft Muon tagger
process.load("RecoBTag.SoftLepton.negativeSoftMuonES_cfi")
process.load("RecoBTag.SoftLepton.positiveSoftMuonES_cfi")
process.load("RecoBTag.SoftLepton.negativeSoftMuonBJetTags_cfi")
process.load("RecoBTag.SoftLepton.positiveSoftMuonBJetTags_cfi")

process.load("RecoBTag.SoftLepton.negativeSoftLeptonByPtES_cfi")
process.load("RecoBTag.SoftLepton.positiveSoftLeptonByPtES_cfi")
process.load("RecoBTag.SoftLepton.negativeSoftMuonByPtBJetTags_cfi")
process.load("RecoBTag.SoftLepton.positiveSoftMuonByPtBJetTags_cfi")

process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")  

process.load("SimTracker.TrackHistory.TrackClassifier_cff")
process.load("RecoBTag.PerformanceMeasurements.MistagAnalyzer_cff")


process.out = cms.OutputModule("PoolOutputModule",
                               outputCommands = cms.untracked.vstring('keep *', 
                                                                      'keep recoJetTags_*_*_*'),
                               fileName = cms.untracked.string('testtt.root')
                               )


#-------------------------------------
#Produce PFJets from PF2PAT

# PAT Layer 0+1
process.load("PhysicsTools.PatAlgos.patSequences_cff")

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(False))

from PhysicsTools.PatAlgos.tools.pfTools import *

postfix = "PF2PAT"  # to have only PF2PAT

usePF2PAT(process,runPF2PAT=True,
          jetAlgo='AK5', runOnMC=True, postfix=postfix,
          jetCorrections=('AK5PFchs', ['L1FastJet', 'L2Relative', 'L3Absolute'])#,
          #pvCollection=cms.InputTag('goodOfflinePrimaryVertices')
          )
process.pfPileUpPF2PAT.checkClosestZVertex = False
#applyPostfix(process,"pfPileUp",postfix).checkClosestZVertex = cms.bool(False) 

process.pfPileUpPF2PAT.Enable = True
process.pfPileUpPF2PAT.Vertices = cms.InputTag('goodOfflinePrimaryVertices')



#-------------------------------------
#Redo the primary vertex

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector

process.goodOfflinePrimaryVertices = cms.EDFilter(
  "PrimaryVertexObjectFilter",
  filterParams = pvSelector.clone( minNdof = cms.double(4.0), maxZ = cms.double(24.0) ),
  src=cms.InputTag('offlinePrimaryVertices')
  )

#-------------------------------------
#JetID
process.load('PhysicsTools.SelectorUtils.pfJetIDSelector_cfi')
process.load('PhysicsTools.SelectorUtils.jetIDSelector_cfi')
process.jetIDSelector.version = cms.string('PURE09')


#-------------------------------------
#Filter for PFJets
process.PATJetsFilter = cms.EDFilter("PATJetSelector",    
                                    src = cms.InputTag("selectedPatJetsPF2PAT"),
                                    cut = cms.string("pt > 10.0 && abs(eta) < 2.5 && neutralHadronEnergyFraction < 0.99 && neutralEmEnergyFraction < 0.99 && nConstituents > 1 && chargedHadronEnergyFraction > 0.0 && chargedMultiplicity > 0.0 && chargedEmEnergyFraction < 0.99"),
                                    #filter = cms.bool(True)
                                    )
#---------------------------------------
process.load("bTag.CommissioningCommonSetup.caloJetIDFilter_cff")


#Filter for removing scraping events
process.noscraping = cms.EDFilter("FilterOutScraping",
                                  applyfilter = cms.untracked.bool(True),
                                  debugOn = cms.untracked.bool(False),
                                  numtrack = cms.untracked.uint32(10),
                                  thresh = cms.untracked.double(0.25)
                                  )


#Filter for good primary vertex
process.load("RecoVertex.PrimaryVertexProducer.OfflinePrimaryVertices_cfi")
process.primaryVertexFilter = cms.EDFilter("GoodVertexFilter",
                                           vertexCollection = cms.InputTag('offlinePrimaryVertices'),
                                           minimumNDOF = cms.uint32(4) ,
                                           maxAbsZ = cms.double(24), 
                                           maxd0 = cms.double(2)	
                                           )


### from Cristina JP calibration for cmsRun only : 
# from CondCore.DBCommon.CondDBCommon_cfi import *
# process.load("RecoBTag.TrackProbability.trackProbabilityFakeCond_cfi")
# process.trackProbabilityFakeCond.connect =cms.string( "sqlite_fip:RecoBTag/PerformanceMeasurements/test/btagnew_Data_2011_41X.db")
# process.es_prefer_trackProbabilityFakeCond = cms.ESPrefer("PoolDBESSource","trackProbabilityFakeCond")

### from Cristina JP calibration for crab only: 
process.GlobalTag.toGet = cms.VPSet(
  cms.PSet(record = cms.string("BTagTrackProbability2DRcd"),
           tag = cms.string("TrackProbabilityCalibration_2D_2012DataTOT_v1_offline"),
           connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_BTAU")),
  cms.PSet(record = cms.string("BTagTrackProbability3DRcd"),
           tag = cms.string("TrackProbabilityCalibration_3D_2012DataTOT_v1_offline"),
           connect = cms.untracked.string("frontier://FrontierPrep/CMS_COND_BTAU"))
  )


#---------------------------------------
# process.TFileService = cms.Service("TFileService", fileName = cms.string("TrackTree.root") )
process.TFileService = cms.Service("TFileService", fileName = cms.string("JetTree.root") )

process.mistag.isData              = False
process.mistag.useTrackHistory     = False
process.mistag.produceJetProbaTree = False
process.mistag.producePtRelTemplate = False
process.mistag.triggerTable = 'TriggerResults::HLT' # Data and MC
#---------------------------------------


process.AK5byRef.jets = "PATJetsFilter"

# process.mistag.Jets = 'ak5PFJets'
process.mistag.Jets = 'PATJetsFilter'
process.mistag.jetCorrector = cms.string('ak5PFL1FastL2L3')
#process.mistag.jetCorrector = cms.string('ak5PFchsL1FastL2L3Residual')

#process.ak5JetTracksAssociatorAtVertex.jets = "selectedPatJetsPF2PAT"
#process.softMuonTagInfos.jets = "selectedPatJetsPF2PAT"

process.ak5JetTracksAssociatorAtVertex.jets = "PATJetsFilter"
process.softMuonTagInfos.jets = "PATJetsFilter"


#---------------------------------------
# trigger selection !

import HLTrigger.HLTfilters.triggerResultsFilter_cfi as hlt
process.JetHLTFilter = hlt.triggerResultsFilter.clone(
     triggerConditions = cms.vstring(
        "HLT_PFJet80_v5"
     ),
     hltResults = cms.InputTag("TriggerResults","","HLT"),
     l1tResults = cms.InputTag( "" ),
     throw = cms.bool( False) #set to false to deal with missing triggers while running over different trigger menus
     )
#---------------------------------------


process.load("RecoBTag.ImpactParameterLearning.ImpactParameterCalibration_cfi")
process.ipCalib.Jets           = cms.InputTag('PATJetsFilter')
process.ipCalib.jetCorrector   = cms.string('ak5PFL1FastL2L3')




process.p = cms.Path(
  #$$
  process.JetHLTFilter*
  
  process.offlinePrimaryVertices 
  *process.goodOfflinePrimaryVertices
  *getattr(process,"patPF2PATSequence"+postfix)
  *process.PATJetsFilter
  *process.myPartons
  *process.AK5Flavour
  *process.noscraping
  *process.primaryVertexFilter
  *process.ak5JetTracksAssociatorAtVertex
  *process.btagging
  *process.positiveOnlyJetProbabilityJetTags*process.negativeOnlyJetProbabilityJetTags
  *process.negativeTrackCountingHighEffJetTags*process.negativeTrackCountingHighPur
  *process.secondaryVertexTagInfos*process.simpleSecondaryVertexHighEffBJetTags*process.simpleSecondaryVertexHighPurBJetTags
  *process.secondaryVertexNegativeTagInfos*process.simpleSecondaryVertexNegativeHighEffBJetTags*process.simpleSecondaryVertexNegativeHighPurBJetTags
  *process.combinedSecondaryVertexNegativeBJetTags*process.combinedSecondaryVertexPositiveBJetTags
  #*process.negativeSoftMuonBJetTags*process.positiveSoftMuonBJetTags	
  *process.negativeSoftLeptonByPtBJetTags*process.positiveSoftLeptonByPtBJetTags	
  *process.negativeOnlyJetBProbabilityJetTags*process.positiveOnlyJetBProbabilityJetTags
  *process.mistag
  *process.ipCalib
  )

from PhysicsTools.PatAlgos.patEventContent_cff import patEventContent
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('patTuple.root'),
                               # save only events passing the full path
                               SelectEvents   = cms.untracked.PSet( SelectEvents = cms.vstring('p') ),
                               # save PAT Layer 1 output; you need a '*' to
                               # unpack the list of commands 'patEventContent'
                               outputCommands = cms.untracked.vstring('drop *',
                                                'keep *_*_*_Mistag'  )
                               )

#process.outpath = cms.EndPath(process.out)




## Output Module Configuration (expects a path 'p')





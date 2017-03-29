import FWCore.ParameterSet.Config as cms

process = cms.Process("rerunMVAIsolationOnMiniAOD")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.Geometry.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
	fileNames = cms.untracked.vstring(
		'root://cms-xrd-global.cern.ch//store/mc/RunIISummer16MiniAODv2/DYJetsToLL_M-50_TuneCUETP8M1_13TeV-madgraphMLM-pythia8/MINIAODSIM/PUMoriond17_80X_mcRun2_asymptotic_2016_TrancheIV_v6_ext1-v2/120000/20587787-18C4-E611-9A68-008CFA197BD4.root'
	)
)

from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants
process.load('RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi')
from RecoTauTag.RecoTau.PATTauDiscriminationByMVAIsolationRun2_cff import *
from RecoTauTag.RecoTau.PATTauDiscriminationAgainstElectronMVA6_cfi import *

process.rerunDiscriminationByIsolationMVArun2v1raw = patDiscriminationByIsolationMVArun2v1raw.clone(
	PATTauProducer = cms.InputTag('slimmedTaus'),
	Prediscriminants = noPrediscriminants,
	loadMVAfromDB = cms.bool(True),
	mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1"),
	mvaOpt = cms.string("DBoldDMwLT"),
	requireDecayMode = cms.bool(True),
	verbosity = cms.int32(0)
)

process.rerunDiscriminationByIsolationMVArun2v1VLoose = patDiscriminationByIsolationMVArun2v1VLoose.clone(
	PATTauProducer = cms.InputTag('slimmedTaus'),    
	Prediscriminants = noPrediscriminants,
	toMultiplex = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1raw'),
	key = cms.InputTag('rerunDiscriminationByIsolationMVArun2v1raw:category'),
	loadMVAfromDB = cms.bool(True),
	mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_mvaOutput_normalization"),
	mapping = cms.VPSet(
		cms.PSet(
			category = cms.uint32(0),
			cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff90"),
			variable = cms.string("pt"),
		)
	)
)

process.rerunDiscriminationAgainstElectronMVA6 = patTauDiscriminationAgainstElectronMVA6.clone(
	PATTauProducer = cms.InputTag('slimmedTaus'),
	Prediscriminants = noPrediscriminants,
	#Prediscriminants = requireLeadTrack,
	loadMVAfromDB = cms.bool(True),
	returnMVA = cms.bool(True),
	method = cms.string("BDTG"),
	mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
	mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
	mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL"),
	mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL"),
	mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC"),
	mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC"),
	mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC"),
	mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC"),
	minMVANoEleMatchWOgWOgsfBL = cms.double(0.0),
	minMVANoEleMatchWgWOgsfBL  = cms.double(0.0),
	minMVAWOgWgsfBL            = cms.double(0.0),
	minMVAWgWgsfBL             = cms.double(0.0),
	minMVANoEleMatchWOgWOgsfEC = cms.double(0.0),
	minMVANoEleMatchWgWOgsfEC  = cms.double(0.0),
	minMVAWOgWgsfEC            = cms.double(0.0),
	minMVAWgWgsfEC             = cms.double(0.0),
	srcElectrons = cms.InputTag('slimmedElectrons')
)

process.rerunDiscriminationByIsolationMVArun2v1Loose = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1Loose.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff80")
process.rerunDiscriminationByIsolationMVArun2v1Medium = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1Medium.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff70")
process.rerunDiscriminationByIsolationMVArun2v1Tight = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1Tight.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff60")
process.rerunDiscriminationByIsolationMVArun2v1VTight = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1VTight.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff50")
process.rerunDiscriminationByIsolationMVArun2v1VVTight = process.rerunDiscriminationByIsolationMVArun2v1VLoose.clone()
process.rerunDiscriminationByIsolationMVArun2v1VVTight.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2016v1_WPEff40")

process.rerunMvaIsolation2SeqRun2 = cms.Sequence(
	process.rerunDiscriminationByIsolationMVArun2v1raw
	*process.rerunDiscriminationByIsolationMVArun2v1VLoose
	*process.rerunDiscriminationByIsolationMVArun2v1Loose
	*process.rerunDiscriminationByIsolationMVArun2v1Medium
	*process.rerunDiscriminationByIsolationMVArun2v1Tight
	*process.rerunDiscriminationByIsolationMVArun2v1VTight
	*process.rerunDiscriminationByIsolationMVArun2v1VVTight
)

process.rerunMVAIsolationOnMiniAOD = cms.EDAnalyzer('rerunMVAIsolationOnMiniAOD'
)

process.rerunMVAIsolationOnMiniAOD.verbosity = cms.int32(0)
process.rerunMVAIsolationOnMiniAOD.additionalCollectionsAvailable = cms.bool(False)

process.p = cms.Path(
	process.rerunMvaIsolation2SeqRun2
	*process.rerunDiscriminationAgainstElectronMVA6
	*process.rerunMVAIsolationOnMiniAOD
)

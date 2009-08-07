import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisGammaIncJet_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisGammaIncJetOnlyMC_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('file:jet15_step1_incJet.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 'rfio:/castor/cern.ch/user/d/dcharfed/gammajet15/jet15_1.root')
)
     #'rfio:/castor/cern.ch/user/d/dcharfed/MinBias/MinBiasGen_1.root',
     #'rfio:/castor/cern.ch/user/d/dcharfed/MinBias/MinBiasGen_2.root',
     #'rfio:/castor/cern.ch/user/d/dcharfed/MinBias/MinBiasGen_3.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_4.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_5.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_6.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_7.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_8.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_9.root',
     #'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_10.root'
    #)

#)

process.p1 = cms.Path(process.UEAnalysisGammaIncJet*process.UEAnalysisJetsOnlyMC*process.UEAnalysisOnlyMC)

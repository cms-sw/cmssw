import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootpleOnlyMC_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('/tmp/lucaroni/MBUEAnalysisRootFileOnlyMC.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_1.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_2.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_3.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_4.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_5.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_6.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_7.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_8.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_9.root',
     'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/D6T/PYTHIA6_MinBias_10TeV_cff_py_GEN_10.root'
    )

)

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisJetsOnlyMC*process.UEAnalysisOnlyMC)


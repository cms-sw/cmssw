import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("FWCore.MessageService.MessageLogger_cfi")

process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJetsSISCone_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootpleOnlyMC_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFileOnlyMCProQ20.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)
process.source = cms.Source("PoolSource",
   duplicateCheckMode = cms.untracked.string('noDuplicateCheck'), 
    fileNames = cms.untracked.vstring(
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_1.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_10.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_11.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_12.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_13.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_14.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_15.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_16.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_17.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_18.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_19.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_2.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_20.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_3.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_4.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_5.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_6.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_7.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_8.root',
#'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProQ20_900/PYTHIA6_MinBiasProPT0_900GeV_cff_py_GEN_9.root'

'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_1.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_10.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_11.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_12.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_13.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_14.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_15.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_16.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_17.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_18.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_19.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_2.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_20.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_3.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_4.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_5.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_6.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_7.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_8.root',
'rfio:///?svcClass=cmscafuser&path=/castor/cern.ch/cms/store/caf/user/lucaroni/ProPT0_900/PYTHIA6_MinBiasProQ20_900GeV_cff_py_GEN_9.root'

    )

)

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisJetsOnlyMC*process.UEAnalysisOnlyMC)



process.UEAnalysisRootpleOnlyMC.GenJetCollectionName      = 'ueSisCone5GenJet'
process.UEAnalysisRootpleOnlyMC.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet'

process.UEAnalysisRootpleOnlyMC500.GenJetCollectionName      = 'ueSisCone5GenJet500'
process.UEAnalysisRootpleOnlyMC500.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet500'

process.UEAnalysisRootpleOnlyMC1500.GenJetCollectionName      = 'ueSisCone5GenJet1500'
process.UEAnalysisRootpleOnlyMC1500.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet1500'

process.UEAnalysisRootpleOnlyMC1100.GenJetCollectionName      = 'ueSisCone5GenJet1100'
process.UEAnalysisRootpleOnlyMC1100.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet1100'

process.UEAnalysisRootpleOnlyMC700.GenJetCollectionName      = 'ueSisCone5GenJet700'
process.UEAnalysisRootpleOnlyMC700.ChgGenJetCollectionName   = 'ueSisCone5ChgGenJet700'



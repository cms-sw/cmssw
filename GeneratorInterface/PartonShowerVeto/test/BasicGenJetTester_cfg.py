import FWCore.ParameterSet.Config as cms

process = cms.Process("Analysis")

# run the input file through the end;
# for a limited number of events, replace -1 with the desired number 
#
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring
   (
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run1.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run2.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run3.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run4.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run5.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run6.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run7.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run8.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run9.root',  
      'file:/storage/local/data1/condor/yarba_j/TestPSVeto/ttbar-py6/ttbar-py6-run10.root'  
   ) 
)
	      
# FileService is mandatory, as the following analyzer module 
# will want it, to create output histogram file
# 
process.TFileService = cms.Service("TFileService",
        fileName = cms.string("GenJets_MG_Py6_ttbar.root")
)

# the analyzer itself - empty parameter set 
#
process.test = cms.EDAnalyzer("BasicGenJetTester",
    qcut = cms.double(40.),
#    src = cms.InputTag("ak5GenJets")
)

process.p1 = cms.Path( process.test )


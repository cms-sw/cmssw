import FWCore.ParameterSet.Config as cms

process = cms.Process("Analysis")

# run the input file through the end;
# for a limited number of events, replace -1 with the desired number
#
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source( "PoolSource",
                             fileNames = cms.untracked.vstring(
    "file:DY_Tune4C_13TeV_pythia8_cfi_py_GEN_VALIDATION.root"
    #,"file:TestHZZ4tau_2.root"
    #,"file:TestHZZ4tau_3.root"
    )
                             )
process.load("GeneratorInterface.MCTESTER.MCTesterCMS_cfi")

process.p1 = cms.Path(process.MCTesterCMS)
      

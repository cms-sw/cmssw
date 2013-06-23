import FWCore.ParameterSet.Config as cms

process = cms.Process("Analysis")

# run the input file through the end;
# for a limited number of events, replace -1 with the desired number 
#
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load( "SimGeneral.HepPDTESSource.pythiapdt_cfi" )

process.source = cms.Source( "PoolSource",
                             fileNames = cms.untracked.vstring(
			     'file:/tmp/yarba/SL4/gcc345/TTbar_cfi_py_GEN.root'
			     )
                           )
	      
# FileService is mandatory, as the following analyzer module 
# will want it, to create output histogram file
# 
process.TFileService = cms.Service("TFileService",
        fileName = cms.string("BasicGenTester_TTbar.root")
)

# the analyzer itself - empty parameter set 
#
process.BasicGenTest = cms.EDAnalyzer( "BasicGenTester",
        NPartForHisto = cms.untracked.int32(1000),
        PtMaxForHisto = cms.untracked.double(5.0)
)

process.p1 = cms.Path( process.BasicGenTest )


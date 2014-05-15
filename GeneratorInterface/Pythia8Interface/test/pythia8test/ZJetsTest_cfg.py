import FWCore.ParameterSet.Config as cms

process = cms.Process("Analysis")

# run the input file through the end;
# for a limited number of events, replace -1 with the desired number 
#
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

#process.load( "SimGeneral.HepPDTESSource.pythiapdt_cfi" )

process.source = cms.Source( "PoolSource",
                             fileNames = cms.untracked.vstring(
			     'file:p8test1step1.root'
			     )
                           )
	      
# FileService is mandatory, as the following analyzer module 
# will want it, to create output histogram file
# 
process.TFileService = cms.Service("TFileService",
        fileName = cms.string("hist.root")
)

# the analyzer itself - empty parameter set 
#
process.TestHepMCEvt = cms.EDAnalyzer( "ZJetsTest" )

process.p1 = cms.Path( process.TestHepMCEvt )

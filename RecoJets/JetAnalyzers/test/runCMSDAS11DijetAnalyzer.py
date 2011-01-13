# PYTHON configuration file for class: CMSDAS11DijetAnalyzer.cc
# Description:  Example of simple EDAnalyzer for dijet mass & dijet spectrum ratio analysis
# Authors: J.P. Chou, Jason St. John
# Date:  01 - January - 2011
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root')
)
#############   Include the jet corrections ##########
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
# set the record's IOV. Must be defined once. Choose ANY correction service. #

#############   Correct Calo Jets on the fly #########
process.dijetAna = cms.EDAnalyzer("CMSDAS11DijetAnalyzer",
                                  jetSrc = cms.InputTag("caloJetsAK7"),
                                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                                  jetCorrections = cms.string("")
)

#############   Path       ###########################
process.p = cms.Path(process.dijetAna)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

#############  This is how CMS handles output ROOT files #################
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("histos.root")
 )



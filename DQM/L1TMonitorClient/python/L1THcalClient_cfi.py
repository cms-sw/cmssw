import FWCore.ParameterSet.Config as cms

l1THcalClient = cms.EDAnalyzer("L1THcalClient",
    #untracked bool saveOutput = true
    #untracked bool Standalone = false
    #untracked string outputFile = "L1THcalClient.root"
    output_dir = cms.untracked.string('L1T/Hcal/')
)



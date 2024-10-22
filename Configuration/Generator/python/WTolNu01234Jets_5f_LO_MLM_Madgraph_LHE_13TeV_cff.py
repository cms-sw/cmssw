import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.5/WJetsToLNu/WJetsToLNu_13TeV-madgraphMLM-pythia8_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz','false','slc6_amd64_gcc630','CMSSW_9_3_16'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(4),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    generateConcurrently = cms.untracked.bool(True),
)

#Link to datacards:
#https://github.com/cms-sw/genproductions/tree/b9f934265b462f4f3bbb8919799fa20668bb6fce/bin/MadGraph5_aMCatNLO/cards/production/13TeV/WJets_HT_LO_MLM/WJetsToLNu_HT-0toInf

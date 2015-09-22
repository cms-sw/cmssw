import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/madgraph/V5_2.2.2/WJets_HT_LO_MLM/WJetsToLNu_HT-incl/V1/WJetsToLNu_HT-incl_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

#Link to datacards:
#https://github.com/cms-sw/genproductions/tree/b9f934265b462f4f3bbb8919799fa20668bb6fce/bin/MadGraph5_aMCatNLO/cards/production/13TeV/WJets_HT_LO_MLM/WJetsToLNu_HT-0toInf

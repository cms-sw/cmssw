import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.5/wellnu012j_5f_LO_MLM/wellnu012j_5f_LO_MLM_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz','false','slc6_amd64_gcc630','CMSSW_9_3_16'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(4),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    generateConcurrently = cms.untracked.bool(True),
)

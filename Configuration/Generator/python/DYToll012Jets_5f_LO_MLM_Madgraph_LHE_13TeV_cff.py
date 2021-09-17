import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer('ExternalLHEProducer',
  scriptName = cms.FileInPath("GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh"),
  outputFile = cms.string("cmsgrid_final.lhe"),
  numberOfParameters = cms.uint32(4),
  args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.5/dyellell012j_5f_LO_MLM/dyellell012j_5f_LO_MLM_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz','false','slc6_amd64_gcc630','CMSSW_9_3_16'),
  nEvents = cms.untracked.uint32(10),
  generateConcurrently = cms.untracked.bool(True)
)

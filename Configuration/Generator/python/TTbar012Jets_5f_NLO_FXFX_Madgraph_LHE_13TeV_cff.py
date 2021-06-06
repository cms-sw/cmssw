import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/UL/13TeV/madgraph/V5_2.6.5/tt012j_5f_ckm_NLO_FXFX/tt012j_5f_ckm_NLO_FXFX_slc6_amd64_gcc630_CMSSW_9_3_16_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh'),
    generateConcurrently = cms.untracked.bool(True),
)

#Link to datacards:
#https://github.com/cms-sw/genproductions/tree/c41ab29f3d86c9e53df8b0d76c12cd519adbf013/bin/MadGraph5_aMCatNLO/cards/production/13TeV/tt012j_5f_ckm_NLO_FXFX

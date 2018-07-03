import FWCore.ParameterSet.Config as cms

# link to cards:
# https://github.com/cms-sw/genproductions/blob/6718f234d90e34ac43b683e323a3edd6e8aa72e2/bin/Powheg/production/V2/13TeV/Higgs/gg_H_quark-mass-effects_NNPDF30_13TeV/gg_H_quark-mass-effects_NNPDF30_13TeV_M125.input


externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/powheg/V2/gg_H_quark-mass-effects_NNPDF30_13TeV_M125/v2/gg_H_quark-mass-effects_NNPDF30_13TeV_M125_tarball.tar.gz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

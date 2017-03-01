import FWCore.ParameterSet.Config as cms

# link to cards:
# https://github.com/cms-sw/genproductions/blob/329fda9f8d07c2d4d4e75c9a00279dcd6e78cda7/bin/Powheg/production/VH_from_Hbb/HWplusJ_HanythingJ_NNPDF30_13TeV_M125.input

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc481/13TeV/powheg/V2/HWplusJ_HanythingJ_NNPDF30_13TeV_M125/v2_folding/HWplusJ_HanythingJ_NNPDF30_13TeV_M125_tarball.tar.gz'),
    nEvents = cms.untracked.uint32(5000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

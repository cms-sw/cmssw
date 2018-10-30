import FWCore.ParameterSet.Config as cms

# link to cards:
# https://github.com/cms-sw/genproductions/blob/060e6d3363a78ecfae90f7f5c46c968992820a56/bin/Powheg/production/V2/13TeV/Higgs/VBF_H_JHUGen_HZZ4L_NNPDF30_13TeV/VBF_H_M125_NNPDF30_13TeV.input
# https://github.com/cms-sw/genproductions/blob/060e6d3363a78ecfae90f7f5c46c968992820a56/bin/Powheg/production/V2/13TeV/Higgs/VBF_H_JHUGen_HZZ4L_NNPDF30_13TeV/JHUGen_VBF_H_ZZ4L.input

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/cvmfs/cms.cern.ch/phys_generator/gridpacks/slc6_amd64_gcc630/13TeV/Powheg/V2/RelValidation/VBFH/VBF_H_slc6_amd64_gcc630_CMSSW_9_3_9_patch1_VBF_new_reducedPdf.tgz'),
    nEvents = cms.untracked.uint32(1000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

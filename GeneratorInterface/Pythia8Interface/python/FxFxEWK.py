import FWCore.ParameterSet.Config as cms

externalLHEProducer = cms.EDProducer("ExternalLHEProducer",
    args = cms.vstring('/afs/cern.ch/user/c/cvicovil/public/TTWJetsToLNu_5f_NLO_FXFX_slc7_amd64_gcc900_CMSSW_12_0_2_tarball.tar.xz'),
    nEvents = cms.untracked.uint32(1000),
    numberOfParameters = cms.uint32(1),
    outputFile = cms.string('cmsgrid_final.lhe'),
    scriptName = cms.FileInPath('GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh')
)

# Link to datacards:
# https://github.com/cms-sw/genproductions/tree/5fb3762c8be13dabb89a8580863856201d56f49c/bin/MadGraph5_aMCatNLO/cards/production/2017/13TeV/TTWJets/TTWJetsToLNu_5f_NLO_FXFX

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from Configuration.Generator.Pythia8aMCatNLOSettings_cfi import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *

generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        pythia8aMCatNLOSettingsBlock,
        pythia8PSweightsSettingsBlock,
        processParameters = cms.vstring(
            'JetMatching:setMad = off',
            'JetMatching:scheme = 1',
            'JetMatching:doVeto = off',  # Turn OFF Pythia JetMatching
            'JetMatching:merge = on',
            'JetMatching:jetAlgorithm = 2',
            'JetMatching:etaJetMax = 1000',
            'JetMatching:coneRadius = 1.',
            'JetMatching:slowJetPower = 1',
            'JetMatching:qCut = 42.', #this is the actual merging scale
            'JetMatching:doFxFx = on',
            'JetMatching:qCutME = 20.',#this must match the ptj cut in the lhe generation step
            'JetMatching:nQmatch = 5', #4 corresponds to 4-flavour scheme (no matching of b-quarks), 5 for 5-flavour scheme
            'JetMatching:nJetMax = 1', #number of partons in born matrix element for highest multiplicity
        
        ), 
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'pythia8aMCatNLOSettings',
                                    'pythia8PSweightsSettings',
                                    'processParameters',
                                    )
    ),
    UserCustomization = cms.VPSet(
           cms.PSet(
             pluginName = cms.string("JetMatchingEWKFxFx"), # Call the new JetMatching plugin
           )
        ), 
)


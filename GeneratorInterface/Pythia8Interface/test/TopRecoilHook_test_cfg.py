import sys
import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
#import pythia8

options = VarParsing.VarParsing ('standard')
options.register('runOnly', '', VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string, "Run only specified analysis")
options.register('yodafile', 'test_toprecoil.yoda', VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string, "Name of yoda output file")
options.register('recoiltoTop','on',VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string, "TopRecoilHook setting")
options.register('recoiltoBottom','off',VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string, "RecoilToColor setting")
options.setDefault('maxEvents', 1000)
options.register('alphaSvalue', 0.1365,VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string, "AlphaS value setting")
options.register('topmass', 172.5,VarParsing.VarParsing.multiplicity.singleton,VarParsing.VarParsing.varType.string, "Top mass setting")
if(hasattr(sys, "argv")):
    options.parseArguments()
print(options)

process = cms.Process("runRivetAnalysis")

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from Configuration.Generator.Pythia8aMCatNLOSettings_cfi import *
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper
#randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
#randSvc.populate()

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')

randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
randSvc.populate()

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(options.maxEvents))

process.source = cms.Source("EmptySource")

process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.generator = cms.EDFilter("Pythia8GeneratorFilter",
     comEnergy = cms.double(13000.0),
     crossSection = cms.untracked.double(421.1),
     filterEfficiency = cms.untracked.double(1),
     maxEventsToPrint = cms.untracked.int32(0),
     pythiaHepMCVerbosity = cms.untracked.bool(False),
     pythiaPylistVerbosity = cms.untracked.int32(1),
     PythiaParameters = cms.PSet(
       pythia8CommonSettingsBlock,
       pythia8CUEP8M1SettingsBlock,
       pythia8aMCatNLOSettingsBlock,
       processParameters = cms.vstring(
             'Main:timesAllowErrors = 10000',
             'ParticleDecays:limitTau0 = on',
             'ParticleDecays:tauMax = 10',
             'Tune:ee 7',
             'Tune:pp 14',      # Monash tune
             'Top:gg2ttbar    = on',
             'Top:qqbar2ttbar = on',
             '6:m0 =  %s'%options.topmass,    # top mass'
             'TopRecoilHook:doTopRecoilIn = %s'%options.recoiltoTop,
             'TimeShower:recoilToColoured = %s'%options.recoiltoBottom,
             'TimeShower:alphaSvalue = %s' %options.alphaSvalue
       ),
         parameterSets = cms.vstring('processParameters',)
     )
)

process.load("GeneratorInterface.RivetInterface.rivetAnalyzer_cfi")

#pythia.Settings.listAll()

if options.runOnly:
    process.rivetAnalyzer.AnalysisNames = cms.vstring(options.runOnly)
else:
    process.rivetAnalyzer.AnalysisNames = cms.vstring(
        'CMS_2018_I1663958', # diff xs lepton+jets
        'CMS_2018_I1662081', # event variables lepton+jets
        'MC_TOPMASS_LJETS', # MC analysis for lepton+jets top mass
        'CMS_2018_I1690148',  # jet substructure
        'MC_BFRAG_LJETS',  # b fragmentation
        'CMS_2018_I1703993', # diff xs dilepton
        'CMS_2019_I1764472', #boosted top mass
        'MC_TOPMASS_LSMT', #dilepton invariant mass, soft muon +primary
        'ATLAS_2019_I1724098', #missing momentum analysis
    )
process.rivetAnalyzer.OutputFile      = options.yodafile
process.rivetAnalyzer.HepMCCollection = cms.InputTag("generator:unsmeared")
process.rivetAnalyzer.CrossSection    = 831.76 # NNLO (arXiv:1303.6254)

process.p = cms.Path(process.generator*process.rivetAnalyzer)



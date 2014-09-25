# Copied from https://github.com/cms-sw/genproductions for RelVal June 5, 2014
import FWCore.ParameterSet.Config as cms
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

generator = cms.EDFilter("Pythia8HadronizerFilter",
                         ExternalDecays = cms.PSet(
    Tauola = cms.untracked.PSet(
    TauolaPolar,
    TauolaDefaultInputCards
    ),
    parameterSets = cms.vstring('Tauola')
    ),
                         UseExternalGenerators = cms.untracked.bool(True),
                         maxEventsToPrint = cms.untracked.int32(1),
                         pythiaPylistVerbosity = cms.untracked.int32(1),
                         filterEfficiency = cms.untracked.double(1.0),
                         pythiaHepMCVerbosity = cms.untracked.bool(False),
                         comEnergy = cms.double(13000.),
                         jetMatching = cms.untracked.PSet(
    scheme = cms.string("Madgraph"),
    mode = cms.string("auto"),# soup, or "inclusive" / "exclusive"
    MEMAIN_etaclmax = cms.double(-1),
    MEMAIN_qcut = cms.double(-1),
    MEMAIN_minjets = cms.int32(-1),
    MEMAIN_maxjets = cms.int32(-1),
    MEMAIN_showerkt = cms.double(0), # use 1=yes only for pt-ordered showers !
    MEMAIN_nqmatch = cms.int32(5), #PID of the flavor until which the QCD radiation are kept in the matching procedure;
    # if nqmatch=4, then all showered partons from b's are NOT taken into account
    # Note (JY): I think the default should be 5 (b); anyway, don't try -1 as it'll result in a throw...
    MEMAIN_excres = cms.string(""),
    outTree_flag = cms.int32(0) # 1=yes, write out the tree for future sanity check
    ),
                         PythiaParameters = cms.PSet(
    processParameters = cms.vstring(
    'Main:timesAllowErrors = 10000',
    'ParticleDecays:limitTau0 = on',
    'ParticleDecays:tauMax = 10',
    'Tune:ee 3',
    'Tune:pp 5'),
    parameterSets = cms.vstring('processParameters')
    )
                         )

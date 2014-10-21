import FWCore.ParameterSet.Config as cms

from JetMETCorrections.Configuration.DefaultJEC_cff import *
from Configuration.Skimming.pdwgLeptonRecoSkim_cfi import *

from Configuration.Skimming.pdwgSingleMu_cfi import *
from Configuration.Skimming.pdwgDoubleMu_cfi import *
from Configuration.Skimming.pdwgMuonElectron_cfi import *
from Configuration.Skimming.pdwgMuonPFElectron_cfi import *
from Configuration.Skimming.pdwgDoubleElectron_cfi import *
from Configuration.Skimming.pdwgDoublePFElectron_cfi import *

filterSingleMu         = cms.Sequence(ak4CaloJetsL2L3+ak4PFJetsL2L3+SingleMu)
filterDoubleMu         = cms.Sequence(ak4CaloJetsL2L3+ak4PFJetsL2L3+DoubleMu)
filterMuonElectron     = cms.Sequence(ak4CaloJetsL2L3+ak4PFJetsL2L3+MuonElectron)
filterMuonPFElectron   = cms.Sequence(ak4CaloJetsL2L3+ak4PFJetsL2L3+MuonPFElectron)
filterDoubleElectron   = cms.Sequence(ak4CaloJetsL2L3+ak4PFJetsL2L3+DoubleElectron)
filterDoublePFElectron = cms.Sequence(ak4CaloJetsL2L3+ak4PFJetsL2L3+DoublePFElectron)

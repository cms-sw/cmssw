import FWCore.ParameterSet.Config as cms

# Temporary code to load database if CMSSW version < 3_11
from RecoTauTag.TauTagTools.TancConditions_cff import TauTagMVAComputerRecord
TauTagMVAComputerRecord.connect = cms.string(
    'sqlite_fip:RecoTauTag/RecoTau/data/hpstanc.db'
)
TauTagMVAComputerRecord.toGet[0].tag = cms.string('Tanc')
# Don't conflict with TaNC global tag
TauTagMVAComputerRecord.appendToDataLabel = cms.string('hpstanc')

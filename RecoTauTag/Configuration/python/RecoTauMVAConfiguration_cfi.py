import FWCore.ParameterSet.Config as cms

# Flag to disable/enable MVA dependent tau sequences
recoTauEnableMVA = True

# Temporary code to load database if CMSSW version < 3_11
# In 3_11 this has entered the global tag.
TauTagMVAComputerRecord = None
#from RecoTauTag.TauTagTools.TancConditions_cff import TauTagMVAComputerRecord
#TauTagMVAComputerRecord.connect = cms.string(
   #'sqlite_fip:RecoTauTag/RecoTau/data/hpstanc.db'
#)
#TauTagMVAComputerRecord.toGet[0].tag = cms.string('Tanc')
## Don't conflict with TaNC global tag
#TauTagMVAComputerRecord.appendToDataLabel = cms.string('hpstanc')
# Provide hook to enable a local database at the same time as the GT
esPreferLocalTancDB = None

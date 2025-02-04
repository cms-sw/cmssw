import FWCore.ParameterSet.Config as cms
from HLTriggerOffline.Scouting. ScoutingEGammaCollectionMonitoring_cfi import *
from HLTriggerOffline.Scouting.ScoutingElectronTagProbeAnalyzer_cfi import *
from HLTriggerOffline.Scouting.PatElectronTagProbeAnalyzer_cfi import *

from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cff import egmGsfElectronIDs

egmGsfElectronIDsForScoutingDQM = egmGsfElectronIDs.clone()
egmGsfElectronIDsForScoutingDQM.physicsObjectsIDs = cms.VPSet()
egmGsfElectronIDsForScoutingDQM.physicsObjectSrc = cms.InputTag('slimmedElectrons')
#note: be careful here to when selecting new ids that the vid tools dont do extra setup for them
#for example the HEEP cuts need an extra producer which vid tools automatically handles
from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff']
for id_module_name in my_id_modules:
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupVIDSelection(egmGsfElectronIDsForScoutingDQM,item)

hltScoutingEGammaDqmOffline = cms.Sequence(egmGsfElectronIDsForScoutingDQM + scoutingMonitoringEGM + scoutingMonitoringTagProbe + scoutingMonitoringPatElectronTagProbe)

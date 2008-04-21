import FWCore.ParameterSet.Config as cms

from HLTrigger.Configuration.main.JetMET_cff import *
from HLTrigger.Configuration.main.Egamma_cff import *
from HLTrigger.Configuration.main.Muon_cff import *
from HLTrigger.Configuration.main.BTau_cff import *
from HLTrigger.Configuration.main.Xchannel_cff import *
from HLTrigger.Configuration.main.Special_cff import *
TriggerFinalPath = cms.Path(cms.SequencePlaceholder("triggerFinalPath"))


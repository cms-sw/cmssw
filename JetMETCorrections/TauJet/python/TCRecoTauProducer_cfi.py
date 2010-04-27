import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackAssociator.DetIdAssociatorESProducer_cff import *
#from TrackingTools.TrackAssociator.default_cfi import TrackAssociatorParameterBlock
from JetMETCorrections.TauJet.TCTauAlgoParameters_cfi import *

tcRecoTauProducer = cms.EDProducer("TCRecoTauProducer",
        tcTauAlgoParameters
)



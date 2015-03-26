import FWCore.ParameterSet.Config as cms

from RecoJets.JetProducers.ak4JetID_cfi import *
from RecoJets.JetProducers.ak7JetID_cfi import *
from RecoJets.JetProducers.ca4JetID_cfi import *
from RecoJets.JetProducers.ca6JetID_cfi import *
from RecoJets.JetProducers.gk5JetID_cfi import *
from RecoJets.JetProducers.gk7JetID_cfi import *
from RecoJets.JetProducers.ic5JetID_cfi import *
from RecoJets.JetProducers.ic7JetID_cfi import *
from RecoJets.JetProducers.kt4JetID_cfi import *
from RecoJets.JetProducers.kt6JetID_cfi import *
from RecoJets.JetProducers.sc5JetID_cfi import *
from RecoJets.JetProducers.sc7JetID_cfi import *

recoAllJetIds = cms.Sequence( ak4JetID + ak7JetID + sc5JetID + sc7JetID + ic5JetID +
                              kt4JetID + #kt6JetID +
                              ca4JetID + ca6JetID + gk5JetID + gk7JetID )

recoJetIds = cms.Sequence( ak4JetID
			  )

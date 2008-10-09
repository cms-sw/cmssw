import FWCore.ParameterSet.Config as cms

from DQMServices.Components.MEtoEDMConverter_cfi import *
endOfProcess=cms.Sequence(MEtoEDMConverter)


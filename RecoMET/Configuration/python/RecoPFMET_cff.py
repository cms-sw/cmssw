import FWCore.ParameterSet.Config as cms

# Name:   RecoPFMET.cff
# Author: R.Cavanaugh
# Date:   30.10.2008
# Notes:  PFMET.cfi assumes that a product with label "particleFlow" is
#         already written into the event.
from RecoMET.METProducers.PFMET_cfi import *

recoPFMET = cms.Sequence( pfMet )

import FWCore.ParameterSet.Config as cms

# Name :   RecoMET_BeamHaloId_cff.py
# Original Author : R. Remington,  UF
# Date : 11.11.2009
# Notes :  Information to Id/reconstruct any halo that may be in the event


from RecoMET.METProducers.CSCHaloData_cfi import *
from RecoMET.METProducers.EcalHaloData_cfi import *
from RecoMET.METProducers.HcalHaloData_cfi import *
from RecoMET.METProducers.GlobalHaloData_cfi import *
from RecoMET.METProducers.BeamHaloSummary_cfi import *
#from L1Trigger.Configuration.L1RawToDigi_cff import *

BeamHaloId = cms.Sequence(CSCHaloData*EcalHaloData*HcalHaloData*GlobalHaloData*BeamHaloSummary)

# Needs FEVT content
#BeamHaloIdWithGtRecord = cms.Sequence(gtDigis*l1GtRecord*CSCHaloData*EcalHaloData*HcalHaloData*GlobalHaloData*BeamHaloSummary)

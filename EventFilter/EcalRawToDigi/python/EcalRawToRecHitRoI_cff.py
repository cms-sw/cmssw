import FWCore.ParameterSet.Config as cms

# geometry tools
from L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi import *
from L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff import *
#cabling tools
from EventFilter.EcalRawToDigi.EcalRegionCablingESProducer_cff import *
#region of interest FEDS + RefGetter
from EventFilter.EcalRawToDigi.ecalRegionalEgammaFEDs_cfi import *
from EventFilter.EcalRawToDigi.ecalRegionalMuonsFEDs_cfi import *
from EventFilter.EcalRawToDigi.ecalRegionalJetsFEDs_cfi import *
from EventFilter.EcalRawToDigi.ecalRegionalTausFEDs_cfi import *
from EventFilter.EcalRawToDigi.ecalRegionalRestFEDs_cfi import *


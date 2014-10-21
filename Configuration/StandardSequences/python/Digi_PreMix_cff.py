import FWCore.ParameterSet.Config as cms

# fragment to turn off ZS in Hcal:
#from Configuration.StandardSequences.Digi_cff import *
from Configuration.StandardSequences.DigiNZS_cff import *

# modifications to the digi sequences defined above (DigiNZS imports the central Digi_cff)

#simMuonCSCDigis.strips.doNoise = False
#simMuonCSCDigis.wires.doNoise = False
#simMuonDTDigis.onlyMuHits = True
#simMuonRPCDigis.Noise = False

# Note: the other noise is turned of in the DigitizersNoNoise sequence defined in the MixingModule
# because the MM holds/controls all of the other digitizers.

# Turn off SR in Ecal
simEcalDigis.UseFullReadout = cms.bool(True)
# This is extra, since the configuration skips it anyway.  Belts and suspenders.
pdigi.remove(simEcalPreshowerDigis)


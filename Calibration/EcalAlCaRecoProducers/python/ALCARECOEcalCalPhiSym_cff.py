import FWCore.ParameterSet.Config as cms

#------------------------------------------------
#AlCaReco filtering for phi symmetry calibration:
#------------------------------------------------
# create sequence for rechit filtering for phi symmetry calibration
from Calibration.EcalAlCaRecoProducers.alcastreamEcalPhiSym_cff import *
seqALCARECOEcalCalPhiSym = cms.Sequence(ecalphiSymHLT)


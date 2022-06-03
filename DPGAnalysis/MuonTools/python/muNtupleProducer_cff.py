import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *

from DPGAnalysis.MuonTools.nano_mu_digi_cff import *
from DPGAnalysis.MuonTools.nano_mu_local_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_reco_cff import *
from DPGAnalysis.MuonTools.nano_mu_l1t_cff import *

muNtupleProducer = cms.Sequence(muDigiProducers 
                                + muLocalRecoProducers 
                                + muRecoProducers
                                + muL1TriggerProducers
                               )
                               
def nanoAOD_customizeCommon(process) :
     
     if hasattr(process, "muGEMMuonExtTableProducer") or hasattr(process, "muCSCTnPFlatTableProducer"):
          process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAlong_cfi")
          process.load("TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorOpposite_cfi")

     if hasattr(process, "NANOAODoutput"):
          process.NANOAODoutput.outputCommands.append("keep nanoaodFlatTable_*Table*_*_*")
          process.NANOAODoutput.outputCommands.append("drop edmTriggerResults_*_*_*")

     return process

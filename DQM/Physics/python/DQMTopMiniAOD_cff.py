import FWCore.ParameterSet.Config as cms

from DQM.Physics.topSingleLeptonDQM_miniAOD_cfi import *

topPhysicsminiAOD = cms.Sequence(    topSingleMuonMediumDQM_miniAOD 
                                     *topSingleElectronMediumDQM_miniAOD 
                                 )

import FWCore.ParameterSet.Config as cms

from DQM.Physics.topSingleLeptonDQM_miniAOD_cfi import *

topPhysicsminiAOD = cms.Sequence(    process.topSingleMuonMediumDQM_miniAOD +
                                     process.topSingleElectronMediumDQM_miniAOD 
                                 )

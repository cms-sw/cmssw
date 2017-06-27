import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

muRecoTest = DQMEDHarvester("MuonRecoTest",
                            phiMin = cms.double(-3.2),
                            # number of luminosity block to analyse
                            diagnosticPrescale = cms.untracked.int32(1),
                            etaMin = cms.double(-3.0),
                            efficiencyTestName = cms.untracked.string('EfficiencyInRange'),
                            # histo binning
                            etaBin = cms.int32(100),
                            phiBin = cms.int32(100),
                            etaMax = cms.double(3.0),
                            phiMax = cms.double(3.2)
                            )




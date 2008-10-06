import FWCore.ParameterSet.Config as cms

source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(
    '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/0A59D9FF-A92B-DD11-8611-001A6434F19C.root',
    '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/2862D945-AB2B-DD11-832D-001A644EB282.root',
    '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/56090B0B-B72B-DD11-88ED-001A64894E06.root',
    '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/AADEE16A-A72B-DD11-839D-00096BB5DCA0.root',
    '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/E8248443-E42B-DD11-A343-00096BB5B7BA.root', 
    '/store/mc/CSA08/Zmumu/GEN-SIM-RECO/CSA08_S156_v1/0002/E84DF339-A12B-DD11-9A09-00145EED0A1C.root' )
                    )


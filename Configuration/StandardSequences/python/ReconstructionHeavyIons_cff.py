import FWCore.ParameterSet.Config as cms

#--------------------------------------------------------------------------
# LOCAL RECO

# Tracker
from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *
from RecoLuminosity.LumiProducer.bunchSpacingProducer_cfi import *
from RecoLocalTracker.Configuration.RecoLocalTrackerHeavyIons_cff import *
from RecoTracker.MeasurementDet.MeasurementTrackerEventProducer_cfi import *
from RecoTracker.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *

# Ecal
from RecoLocalCalo.Configuration.ecalLocalRecoSequence_cff import *
from RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi import *

# Hcal
from RecoLocalCalo.Configuration.hcalLocalReco_cff import *
from RecoLocalCalo.Configuration.hcalLocalRecoNZS_cff import *

#castor
from RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi import *

# Muons
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *

#--------------------------------------------------------------------------
# HIGH LEVEL RECO

from RecoHI.Configuration.Reconstruction_HI_cff import *
from RecoHI.Configuration.Reconstruction_hiPF_cff import *
from RecoLocalCalo.Castor.Castor_cff import *
from RecoHI.HiEgammaAlgos.HiElectronSequence_cff import *
#--------------------------------------------------------------------------

from RecoTracker.PixelLowPtUtilities.siPixelClusterShapeCache_cfi import *
siPixelClusterShapeCachePreSplitting = siPixelClusterShapeCache.clone(
    src = 'siPixelClustersPreSplitting'
    )

caloRecoTask = cms.Task(ecalLocalRecoTask,hcalLocalRecoTask)
muonRecoTask = cms.Task(trackerlocalrecoTask,MeasurementTrackerEventPreSplitting,siPixelClusterShapeCachePreSplitting,muonlocalrecoTask)
localRecoTask = cms.Task(bunchSpacingProducer,offlineBeamSpot,muonRecoTask,caloRecoTask,castorreco)

#hbherecoMB = hbheprerecoMB.clone()
#hcalLocalRecoSequenceNZS.replace(hbheprerecoMB,hbherecoMB)

caloRecoNZSTask = cms.Task(caloRecoTask,hcalLocalRecoTaskNZS)
localReco_HcalNZSTask = cms.Task(bunchSpacingProducer,offlineBeamSpot,muonRecoTask,caloRecoNZSTask)

#--------------------------------------------------------------------------
# Main Sequence
reconstruct_PbPbTask = cms.Task(localRecoTask,CastorFullRecoTask,globalRecoPbPbTask)
reconstructionHeavyIons = cms.Sequence(reconstruct_PbPbTask)

reconstructionHeavyIons_HcalNZSTask = cms.Task(localReco_HcalNZSTask,globalRecoPbPbTask)

reconstructionHeavyIons_withRegitMu = cms.Sequence(reconstructionHeavyIons*regionalMuonRecoPbPb)
#--------------------------------------------------------------------------

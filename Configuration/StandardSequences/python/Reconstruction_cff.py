import FWCore.ParameterSet.Config as cms

from RecoLuminosity.LumiProducer.lumiProducer_cff import *
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
from RecoLocalCalo.Configuration.RecoLocalCalo_cff import *
from RecoTracker.Configuration.RecoTracker_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from TrackingTools.Configuration.TrackingTools_cff import *
# Global  reco
from RecoEcal.Configuration.RecoEcal_cff import *
from RecoJets.Configuration.CaloTowersRec_cff import *
from RecoMET.Configuration.RecoMET_cff import *
from RecoMuon.Configuration.RecoMuon_cff import *
# Higher level objects
from RecoVertex.Configuration.RecoVertex_cff import *
from RecoEgamma.Configuration.RecoEgamma_cff import *
from RecoPixelVertexing.Configuration.RecoPixelVertexing_cff import *


from RecoJets.Configuration.RecoJetsGlobal_cff import *
from RecoMET.Configuration.RecoPFMET_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *
#
# please understand that division global,highlevel is completely fake !
#
#local reconstruction
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
from RecoParticleFlow.Configuration.RecoParticleFlow_cff import *
#
# new tau configuration
#
from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.Configuration.RecoTauTag_cff import *
# Also BeamSpot
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

from RecoLocalCalo.CastorReco.CastorSimpleReconstructor_cfi import *

localreco = cms.Sequence(trackerlocalreco+muonlocalreco+calolocalreco+castorreco+lumiProducer)
localreco_HcalNZS = cms.Sequence(trackerlocalreco+muonlocalreco+calolocalrecoNZS+castorreco+lumiProducer)

#
# temporarily switching off recoGenJets; since this are MC and wil be moved to a proper sequence
#

from RecoLocalCalo.Castor.Castor_cff import *
from RecoLocalCalo.Configuration.hcalGlobalReco_cff import *

globalreco = cms.Sequence(offlineBeamSpot+recopixelvertexing+trackingGlobalReco+hcalGlobalRecoSequence+particleFlowCluster+ecalClusters+caloTowersRec*vertexreco*electronGsfTracking*egammarecoGlobal*jetGlobalReco+muonrecoComplete+muoncosmicreco+CastorFullReco)
globalreco_plusRS = cms.Sequence(globalreco*rstracks)
globalreco_plusPL= cms.Sequence(globalreco*ctfTracksPixelLess)


reducedRecHits = cms.Sequence ( reducedEcalRecHitsSequence * reducedHcalRecHitsSequence )

highlevelreco = cms.Sequence(particleFlowReco*reducedRecHits*egammarecoFull*jetHighLevelReco*tautagging*metrecoPlusHCALNoise*btagging*recoPFMET*PFTau*regionalCosmicTracksSeq*muoncosmichighlevelreco)


from FWCore.Modules.logErrorHarvester_cfi import *

# "Export" Section
reconstruction         = cms.Sequence(localreco        *globalreco       *highlevelreco*logErrorHarvester)

#need a fully expanded sequence copy
reconstruction_fromRECO = reconstruction.expandAndClone() # copy does not work well
reconstruction_fromRECO.remove(siPixelClusters)
reconstruction_fromRECO.remove(siStripZeroSuppression)
reconstruction_fromRECO.remove(siStripClusters)
reconstruction_fromRECO.remove(dt1DRecHits)
reconstruction_fromRECO.remove(dt1DCosmicRecHits)
reconstruction_fromRECO.remove(csc2DRecHits)
reconstruction_fromRECO.remove(rpcRecHits)
reconstruction_fromRECO.remove(ecalGlobalUncalibRecHit)
reconstruction_fromRECO.remove(ecalDetIdToBeRecovered)
reconstruction_fromRECO.remove(ecalRecHit)
reconstruction_fromRECO.remove(ecalCompactTrigPrim)
reconstruction_fromRECO.remove(ecalTPSkim)
reconstruction_fromRECO.remove(ecalPreshowerRecHit)
reconstruction_fromRECO.remove(selectDigi)
reconstruction_fromRECO.remove(hbheprereco)
reconstruction_fromRECO.remove(hbhereco)
reconstruction_fromRECO.remove(hfreco)
reconstruction_fromRECO.remove(horeco)
reconstruction_fromRECO.remove(hcalnoise)
reconstruction_fromRECO.remove(zdcreco)
reconstruction_fromRECO.remove(castorreco)
#reconstruction_fromRECO.remove()
#reconstruction_fromRECO.remove()
#reconstruction_fromRECO.remove()


#sequences with additional stuff
reconstruction_withPixellessTk  = cms.Sequence(localreco        *globalreco_plusPL*highlevelreco*logErrorHarvester)
reconstruction_withRS  = cms.Sequence(localreco        *globalreco_plusRS*highlevelreco*logErrorHarvester)
reconstruction_HcalNZS = cms.Sequence(localreco_HcalNZS*globalreco       *highlevelreco*logErrorHarvester)

#sequences without some stuffs
#
reconstruction_woCosmicMuons = cms.Sequence(localreco*globalreco*highlevelreco       *logErrorHarvester)


# define a standard candle. please note I am picking up individual
# modules instead of sequences
#
reconstruction_standard_candle = cms.Sequence(localreco*globalreco*vertexreco*recoJetAssociations*btagging*electronSequence*photonSequence)



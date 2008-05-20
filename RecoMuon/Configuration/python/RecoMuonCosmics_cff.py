import FWCore.ParameterSet.Config as cms

# All the services like the magnitc field, the geometries and so on are included
# in the *.cff
from RecoMuon.StandAloneMuonProducer.standAloneCosmicMuons_cff import *
# Seed generator
from RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi import *
# Stand alone muon track producer
from RecoMuon.CosmicMuonProducer.cosmicMuons_cff import *
# Global muon track producer
from RecoMuon.CosmicMuonProducer.globalCosmicMuons_cff import *
# Muon Id producer
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *
import RecoMuon.MuonIdentification.muons_cfi
STAMuons = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
TKMuons = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
GLBMuons = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
# Muon Isolation sequence
#include "RecoMuon/MuonIsolationProducers/data/muIsolation.cff"
#
# Muon Reconstruction plus Isolation
#sequence muonrecoforcosmics_plus_isolation = {muonrecoforcosmics,muIsolation}
# Alternative cosmic muon reconstruction
# Temp sequences until the endcaps are not close to the barrel  ###
# Only barrel
CosmicMuonSeedBarrelOnly = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsBarrelOnly = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
import RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi
globalCosmicMuonsBarrelOnly = RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi.globalCosmicMuons.clone()
import RecoMuon.MuonIdentification.muons_cfi
STAMuonsBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
GLBMuonsBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
muonsBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
# Only endcaps
CosmicMuonSeedEndCapsOnly = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsEndCapsOnly = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
import RecoMuon.MuonIdentification.muons_cfi
STAMuonsEndCapsOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
# Only barrel 1 leg mode
cosmicMuons1LegBarrelOnly = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
import RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi
globalCosmicMuons1LegBarrelOnly = RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi.globalCosmicMuons.clone()
import RecoMuon.MuonIdentification.muons_cfi
STAMuons1LegBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
GLBMuons1LegBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
muons1LegBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
# Standard reco
from RecoMuon.Configuration.RecoMuon_cff import *
import RecoMuon.MuonSeedGenerator.standAloneMuonSeedProducer_cfi
lhcMuonSeedBarrelOnly = RecoMuon.MuonSeedGenerator.standAloneMuonSeedProducer_cfi.MuonSeed.clone()
import RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi
lhcStandAloneMuonsBarrelOnly = RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi.standAloneMuons.clone()
import RecoMuon.MuonSeedGenerator.standAloneMuonSeedProducer_cfi
lhcMuonSeedEndCapsOnly = RecoMuon.MuonSeedGenerator.standAloneMuonSeedProducer_cfi.MuonSeed.clone()
import RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi
lhcStandAloneMuonsEndCapsOnly = RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi.standAloneMuons.clone()
import RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi
# Only barrel No drift
CosmicMuonSeedNoDriftBarrelOnly = RecoMuon.MuonSeedGenerator.CosmicMuonSeedProducer_cfi.CosmicMuonSeed.clone()
import RecoMuon.CosmicMuonProducer.cosmicMuons_cfi
cosmicMuonsNoDriftBarrelOnly = RecoMuon.CosmicMuonProducer.cosmicMuons_cfi.cosmicMuons.clone()
import RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi
globalCosmicMuonsNoDriftBarrelOnly = RecoMuon.CosmicMuonProducer.globalCosmicMuons_cfi.globalCosmicMuons.clone()
import RecoMuon.MuonIdentification.muons_cfi
STAMuonsNoDriftBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
GLBMuonsNoDriftBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
import RecoMuon.MuonIdentification.muons_cfi
muonsNoDriftBarrelOnly = RecoMuon.MuonIdentification.muons_cfi.muons.clone()
STAmuontrackingforcosmics = cms.Sequence(CosmicMuonSeed*cosmicMuons)
muontrackingforcosmics = cms.Sequence(STAmuontrackingforcosmics*globalCosmicMuons)
allmuons = cms.Sequence(muons*STAMuons*TKMuons*GLBMuons)
muonrecoforcosmics = cms.Sequence(muontrackingforcosmics*allmuons)
STAmuonrecoforcosmics = cms.Sequence(STAmuontrackingforcosmics*STAMuons)
STAmuontrackingforcosmicsBarrelOnly = cms.Sequence(CosmicMuonSeedBarrelOnly*cosmicMuonsBarrelOnly)
muontrackingforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*globalCosmicMuonsBarrelOnly)
allmuonsBarrelOnly = cms.Sequence(muonsBarrelOnly*STAMuonsBarrelOnly*GLBMuonsBarrelOnly)
muonrecoforcosmicsBarrelOnly = cms.Sequence(muontrackingforcosmicsBarrelOnly*allmuonsBarrelOnly)
STAmuonrecoforcosmicsBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsBarrelOnly*STAMuonsBarrelOnly)
STAmuontrackingforcosmicsEnsCapsOnly = cms.Sequence(CosmicMuonSeedEndCapsOnly*cosmicMuonsEndCapsOnly)
muontrackingforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly)
allmuonsEndCapsOnly = cms.Sequence(STAMuonsEndCapsOnly)
muonrecoforcosmicsEndCapsOnly = cms.Sequence(muontrackingforcosmicsEndCapsOnly*allmuonsEndCapsOnly)
STAmuonrecoforcosmicsEndCapsOnly = cms.Sequence(STAmuontrackingforcosmicsEnsCapsOnly*STAMuonsEndCapsOnly)
STAmuontrackingforcosmics1LegBarrelOnly = cms.Sequence(CosmicMuonSeedBarrelOnly*cosmicMuons1LegBarrelOnly)
muontrackingforcosmics1LegBarrelOnly = cms.Sequence(STAmuontrackingforcosmics1LegBarrelOnly*globalCosmicMuons1LegBarrelOnly)
allmuons1LegBarrelOnly = cms.Sequence(muons1LegBarrelOnly*STAMuons1LegBarrelOnly*GLBMuons1LegBarrelOnly)
muonrecoforcosmics1LegBarrelOnly = cms.Sequence(muontrackingforcosmics1LegBarrelOnly*allmuons1LegBarrelOnly)
STAmuonrecoforcosmics1LegBarrelOnly = cms.Sequence(STAmuontrackingforcosmics1LegBarrelOnly*STAMuons1LegBarrelOnly)
lhcMuonBarrelOnly = cms.Sequence(lhcMuonSeedBarrelOnly*lhcStandAloneMuonsBarrelOnly)
lhcMuonEndCapsOnly = cms.Sequence(lhcMuonSeedEndCapsOnly*lhcStandAloneMuonsEndCapsOnly)
STAmuontrackingforcosmicsNoDriftBarrelOnly = cms.Sequence(CosmicMuonSeedNoDriftBarrelOnly*cosmicMuonsNoDriftBarrelOnly)
muontrackingforcosmicsNoDriftBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsNoDriftBarrelOnly*globalCosmicMuonsNoDriftBarrelOnly)
allmuonsNoDriftBarrelOnly = cms.Sequence(muonsNoDriftBarrelOnly*STAMuonsNoDriftBarrelOnly*GLBMuonsNoDriftBarrelOnly)
muonrecoforcosmicsNoDriftBarrelOnly = cms.Sequence(muontrackingforcosmicsNoDriftBarrelOnly*allmuonsNoDriftBarrelOnly)
STAmuonrecoforcosmicsNoDriftBarrelOnly = cms.Sequence(STAmuontrackingforcosmicsNoDriftBarrelOnly*STAMuonsNoDriftBarrelOnly)
muonRecoAllGR = cms.Sequence(muonrecoforcosmics)
muonRecoBarrelGR = cms.Sequence(muonrecoforcosmicsBarrelOnly+muonrecoforcosmics1LegBarrelOnly+muonrecoforcosmicsNoDriftBarrelOnly)
muonRecoEndCapsGR = cms.Sequence(muonrecoforcosmicsEndCapsOnly)
muonRecoLHC = cms.Sequence(lhcMuonBarrelOnly*lhcMuonEndCapsOnly)
muonRecoGR = cms.Sequence(muonRecoAllGR*muonRecoBarrelGR*muonRecoEndCapsGR*muonRecoLHC)
globalCosmicMuons.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
muons.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons', 'cosmicMuons']
muons.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons.fillIsolation = False
STAMuons.inputCollectionLabels = ['cosmicMuons']
STAMuons.inputCollectionTypes = ['outer tracks']
STAMuons.fillIsolation = False
TKMuons.inputCollectionLabels = ['ctfWithMaterialTracksP5']
TKMuons.inputCollectionTypes = ['inner tracks']
TKMuons.fillIsolation = False
GLBMuons.inputCollectionLabels = ['globalCosmicMuons']
GLBMuons.inputCollectionTypes = ['links']
GLBMuons.fillIsolation = False
CosmicMuonSeedBarrelOnly.EnableCSCMeasurement = False
cosmicMuonsBarrelOnly.TrajectoryBuilderParameters.EnableCSCMeasurement = False
cosmicMuonsBarrelOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedBarrelOnly'
globalCosmicMuonsBarrelOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsBarrelOnly.MuonCollectionLabel = 'cosmicMuonsBarrelOnly'
STAMuonsBarrelOnly.inputCollectionLabels = ['cosmicMuonsBarrelOnly']
STAMuonsBarrelOnly.inputCollectionTypes = ['outer tracks']
STAMuonsBarrelOnly.fillIsolation = False
GLBMuonsBarrelOnly.inputCollectionLabels = ['globalCosmicMuonsBarrelOnly']
GLBMuonsBarrelOnly.inputCollectionTypes = ['links']
GLBMuonsBarrelOnly.fillIsolation = False
muonsBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsBarrelOnly', 'cosmicMuonsBarrelOnly']
muonsBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsBarrelOnly.fillIsolation = False
CosmicMuonSeedEndCapsOnly.EnableDTMeasurement = False
cosmicMuonsEndCapsOnly.TrajectoryBuilderParameters.EnableDTMeasurement = False
cosmicMuonsEndCapsOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedEndCapsOnly'
STAMuonsEndCapsOnly.inputCollectionLabels = ['cosmicMuonsEndCapsOnly']
STAMuonsEndCapsOnly.inputCollectionTypes = ['outer tracks']
STAMuonsEndCapsOnly.fillIsolation = False
cosmicMuons1LegBarrelOnly.TrajectoryBuilderParameters.EnableCSCMeasurement = False
cosmicMuons1LegBarrelOnly.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuons1LegBarrelOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedBarrelOnly'
globalCosmicMuons1LegBarrelOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuons1LegBarrelOnly.MuonCollectionLabel = 'cosmicMuons1LegBarrelOnly'
STAMuons1LegBarrelOnly.inputCollectionLabels = ['cosmicMuons1LegBarrelOnly']
STAMuons1LegBarrelOnly.inputCollectionTypes = ['outer tracks']
STAMuons1LegBarrelOnly.fillIsolation = False
GLBMuons1LegBarrelOnly.inputCollectionLabels = ['globalCosmicMuons1LegBarrelOnly']
GLBMuons1LegBarrelOnly.inputCollectionTypes = ['links']
GLBMuons1LegBarrelOnly.fillIsolation = False
muons1LegBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuons1LegBarrelOnly', 'cosmicMuons1LegBarrelOnly']
muons1LegBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muons1LegBarrelOnly.fillIsolation = False
lhcMuonSeedBarrelOnly.EnableCSCMeasurement = False
lhcStandAloneMuonsBarrelOnly.STATrajBuilderParameters.RefitterParameters.EnableCSCMeasurement = False
lhcStandAloneMuonsBarrelOnly.STATrajBuilderParameters.BWFilterParameters.EnableCSCMeasurement = False
lhcStandAloneMuonsBarrelOnly.InputObjects = 'lhcMuonSeedBarrelOnly'
lhcMuonSeedEndCapsOnly.EnableDTMeasurement = False
lhcStandAloneMuonsEndCapsOnly.STATrajBuilderParameters.RefitterParameters.EnableDTMeasurement = False
lhcStandAloneMuonsEndCapsOnly.STATrajBuilderParameters.BWFilterParameters.EnableDTMeasurement = False
lhcStandAloneMuonsEndCapsOnly.InputObjects = 'lhcMuonSeedEndCapsOnly'
CosmicMuonSeedNoDriftBarrelOnly.EnableCSCMeasurement = False
CosmicMuonSeedNoDriftBarrelOnly.DTRecSegmentLabel = 'dt4DSegmentsNoDrift'
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.EnableCSCMeasurement = False
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.BuildTraversingMuon = True
cosmicMuonsNoDriftBarrelOnly.MuonSeedCollectionLabel = 'CosmicMuonSeedNoDriftBarrelOnly'
cosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.DTRecSegmentLabel = 'dt4DSegmentsNoDrift'
globalCosmicMuonsNoDriftBarrelOnly.TrajectoryBuilderParameters.TkTrackCollectionLabel = 'ctfWithMaterialTracksP5'
globalCosmicMuonsNoDriftBarrelOnly.MuonCollectionLabel = 'cosmicMuonsNoDriftBarrelOnly'
STAMuonsNoDriftBarrelOnly.inputCollectionLabels = ['cosmicMuonsNoDriftBarrelOnly']
STAMuonsNoDriftBarrelOnly.inputCollectionTypes = ['outer tracks']
STAMuonsNoDriftBarrelOnly.fillIsolation = False
GLBMuonsNoDriftBarrelOnly.inputCollectionLabels = ['globalCosmicMuonsNoDriftBarrelOnly']
GLBMuonsNoDriftBarrelOnly.inputCollectionTypes = ['links']
GLBMuonsNoDriftBarrelOnly.fillIsolation = False
muonsNoDriftBarrelOnly.inputCollectionLabels = ['ctfWithMaterialTracksP5', 'globalCosmicMuonsNoDriftBarrelOnly', 'cosmicMuonsNoDriftBarrelOnly']
muonsNoDriftBarrelOnly.inputCollectionTypes = ['inner tracks', 'links', 'outer tracks']
muonsNoDriftBarrelOnly.fillIsolation = False



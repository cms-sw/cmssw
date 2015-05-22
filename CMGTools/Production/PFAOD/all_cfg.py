import sys
import os

sys.path.append( os.getcwd() )
from base import *

import __main__

tag = 'all'

output = '_'.join( [sampleStr, tag] )
output += '.root'

print 'output file', output 

process.out.fileName = output 



process.out.outputCommands.extend(
    [
    'drop *Castor*_*_*_*',
    'keep recoCaloClusters_hybridSuperClusters_hybridBarrelBasicClusters_*',
    'keep recoCaloClusters_multi5x5BasicClusters_multi5x5EndcapBasicClusters_*',
    'keep recoCaloClusters_hybridSuperClusters_uncleanOnlyHybridBarrelBasicClusters_*',
    'keep recoSuperClusters_hybridSuperClusters_uncleanOnlyHybridSuperClusters_*',
    'keep recoCaloClusters_pfPhotonTranslator_pfphot_*',
    'keep recoTracks_tevMuons_default_*',
    'keep recoTracks_tevMuons_dyt_*',
    'keep recoTracks_tevMuons_firstHit_*',
    'keep recoTracks_tevMuons_picky_*',
    'keep recoTrackExtras_tevMuons_default_*',
    'keep recoTrackExtras_tevMuons_dyt_*',
    'keep recoTrackExtras_tevMuons_firstHit_*',
    'keep recoTrackExtras_tevMuons_picky_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_default_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_dyt_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_firstHit_*',
    'keep recoTracksToOnerecoTracksAssociation_tevMuons_picky_*',
    'keep *_ak7CaloJets_*_*',
    'keep recoPhotonCores_photonCore__*',
    'keep recoPhotons_pfPhotonTranslator_pfphot_*',
    'keep recoPhotons_photons__*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLoose_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDLooseEM_*',
    'keep booledmValueMap_PhotonIDProd_PhotonCutBasedIDTight_*',
    'keep recoPreshowerClusters_pfPhotonTranslator_pfphot_*',
    'keep recoSuperClusters_pfPhotonTranslator_pfphot_*',
    ])


# Author     : Andreas Mussgiller
# Date       : July 1st, 2010
# last update: $Date: 2010/07/06 11:48:22 $ by $Author: mussgill $

import FWCore.ParameterSet.Config as cms

#_________________________________HLT bits___________________________________________
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsInCollisionsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlCosmics',
    throw = False # tolerate triggers not available
    )

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlCosmicsInCollisionsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

#________________________________Track selection____________________________________
# AlCaReco for track based alignment using Cosmic muons reconstructed by Combinatorial Track Finder
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlCosmicsInCollisions = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'regionalCosmicTracks',
    filter = True,
    applyBasicCuts = True,

    ptMin = 0., ##10
    ptMax = 99999.,
    pMin = 4., ##10
    pMax = 99999.,
    etaMin = -99., ##-2.4 keep also what is going through...
    etaMax = 99., ## 2.4 ...both TEC with flat slope

    nHitMin = 7,
    nHitMin2D = 2,
    chi2nMax = 999999.,
    
    applyMultiplicityFilter = False,
    applyNHighestPt = True, ## select only highest pT track
    nHighestPt = 1
    )

#________________________________Sequences____________________________________
seqALCARECOTkAlCosmicsInCollisions = cms.Sequence(ALCARECOTkAlCosmicsInCollisionsHLT+ALCARECOTkAlCosmicsInCollisionsDCSFilter+ALCARECOTkAlCosmicsInCollisions)

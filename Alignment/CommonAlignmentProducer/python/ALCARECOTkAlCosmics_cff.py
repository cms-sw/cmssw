# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2010/03/17 18:17:34 $ by $Author: mussgill $

import FWCore.ParameterSet.Config as cms

# DCS partitions
# "EBp","EBm","EEp","EEm","HBHEa","HBHEb","HBHEc","HF","HO","RPC"
# "DT0","DTp","DTm","CSCp","CSCm","CASTOR","TIBTID","TOB","TECp","TECm"
# "BPIX","FPIX","ESp","ESm"
import DPGAnalysis.Skims.skim_detstatus_cfi
ALCARECOTkAlCosmicsDCSFilter = DPGAnalysis.Skims.skim_detstatus_cfi.dcsstatus.clone(
    DetectorType = cms.vstring('TIBTID','TOB','TECp','TECm','BPIX','FPIX'),
    ApplyFilter  = cms.bool(True),
    AndOr        = cms.bool(True),
    DebugOn      = cms.untracked.bool(False)
)

#________________________________Track selection____________________________________
# AlCaReco for track based alignment using Cosmic muons reconstructed by Combinatorial Track Finder
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlCosmicsCTF = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ctfWithMaterialTracksP5',
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

# AlCaReco for track based alignment using Cosmic muons reconstructed by Cosmic Track Finder
# (same cuts)
ALCARECOTkAlCosmicsCosmicTF = ALCARECOTkAlCosmicsCTF.clone(
    src = 'cosmictrackfinderP5' ## different for CTF
    )

#________________________________Sequences____________________________________
# Work around since only one filter can be used:
# Run the RS and CosmicTF filters before CTF filter, but ignore their results.
# So whenever there is CTF, we keep also the others...
#sequence seqALCARECOTkAlCosmicsCTF      = { 
#    -ALCARECOTkAlCosmicsRS & -ALCARECOTkAlCosmicsCosmicTF & ALCARECOTkAlCosmicsCTF 
#}
#sequence seqALCARECOTkAlCosmicsCTF_orig = { ALCARECOTkAlCosmicsCTF }
# Benedikt tells me it works:
seqALCARECOTkAlCosmicsCTF = cms.Sequence(ALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTF = cms.Sequence(ALCARECOTkAlCosmicsCosmicTF)

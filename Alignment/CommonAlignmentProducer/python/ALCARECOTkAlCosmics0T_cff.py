import FWCore.ParameterSet.Config as cms

# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/06/19 18:25:55 $ by $Author: flucke $
#________________________________Track selection____________________________________
# AlCaReco for track based alignment using Cosmic muons reconstructed by Combinatorial Track Finder
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
ALCARECOTkAlCosmicsCTF0T = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone(
    src = 'ctfWithMaterialTracksP5',
    filter = True,
    applyBasicCuts = True,
    ptMin = 0., ##10
    ptMax = 99999.,
    etaMin = -99., ##-2.4 keep also what is going through...
    etaMax = 99., ## 2.4 ...both TEC with flat slope
    nHitMin = 7,
    nHitMin2D = 2,
    chi2nMax = 999999.,
    applyNHighestPt = False, ## no pT measurement -> sort meaningless
    nHighestPt = 1,
    applyMultiplicityFilter = False
    )

# AlCaReco for track based alignment using Cosmic muons reconstructed by Cosmic Track Finder
# (same cuts)
ALCARECOTkAlCosmicsCosmicTF0T = ALCARECOTkAlCosmicsCTF0T.clone(
    src = 'cosmictrackfinderP5' ## different for CTF
    )

# AlCaReco for track based alignment using Cosmic muons reconstructed by Road Search Track Finder
# (same cuts)
ALCARECOTkAlCosmicsRS0T = ALCARECOTkAlCosmicsCTF0T.clone(
    src = 'rsWithMaterialTracksP5'
    )

#________________________________Sequences____________________________________
seqALCARECOTkAlCosmicsCTF0T = cms.Sequence(ALCARECOTkAlCosmicsCTF0T)
seqALCARECOTkAlCosmicsCosmicTF0T = cms.Sequence(ALCARECOTkAlCosmicsCosmicTF0T)
seqALCARECOTkAlCosmicsRS0T = cms.Sequence(ALCARECOTkAlCosmicsRS0T)

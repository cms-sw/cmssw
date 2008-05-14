import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/05/09 15:18:17 $ by $Author: emiglior $
#_________________________________HLT bits___________________________________________
ALCARECOTkAlCosmicsCTF0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsCosmicTF0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsRS0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
#________________________________Track selection____________________________________
# AlCaReco for track based alignment using Cosmic muons reconstructed by Combinatorial Track Finder
ALCARECOTkAlCosmicsCTF0T = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
# AlCaReco for track based alignment using Cosmic muons reconstructed by Cosmic Track Finder
ALCARECOTkAlCosmicsCosmicTF0T = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
# AlCaReco for track based alignment using Cosmic muons reconstructed by Road Search Track Finder
ALCARECOTkAlCosmicsRS0T = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
#________________________________Sequences____________________________________
seqALCARECOTkAlCosmicsCTF0THLT = cms.Sequence(ALCARECOTkAlCosmicsCTF0THLT+ALCARECOTkAlCosmicsCTF0T)
seqALCARECOTkAlCosmicsCosmicTF0THLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTF0THLT+ALCARECOTkAlCosmicsCosmicTF0T)
seqALCARECOTkAlCosmicsRS0THLT = cms.Sequence(ALCARECOTkAlCosmicsRS0THLT+ALCARECOTkAlCosmicsRS0T)
ALCARECOTkAlCosmicsCTF0THLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCTF0THLT.HLTPaths = ['CandHLTTrackerCosmicsCTF']
ALCARECOTkAlCosmicsCosmicTF0THLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCosmicTF0THLT.HLTPaths = ['CandHLTTrackerCosmicsCoTF']
ALCARECOTkAlCosmicsRS0THLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsRS0THLT.HLTPaths = ['CandHLTTrackerCosmicsRS']
ALCARECOTkAlCosmicsCTF0T.src = 'ctfWithMaterialTracksP5'
ALCARECOTkAlCosmicsCTF0T.filter = True
ALCARECOTkAlCosmicsCTF0T.applyBasicCuts = True
ALCARECOTkAlCosmicsCTF0T.ptMin = 0. ##10

ALCARECOTkAlCosmicsCTF0T.ptMax = 99999.
ALCARECOTkAlCosmicsCTF0T.etaMin = -99. ##-2.4 keep also what is going through...

ALCARECOTkAlCosmicsCTF0T.etaMax = 99. ## 2.4 ...both TEC with flat slope

ALCARECOTkAlCosmicsCTF0T.nHitMin = 5 ##8

ALCARECOTkAlCosmicsCTF0T.chi2nMax = 999999.
ALCARECOTkAlCosmicsCTF0T.applyNHighestPt = False ## no pT measurement -> sort meaningless

ALCARECOTkAlCosmicsCTF0T.nHighestPt = 1
ALCARECOTkAlCosmicsCTF0T.applyMultiplicityFilter = False
ALCARECOTkAlCosmicsCosmicTF0T.src = 'cosmictrackfinderP5' ## different for CTF

ALCARECOTkAlCosmicsCosmicTF0T.filter = True
ALCARECOTkAlCosmicsCosmicTF0T.applyBasicCuts = True
ALCARECOTkAlCosmicsCosmicTF0T.ptMin = 0.
ALCARECOTkAlCosmicsCosmicTF0T.ptMax = 99999.
ALCARECOTkAlCosmicsCosmicTF0T.etaMin = -99.
ALCARECOTkAlCosmicsCosmicTF0T.etaMax = 99.
ALCARECOTkAlCosmicsCosmicTF0T.nHitMin = 7 ## more hits than CTF: 2D are counted twice

ALCARECOTkAlCosmicsCosmicTF0T.chi2nMax = 999999.
ALCARECOTkAlCosmicsCosmicTF0T.applyNHighestPt = False ## no pT measurement -> sort meaningless

ALCARECOTkAlCosmicsCosmicTF0T.nHighestPt = 1
ALCARECOTkAlCosmicsCosmicTF0T.applyMultiplicityFilter = False
ALCARECOTkAlCosmicsRS0T.src = 'rsWithMaterialTracksP5'
ALCARECOTkAlCosmicsRS0T.filter = True
ALCARECOTkAlCosmicsRS0T.applyBasicCuts = True
ALCARECOTkAlCosmicsRS0T.ptMin = 0. ##10

ALCARECOTkAlCosmicsRS0T.ptMax = 99999.
ALCARECOTkAlCosmicsRS0T.etaMin = -99. ##-2.4 keep also what is going through...

ALCARECOTkAlCosmicsRS0T.etaMax = 99. ## 2.4 ...both TEC with flat slope

ALCARECOTkAlCosmicsRS0T.nHitMin = 5 ##8

ALCARECOTkAlCosmicsRS0T.chi2nMax = 999999.
ALCARECOTkAlCosmicsRS0T.applyNHighestPt = False ## no pT measurement -> sort meaningless

ALCARECOTkAlCosmicsRS0T.nHighestPt = 1
ALCARECOTkAlCosmicsRS0T.applyMultiplicityFilter = False


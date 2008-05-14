import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/05/09 15:19:27 $ by $Author: emiglior $
#_________________________________HLT bits___________________________________________
ALCARECOTkAlCosmicsCTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsCosmicTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsRSHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
#________________________________Track selection____________________________________
# AlCaReco for track based alignment using Cosmic muons reconstructed by Combinatorial Track Finder
ALCARECOTkAlCosmicsCTF = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
# AlCaReco for track based alignment using Cosmic muons reconstructed by Cosmic Track Finder
ALCARECOTkAlCosmicsCosmicTF = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
import Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi
# AlCaReco for track based alignment using Cosmic muons reconstructed by Road Search Track Finder
ALCARECOTkAlCosmicsRS = Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi.AlignmentTrackSelector.clone()
#________________________________Sequences____________________________________
seqALCARECOTkAlCosmicsCTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCTFHLT+ALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTFHLT+ALCARECOTkAlCosmicsCosmicTF)
seqALCARECOTkAlCosmicsRSHLT = cms.Sequence(ALCARECOTkAlCosmicsRSHLT+ALCARECOTkAlCosmicsRS)
ALCARECOTkAlCosmicsCTFHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCTFHLT.HLTPaths = ['CandHLTTrackerCosmicsCTF']
ALCARECOTkAlCosmicsCosmicTFHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCosmicTFHLT.HLTPaths = ['CandHLTTrackerCosmicsCoTF']
ALCARECOTkAlCosmicsRSHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsRSHLT.HLTPaths = ['CandHLTTrackerCosmicsRS']
ALCARECOTkAlCosmicsCTF.src = 'ctfWithMaterialTracksP5'
ALCARECOTkAlCosmicsCTF.filter = True
ALCARECOTkAlCosmicsCTF.applyBasicCuts = True
ALCARECOTkAlCosmicsCTF.ptMin = 4. ##10

ALCARECOTkAlCosmicsCTF.ptMax = 99999.
ALCARECOTkAlCosmicsCTF.etaMin = -99. ##-2.4 keep also what is going through...

ALCARECOTkAlCosmicsCTF.etaMax = 99. ## 2.4 ...both TEC with flat slope

ALCARECOTkAlCosmicsCTF.nHitMin = 5 ##8

ALCARECOTkAlCosmicsCTF.chi2nMax = 999999.
ALCARECOTkAlCosmicsCTF.applyNHighestPt = True ## select only highest pT track

ALCARECOTkAlCosmicsCTF.nHighestPt = 1
ALCARECOTkAlCosmicsCTF.applyMultiplicityFilter = False
ALCARECOTkAlCosmicsCosmicTF.src = 'cosmictrackfinderP5' ## different for CTF

ALCARECOTkAlCosmicsCosmicTF.filter = True
ALCARECOTkAlCosmicsCosmicTF.applyBasicCuts = True
ALCARECOTkAlCosmicsCosmicTF.ptMin = 4.
ALCARECOTkAlCosmicsCosmicTF.ptMax = 99999.
ALCARECOTkAlCosmicsCosmicTF.etaMin = -99.
ALCARECOTkAlCosmicsCosmicTF.etaMax = 99.
ALCARECOTkAlCosmicsCosmicTF.nHitMin = 7 ## more hits than CTF: 2D are counted twice

ALCARECOTkAlCosmicsCosmicTF.chi2nMax = 999999.
ALCARECOTkAlCosmicsCosmicTF.applyNHighestPt = True ## select only highest pT track

ALCARECOTkAlCosmicsCosmicTF.nHighestPt = 1
ALCARECOTkAlCosmicsCosmicTF.applyMultiplicityFilter = False
ALCARECOTkAlCosmicsRS.src = 'rsWithMaterialTracksP5'
ALCARECOTkAlCosmicsRS.filter = True
ALCARECOTkAlCosmicsRS.applyBasicCuts = True
ALCARECOTkAlCosmicsRS.ptMin = 4. ##10

ALCARECOTkAlCosmicsRS.ptMax = 99999.
ALCARECOTkAlCosmicsRS.etaMin = -99. ##-2.4 keep also what is going through...

ALCARECOTkAlCosmicsRS.etaMax = 99. ## 2.4 ...both TEC with flat slope

ALCARECOTkAlCosmicsRS.nHitMin = 5 ##8

ALCARECOTkAlCosmicsRS.chi2nMax = 999999.
ALCARECOTkAlCosmicsRS.applyNHighestPt = True ## select only highest pT track

ALCARECOTkAlCosmicsRS.nHighestPt = 1
ALCARECOTkAlCosmicsRS.applyMultiplicityFilter = False


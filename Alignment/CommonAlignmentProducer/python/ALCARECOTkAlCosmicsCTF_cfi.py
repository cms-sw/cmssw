import FWCore.ParameterSet.Config as cms

import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/03/14 20:45:06 $ by $Author: flucke $
# AlCaReco for track based alignment using Cosmic muons reconstructed by 
# Combinatorial Track Finder
ALCARECOTkAlCosmicsCTF = copy.deepcopy(AlignmentTrackSelector)
ALCARECOTkAlCosmicsCTF.src = 'ctfWithMaterialTracksP5'
ALCARECOTkAlCosmicsCTF.filter = True
ALCARECOTkAlCosmicsCTF.applyBasicCuts = True
ALCARECOTkAlCosmicsCTF.ptMin = 5. ##10

ALCARECOTkAlCosmicsCTF.ptMax = 99999.
ALCARECOTkAlCosmicsCTF.etaMin = -99. ##-2.4 keep also what is going through...

ALCARECOTkAlCosmicsCTF.etaMax = 99. ## 2.4 ...both TEC with flat slope

ALCARECOTkAlCosmicsCTF.nHitMin = 10 ##8

ALCARECOTkAlCosmicsCTF.chi2nMax = 999999.
ALCARECOTkAlCosmicsCTF.applyNHighestPt = True
ALCARECOTkAlCosmicsCTF.nHighestPt = 1
ALCARECOTkAlCosmicsCTF.applyMultiplicityFilter = False


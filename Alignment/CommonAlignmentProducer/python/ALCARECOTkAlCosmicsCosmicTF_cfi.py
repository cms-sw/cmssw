import FWCore.ParameterSet.Config as cms

import copy
from Alignment.CommonAlignmentProducer.AlignmentTrackSelector_cfi import *
# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2008/03/14 20:45:06 $ by $Author: flucke $
# AlCaReco for track based alignment using Cosmic muons from Cosmic Track Finder
#module ALCARECOTkAlCosmicsCosmicTF = ALCARECOTkAlCosmicsCTF from "Alignment/CommonAlignmentProducer/data/ALCARECOTkAlCosmicsCTF.cfi"
#
#replace ALCARECOTkAlCosmicsCosmicTF.src = cosmicTrackFinderP5
#replace ALCARECOTkAlCosmicsCosmicTF.nHitMin = 14 ## more hits than CTF: 2D are counted twice
 # more hits: 2D are counted twice
#
# Unfortunately the above does not work like that, but gives an exception, as reported also 
# in https://hypernews.cern.ch/HyperNews/CMS/get/edmFramework/949.html :
#%MSG-s CMSException:  19-Jul-2007 10:54:09 CEST pre-events
#cms::Exception caught in cmsRun
#---- Configuration BEGIN
#Could not find node ALCARECOTkAlCosmicsCTF in 
#file Alignment/CommonAlignmentProducer/data/ALCARECOTkAlCosmicsCTF.cfi
#---- Configuration END
ALCARECOTkAlCosmicsCosmicTF = copy.deepcopy(AlignmentTrackSelector)
ALCARECOTkAlCosmicsCosmicTF.src = 'cosmictrackfinderP5' ## different for CTF

ALCARECOTkAlCosmicsCosmicTF.filter = True
ALCARECOTkAlCosmicsCosmicTF.applyBasicCuts = True
ALCARECOTkAlCosmicsCosmicTF.ptMin = 5.
ALCARECOTkAlCosmicsCosmicTF.ptMax = 99999.
ALCARECOTkAlCosmicsCosmicTF.etaMin = -99.
ALCARECOTkAlCosmicsCosmicTF.etaMax = 99.
ALCARECOTkAlCosmicsCosmicTF.nHitMin = 14 ## more hits than CTF: 2D are counted twice

ALCARECOTkAlCosmicsCosmicTF.chi2nMax = 999999.
ALCARECOTkAlCosmicsCosmicTF.applyNHighestPt = True
ALCARECOTkAlCosmicsCosmicTF.nHighestPt = 1
ALCARECOTkAlCosmicsCosmicTF.applyMultiplicityFilter = False


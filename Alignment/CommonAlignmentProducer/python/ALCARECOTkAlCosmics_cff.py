import FWCore.ParameterSet.Config as cms

# Author     : Gero Flucke
# Date       :   July 19th, 2007
# last update: $Date: 2007/07/19 12:58:49 $ by $Author: flucke $
# AlCaReco for track based alignment using Cosmic muon events
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsCTF_cfi import *
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmicsCosmicTF_cfi import *
seqALCARECOTkAlCosmicsCTF = cms.Sequence(ALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTF = cms.Sequence(ALCARECOTkAlCosmicsCosmicTF)


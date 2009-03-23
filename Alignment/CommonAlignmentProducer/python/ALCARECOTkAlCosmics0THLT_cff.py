# last update: $Date: 2009/03/04 11:20:13 $ by $Author: flucke $
import FWCore.ParameterSet.Config as cms

#_________________________________HLT bits___________________________________________
import HLTrigger.HLTfilters.hltHighLevel_cfi



ALCARECOTkAlCosmics0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlCosmics0T',
    throw = False # tolerate triggers not available
    )

#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *

#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTF0THLT = cms.Sequence(ALCARECOTkAlCosmics0THLT+seqALCARECOTkAlCosmicsCTF0T)
seqALCARECOTkAlCosmicsCosmicTF0THLT = cms.Sequence(ALCARECOTkAlCosmics0THLT+seqALCARECOTkAlCosmicsCosmicTF0T)
seqALCARECOTkAlCosmicsRS0THLT = cms.Sequence(ALCARECOTkAlCosmics0THLT+seqALCARECOTkAlCosmicsRS0T)


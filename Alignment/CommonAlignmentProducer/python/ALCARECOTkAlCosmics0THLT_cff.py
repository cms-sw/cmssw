# last update: $Date: 2008/07/21 12:45:23 $ by $Author: flucke $
import FWCore.ParameterSet.Config as cms

#_________________________________HLT bits___________________________________________
import HLTrigger.HLTfilters.hltHighLevel_cfi

ALCARECOTkAlCosmicsCTF0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOTkAlCosmicsCTF0THLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlCosmicsCTF0THLT.HLTPaths = ['HLT_TrackerCosmics_CTF']

ALCARECOTkAlCosmicsCosmicTF0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOTkAlCosmicsCosmicTF0THLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlCosmicsCosmicTF0THLT.HLTPaths = ['HLT_TrackerCosmics_CoTF']

ALCARECOTkAlCosmicsRS0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOTkAlCosmicsRS0THLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlCosmicsRS0THLT.HLTPaths = ['HLT_TrackerCosmics_RS']


#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *

#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTF0THLT = cms.Sequence(ALCARECOTkAlCosmicsCTF0THLT+seqALCARECOTkAlCosmicsCTF0T)
seqALCARECOTkAlCosmicsCosmicTF0THLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTF0THLT+seqALCARECOTkAlCosmicsCosmicTF0T)
seqALCARECOTkAlCosmicsRS0THLT = cms.Sequence(ALCARECOTkAlCosmicsRS0THLT+seqALCARECOTkAlCosmicsRS0T)


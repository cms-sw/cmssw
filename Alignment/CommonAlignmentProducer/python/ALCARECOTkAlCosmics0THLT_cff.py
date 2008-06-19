import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# last update: $Date: 2008/06/19 18:06:49 $ by $Author: flucke $
#_________________________________HLT bits___________________________________________
ALCARECOTkAlCosmicsCTF0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsCosmicTF0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsRS0THLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics0T_cff import *
#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTF0THLT = cms.Sequence(ALCARECOTkAlCosmicsCTF0THLT+seqALCARECOTkAlCosmicsCTF0T)
seqALCARECOTkAlCosmicsCosmicTF0THLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTF0THLT+seqALCARECOTkAlCosmicsCosmicTF0T)
seqALCARECOTkAlCosmicsRS0THLT = cms.Sequence(ALCARECOTkAlCosmicsRS0THLT+seqALCARECOTkAlCosmicsRS0T)
ALCARECOTkAlCosmicsCTF0THLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCTF0THLT.HLTPaths = ['CandHLTTrackerCosmicsCTF']
ALCARECOTkAlCosmicsCosmicTF0THLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCosmicTF0THLT.HLTPaths = ['CandHLTTrackerCosmicsCoTF']
ALCARECOTkAlCosmicsRS0THLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsRS0THLT.HLTPaths = ['CandHLTTrackerCosmicsRS']


import FWCore.ParameterSet.Config as cms

import HLTrigger.HLTfilters.hltHighLevel_cfi
# last update: $Date: 2008/06/19 18:25:55 $ by $Author: flucke $
#_________________________________HLT bits___________________________________________
ALCARECOTkAlCosmicsCTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsCosmicTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsRSHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *
#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCTFHLT+seqALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTFHLT+seqALCARECOTkAlCosmicsCosmicTF)
seqALCARECOTkAlCosmicsRSHLT = cms.Sequence(ALCARECOTkAlCosmicsRSHLT+seqALCARECOTkAlCosmicsRS)
ALCARECOTkAlCosmicsCTFHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCTFHLT.HLTPaths = ['HLT_TrackerCosmics_CTF']
ALCARECOTkAlCosmicsCosmicTFHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsCosmicTFHLT.HLTPaths = ['HLT_TrackerCosmics_CoTF']
ALCARECOTkAlCosmicsRSHLT.andOr = True ## choose logical OR between Triggerbits

ALCARECOTkAlCosmicsRSHLT.HLTPaths = ['HLT_TrackerCosmics_RS']


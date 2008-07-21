# last update: $Date: 2008/07/21 12:45:23 $ by $Author: flucke $
import FWCore.ParameterSet.Config as cms

#_________________________________HLT bits___________________________________________
import HLTrigger.HLTfilters.hltHighLevel_cfi

ALCARECOTkAlCosmicsCTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOTkAlCosmicsCTFHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlCosmicsCTFHLT.HLTPaths = ['HLT_TrackerCosmics_CTF']

ALCARECOTkAlCosmicsCosmicTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOTkAlCosmicsCosmicTFHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlCosmicsCosmicTFHLT.HLTPaths = ['HLT_TrackerCosmics_CoTF']

ALCARECOTkAlCosmicsRSHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone()
ALCARECOTkAlCosmicsRSHLT.andOr = True ## choose logical OR between Triggerbits
ALCARECOTkAlCosmicsRSHLT.HLTPaths = ['HLT_TrackerCosmics_RS']

#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *

#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCTFHLT+seqALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTFHLT+seqALCARECOTkAlCosmicsCosmicTF)
seqALCARECOTkAlCosmicsRSHLT = cms.Sequence(ALCARECOTkAlCosmicsRSHLT+seqALCARECOTkAlCosmicsRS)


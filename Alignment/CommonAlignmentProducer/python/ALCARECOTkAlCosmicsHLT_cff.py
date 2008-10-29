# last update: $Date: 2008/07/21 12:57:08 $ by $Author: flucke $
import FWCore.ParameterSet.Config as cms

#_________________________________HLT bits___________________________________________
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsCTFHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    HLTPaths = ['HLT_TrackerCosmics_CTF'],
    throw = False # tolerate triggers stated above, but not available
    )
    
# same as above, but other path
ALCARECOTkAlCosmicsCosmicTFHLT = ALCARECOTkAlCosmicsCTFHLT.clone(
    HLTPaths = ['HLT_TrackerCosmics_CoTF']
    )

# same as above, but other path
ALCARECOTkAlCosmicsRSHLT = ALCARECOTkAlCosmicsCTFHLT.clone(
    HLTPaths = ['HLT_TrackerCosmics_RS']
    )

#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *

#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCTFHLT+seqALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTFHLT = cms.Sequence(ALCARECOTkAlCosmicsCosmicTFHLT+seqALCARECOTkAlCosmicsCosmicTF)
seqALCARECOTkAlCosmicsRSHLT = cms.Sequence(ALCARECOTkAlCosmicsRSHLT+seqALCARECOTkAlCosmicsRS)


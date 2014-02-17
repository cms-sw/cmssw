# last update: $Date: 2011/07/01 07:01:20 $ by $Author: mussgill $
import FWCore.ParameterSet.Config as cms

#_________________________________HLT bits___________________________________________
import HLTrigger.HLTfilters.hltHighLevel_cfi
ALCARECOTkAlCosmicsHLT = HLTrigger.HLTfilters.hltHighLevel_cfi.hltHighLevel.clone(
    andOr = True, ## choose logical OR between Triggerbits
    eventSetupPathsKey = 'TkAlCosmics',
    throw = False # tolerate triggers not available
    )
    
#________________________________Track selection____________________________________
# take from stream that is HLT-less
from Alignment.CommonAlignmentProducer.ALCARECOTkAlCosmics_cff import *

#________________________________Sequences____________________________________
# simply add HLT in front of HLT-less paths
seqALCARECOTkAlCosmicsCTFHLT = cms.Sequence(ALCARECOTkAlCosmicsHLT+seqALCARECOTkAlCosmicsCTF)
seqALCARECOTkAlCosmicsCosmicTFHLT = cms.Sequence(ALCARECOTkAlCosmicsHLT+seqALCARECOTkAlCosmicsCosmicTF)
seqALCARECOTkAlCosmicsRegionalHLT = cms.Sequence(ALCARECOTkAlCosmicsHLT+seqALCARECOTkAlCosmicsRegional)


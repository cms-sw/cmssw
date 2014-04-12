import FWCore.ParameterSet.Config as cms
 
# import the whole HLT menu
from HLTrigger.HLTanalyzers.HLT_FULL_cff import *

# create the HI jet HLT reco path
DoHLTHIJets = cms.Path(HLTBeginSequence + 
    HLTDoHIJetRecoSequence)

# create the muon HLT reco path
DoHLTHIMuon = cms.Path(HLTBeginSequence + 
        HLTEndSequence)

#### For the future of Muon HLT in the case of including L3 sequence
#from CmsHi.HiMuonAlgos.HiL3MuonCandidateProducer_cfi import *
#DoHLTHIMuon = cms.Path(HLTBeginSequence + HLTL2muonrecoSequence + HLTDoLocalPixelSequence + HLTHIRecopixelvertexingSequence + HLTDoLocalStripSequence + hltIMML3Filter + hltHIL3MuonCandidate + HLTEndSequence)

# create the Egamma HLT reco paths
DoHLTHIPhoton = cms.Path(
    HLTBeginSequence +
    HLTDoCaloSequence +
# FIXME
#    HLTDoHIEcalClusSequence +
    HLTDoHIEcalClusWithCleaningSequence +
#
    HLTEndSequence )

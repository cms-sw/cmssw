import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Reconstruction_cff import *

from L1Trigger.L1TNtuples.l1EventTree_cfi import *
from L1Trigger.L1TNtuples.l1RecoTree_cfi import *
from L1Trigger.L1TNtuples.l1JetRecoTree_cfi import *
from L1Trigger.L1TNtuples.l1MetFilterRecoTree_cfi import *
from L1Trigger.L1TNtuples.l1ElectronRecoTree_cfi import *
from L1Trigger.L1TNtuples.l1PhotonRecoTree_cfi import *
from L1Trigger.L1TNtuples.l1TauRecoTree_cfi import *
#from L1Trigger.L1TNtuples.l1TauRecoTree_2015_cfi import *
from L1Trigger.L1TNtuples.l1MuonRecoTree_cfi import *

L1NtupleAOD = cms.Sequence(
  l1EventTree
  +l1RecoTree
  +l1JetRecoTree
  +l1MetFilterRecoTree
  +l1ElectronRecoTree
  +l1PhotonRecoTree
  +l1TauRecoTree
  +l1MuonRecoTree
)


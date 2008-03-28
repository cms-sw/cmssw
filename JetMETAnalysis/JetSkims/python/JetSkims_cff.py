import FWCore.ParameterSet.Config as cms

from JetMETAnalysis.JetSkims.photonjets_Sequences_cff import *
from JetMETAnalysis.JetSkims.onejet_Sequences_cff import *
jetSkims = cms.Sequence(photonjetsHLTFilter+cms.SequencePlaceholder("onejetsHLTFilter")+cms.SequencePlaceholder("dijetbalanceHLTFilter"))


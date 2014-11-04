import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Core.genFilterEfficiencyProducer_cfi import *
from GeneratorInterface.Core.genXSecAnalyzer_cfi import *

genFilterSummary = cms.Sequence(genFilterEfficiencyProducer*genXSecAnalyzer)

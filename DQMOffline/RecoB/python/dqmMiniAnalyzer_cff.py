import FWCore.ParameterSet.Config as cms

from DQMOffline.RecoB.DeepFlavour import *
from DQMOffline.RecoB.DeepCSV import *


bTagAnalyzer_mini = cms.Sequence(
    DeepFlavourAnalyzer
    #* DeepCSVAnalyzer
)

bTagHarvester_mini = cms.Sequence(
    DeepFlavourHarvester
    #* DeepCSVHarvester
)

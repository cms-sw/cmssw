import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi import *

towerMaker = calotowermaker.clone()

towerMaker.UseHO = False

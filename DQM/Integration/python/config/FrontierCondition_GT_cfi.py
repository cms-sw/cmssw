import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.AlCa.autoCond import autoCond
GlobalTag.globaltag = autoCond['run3_hlt']

import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Digi_cff import *

# switch off HCAL ZS in digi
simHcalDigis.markAndPass = True
simHcalDigis.HBlevel = -999
simHcalDigis.HElevel = -999
simHcalDigis.HOlevel = -999
simHcalDigis.HFlevel = -999
simHcalDigis.useConfigZSvalues = 1

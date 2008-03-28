import FWCore.ParameterSet.Config as cms

import copy
from EventFilter.EcalRawToDigi.EcalRawToRecHitProducer_cfi import *
ecalRegionalEgammaRecHit = copy.deepcopy(EcalRawToRecHitProducer)
import copy
from EventFilter.EcalRawToDigi.EcalRawToRecHitProducer_cfi import *
ecalRegionalMuonsRecHit = copy.deepcopy(EcalRawToRecHitProducer)
import copy
from EventFilter.EcalRawToDigi.EcalRawToRecHitProducer_cfi import *
ecalRegionalJetsRecHit = copy.deepcopy(EcalRawToRecHitProducer)
import copy
from EventFilter.EcalRawToDigi.EcalRawToRecHitProducer_cfi import *
# a second on to se if things are unpcaked twice.
#module ecalRegionalJetsRecHitBIS = EcalRawToRecHitProducer from "EventFilter/EcalRawToDigi/data/EcalRawToRecHitProducer.cfi" 
#replace ecalRegionalJetsRecHitBIS.sourceTag=ecalRegionalJetsFEDs
ecalRegionalTausRecHit = copy.deepcopy(EcalRawToRecHitProducer)
import copy
from EventFilter.EcalRawToDigi.EcalRawToRecHitProducer_cfi import *
ecalRecHitAll = copy.deepcopy(EcalRawToRecHitProducer)
ecalRegionalEgammaRecHit.sourceTag = 'ecalRegionalEgammaFEDs'
ecalRegionalMuonsRecHit.sourceTag = 'ecalRegionalMuonsFEDs'
ecalRegionalJetsRecHit.sourceTag = 'ecalRegionalJetsFEDs'
ecalRegionalTausRecHit.sourceTag = 'ecalRegionalTausFEDs'
ecalRecHitAll.sourceTag = 'ecalRegionalRestFEDs'


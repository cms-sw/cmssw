import FWCore.ParameterSet.Config as cms

import copy
from HLTrigger.HLTfilters.hltLevel1GTSeed_cfi import *
#/**
#*  Definition of AlcaReco online selection for Ecal PhiSymmetry Calibration
#*
#*  \author Stefano Argiro
#*  \Id $Id: AlcastreamEcalPhiSym.cff,v 1.6 2008/01/14 10:09:24 argiro Exp $
#*/
#event selection
l1sEcalPhiSym = copy.deepcopy(hltLevel1GTSeed)
import copy
from HLTrigger.HLTcore.hltPrescaler_cfi import *
#prescaler
ecalPhiSymPresc = copy.deepcopy(hltPrescaler)
#unpacker configuration
#replace ecalDigis.DoRegional = false
#create the alcareco stream with selected rechits
from HLTrigger.special.HLTEcalPhiSymFilter_cfi import *
doEcalPhiSymSequence = cms.Sequence(l1sEcalPhiSym+ecalPhiSymPresc+cms.SequencePlaceholder("doLocalEcal_nopreshower")+alCaPhiSymStream)
alcaEcalPhiSymSequence = cms.Sequence(cms.SequencePlaceholder("hltBegin")+doEcalPhiSymSequence)
l1sEcalPhiSym.L1SeedsLogicalExpression = 'L1_ZeroBias'
ecalPhiSymPresc.prescaleFactor = 1


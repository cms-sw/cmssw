import FWCore.ParameterSet.Config as cms

#
#  sequence to make both pixel-based and pixel-less electrons
#  $Id: electronSequence.cff,v 1.5 2007/11/28 22:20:21 futyand Exp $
#
# Created by Shahram Rahatlou, University of Rome & INFN, 4 Aug 2006
#
# sequence to make pixel-based electrons
from RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectronSequence_cff import *
# sequence to make si-strip based electrons
from RecoEgamma.EgammaElectronProducers.siStripElectronSequence_cff import *
electronSequence = cms.Sequence(pixelMatchGsfElectronSequence)


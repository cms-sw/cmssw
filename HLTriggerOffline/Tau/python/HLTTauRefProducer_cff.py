import FWCore.ParameterSet.Config as cms

from HLTriggerOffline.Tau.HLTTauRefProducer_cfi import *
import RecoEgamma.ElectronIdentification.electronIdCutBased_cfi
elecID = RecoEgamma.ElectronIdentification.electronIdCutBased_cfi.eidCutBased.clone()
import RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi
elecIDext = RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi.eidCutBasedExt.clone()
elecID.electronQuality = 'robust'



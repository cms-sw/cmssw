import FWCore.ParameterSet.Config as cms

#
# define our AlCa isolation sequence
#

from RecoEgamma.EgammaIsolationAlgos.eleIsolationSequence_cff import *

#this copy might look silly, but it can be useful to decouple from
#the standard isolation sequence
alcaElectronIsolationSequence=cms.Sequence(eleIsolationSequence)

#
# SA: in case we need to modify the sequence, we should use something like
#
# alcaModule = module_to_modify.clone(value_to_change='newvalue')
# alcaElectronIsolationSequence.replace(module, alcaModule)
#

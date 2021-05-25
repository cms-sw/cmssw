import FWCore.ParameterSet.Config as cms

# This modifier is for configuration changes specific to run-dependent MC.
# In order to preserve the possibily of multi-year MC, only one run-dependent MC
# modifier should exist and be used for all years.

runDependent =  cms.Modifier()

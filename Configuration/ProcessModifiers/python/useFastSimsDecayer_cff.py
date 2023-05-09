import FWCore.ParameterSet.Config as cms

# Desiged to disable the bug in Run II samples that duplicates hits for long lived particles
useFastSimsDecayer =  cms.Modifier()

import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatUtils.l1ECALPrefiringWeightProducer_cfi import l1ECALPrefiringWeightProducer

prefiringweight = l1ECALPrefiringWeightProducer.clone(SkipWarnings = False)


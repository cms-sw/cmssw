import FWCore.ParameterSet.Config as cms

def efficSet(nameIn, titleIn, numeratorIn, denominatorIn, typeIn="eff"):
    pset = cms.PSet(name=cms.untracked.string(nameIn),
                    title=cms.untracked.string(titleIn), 
                    numerator=cms.untracked.string(numeratorIn), 
                    denominator=cms.untracked.string(denominatorIn),
                    type=cms.untracked.string(typeIn))
    return pset


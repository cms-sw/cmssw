import FWCore.ParameterSet.Config as cms

particleFlowTmp = cms.EDProducer("PFProducer")

from Configuration.Eras.Modifier_pf_badHcalMitigation_cff import pf_badHcalMitigation
pf_badHcalMitigation.toModify(particleFlowTmp,
        electron_protectionsForBadHcal = dict(enableProtections = True),
        photon_protectionsForBadHcal   = dict(enableProtections = True))

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(particleFlowTmp,photon_MinEt = 1.)

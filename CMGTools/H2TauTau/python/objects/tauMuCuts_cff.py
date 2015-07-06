import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.objects.muCuts_cff import getMuCuts
from CMGTools.H2TauTau.objects.tauCuts_cff import getTauCuts

tauMuCuts = cms.PSet(
    baseline = cms.PSet(
         mass = cms.string('mass()>10'),
         tauLeg = getTauCuts('leg1','tauMu').clone(),
         muLeg = getMuCuts('leg2', 'tauMu').clone()
         ),
    # this cut is kept out of the baseline for now, until it is studied.
    caloMuVeto = cms.string('leg1().eOverP()>0.2'),
    )

if __name__ == '__main__':

    print tauMuCuts

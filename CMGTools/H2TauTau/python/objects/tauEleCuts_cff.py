import FWCore.ParameterSet.Config as cms

from CMGTools.H2TauTau.objects.eleCuts_cff import getEleCuts
from CMGTools.H2TauTau.objects.tauCuts_cff import getTauCuts
# from CMGTools.H2TauTau.objects.eleSkimCuts_cff import * 
# from CMGTools.H2TauTau.objects.tauSkimCuts_cff import * 

tauEleCuts = cms.PSet(
    baseline = cms.PSet(
         #mass = cms.string('mass()>10'),
         tauLeg = getTauCuts('leg1','tauEle').clone(),
         eleLeg = getEleCuts('leg2', 'tauEle').clone()
         ),
    )

if __name__ == '__main__':

    print tauEleCuts

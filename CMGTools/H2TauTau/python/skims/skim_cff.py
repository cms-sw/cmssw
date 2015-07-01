from CMGTools.H2TauTau.skims.cmgTauMuCount_cfi import *
from CMGTools.H2TauTau.skims.cmgTauEleCount_cfi import *
from CMGTools.H2TauTau.skims.cmgMuEleCount_cfi import *
from CMGTools.H2TauTau.skims.cmgDiTauCount_cfi import *


# tau-mu

cmgTauMuCount.minNumber = 1
tauMuPreSelCount = cmgTauMuCount.clone( minNumber = 1,
                                        src = 'cmgTauMuCorSVFitPreSel')

tauMuFullSelCount = tauMuPreSelCount.clone( src = 'cmgTauMuCorSVFitFullSel')


tauMuPreSelSkimSequence = cms.Sequence(
    tauMuPreSelCount 
    )

tauMuFullSelSkimSequence = cms.Sequence(
    tauMuFullSelCount
    )


# tau-ele

cmgTauEleCount.minNumber = 1
tauElePreSelCount = cmgTauEleCount.clone( minNumber = 1,
                                          src = 'cmgTauEleCorSVFitPreSel')

tauEleFullSelCount = tauElePreSelCount.clone( src = 'cmgTauEleCorSVFitFullSel')


tauElePreSelSkimSequence = cms.Sequence(
    tauElePreSelCount 
    )

tauEleFullSelSkimSequence = cms.Sequence(
    tauEleFullSelCount
    )


# mu-ele

cmgMuEleCount.minNumber = 1
muElePreSelCount = cmgMuEleCount.clone( minNumber = 1,
                                        src = 'cmgMuEleCorSVFitPreSel')

muEleFullSelCount = muElePreSelCount.clone( src = 'cmgMuEleCorSVFitFullSel')


muElePreSelSkimSequence = cms.Sequence(
    muElePreSelCount 
    )

muEleFullSelSkimSequence = cms.Sequence(
    muEleFullSelCount
    )


# tau-tau

cmgDiTauCount.minNumber = 1
diTauPreSelCount = cmgDiTauCount.clone( minNumber = 1,
                                        src = 'cmgDiTauSVFit')

diTauFullSelCount = diTauPreSelCount.clone( src = 'cmgDiTauCorSVFitFullSel')


diTauPreSelSkimSequence = cms.Sequence(
    diTauPreSelCount 
    )

diTauFullSelSkimSequence = cms.Sequence(
    diTauFullSelCount
    )

# mu-mu
diMuPreSelCount = cmgDiTauCount.clone( minNumber = 1,
                                        src = 'cmgDiMuSVFit')

diMuFullSelCount = diMuPreSelCount.clone( src = 'cmgDiMuCorSVFitFullSel')


diMuPreSelSkimSequence = cms.Sequence(
    diMuPreSelCount 
    )

diMuFullSelSkimSequence = cms.Sequence(
    diMuFullSelCount
    )
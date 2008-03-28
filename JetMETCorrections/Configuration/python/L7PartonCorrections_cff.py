import FWCore.ParameterSet.Config as cms

#
# services providing different jet parton corrections
#
# [gJ] (gluons from diJet mixture)
# [qJ] (uds quarks from diJet mixture)
# [cJ] (c quark from diJet mixture)
# [bJ] (b quark from diJet mixture)
# [jJ] (diJet jet mixture)
# [qT] (uds quarks from ttbar events)
# [cT] (c quark from ttbar events)
# [bT] (b quark from ttbar events)
# [jT] (ttbar jet mixture)
#
#======================================//
#== Iterative Cone Algo 0.5 --> IC5  ==//
#======================================//
L7PartonJetCorrectorIC5gJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('gJ'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5gJet')
)

L7PartonJetCorrectorIC5qJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qJ'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5qJet')
)

L7PartonJetCorrectorIC5cJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cJ'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5cJet')
)

L7PartonJetCorrectorIC5bJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bJ'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5bJet')
)

L7PartonJetCorrectorIC5jJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('jJ'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5jJet')
)

L7PartonJetCorrectorIC5qTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qT'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5qTop')
)

L7PartonJetCorrectorIC5cTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cT'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5cTop')
)

L7PartonJetCorrectorIC5bTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bT'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5bTop')
)

L7PartonJetCorrectorIC5tTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('tT'),
    tagName = cms.string('L7parton_IC5_080301'),
    label = cms.string('L7PartonJetCorrectorIC5tTop')
)

#======================================//
#== KT Algo D=0.4 --> KT4            ==//
#======================================//
L7PartonJetCorrectorKT4gJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('gJ'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4gJet')
)

L7PartonJetCorrectorKT4qJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qJ'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4qJet')
)

L7PartonJetCorrectorKT4cJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cJ'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4cJet')
)

L7PartonJetCorrectorKT4bJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bJ'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4bJet')
)

L7PartonJetCorrectorKT4jJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('jJ'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4jJet')
)

L7PartonJetCorrectorKT4qTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qT'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4qTop')
)

L7PartonJetCorrectorKT4cTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cT'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4cTop')
)

L7PartonJetCorrectorKT4bTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bT'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4bTop')
)

L7PartonJetCorrectorKT4tTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('tT'),
    tagName = cms.string('L7parton_KT4_080301'),
    label = cms.string('L7PartonJetCorrectorKT4tTop')
)

#======================================//
#== KT Algo D=0.6 --> KT6            ==//
#======================================//
L7PartonJetCorrectorKT6gJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('gJ'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6gJet')
)

L7PartonJetCorrectorKT6qJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qJ'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6qJet')
)

L7PartonJetCorrectorKT6cJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cJ'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6cJet')
)

L7PartonJetCorrectorKT6bJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bJ'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6bJet')
)

L7PartonJetCorrectorKT6jJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('jJ'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6jJet')
)

L7PartonJetCorrectorKT6qTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qT'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6qTop')
)

L7PartonJetCorrectorKT6cTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cT'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6cTop')
)

L7PartonJetCorrectorKT6bTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bT'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6bTop')
)

L7PartonJetCorrectorKT6tTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('tT'),
    tagName = cms.string('L7parton_KT6_080301'),
    label = cms.string('L7PartonJetCorrectorKT6tTop')
)

#======================================//
#== Sis Cone Algo 0.5 --> SC5        ==//
#======================================//
L7PartonJetCorrectorSC5gJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('gJ'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5gJet')
)

L7PartonJetCorrectorSC5qJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qJ'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5qJet')
)

L7PartonJetCorrectorSC5cJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cJ'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5cJet')
)

L7PartonJetCorrectorSC5bJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bJ'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5bJet')
)

L7PartonJetCorrectorSC5jJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('jJ'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5jJet')
)

L7PartonJetCorrectorSC5qTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qT'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5qTop')
)

L7PartonJetCorrectorSC5cTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cT'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5cTop')
)

L7PartonJetCorrectorSC5bTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bT'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5bTop')
)

L7PartonJetCorrectorSC5tTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('tT'),
    tagName = cms.string('L7parton_SC5_080301'),
    label = cms.string('L7PartonJetCorrectorSC5tTop')
)

#======================================//
#== Sis Cone Algo 0.7 --> SC7        ==//
#======================================//
L7PartonJetCorrectorSC7gJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('gJ'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7gJet')
)

L7PartonJetCorrectorSC7qJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qJ'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7qJet')
)

L7PartonJetCorrectorSC7cJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cJ'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7cJet')
)

L7PartonJetCorrectorSC7bJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bJ'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7bJet')
)

L7PartonJetCorrectorSC7jJet = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('jJ'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7jJet')
)

L7PartonJetCorrectorSC7qTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('qT'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7qTop')
)

L7PartonJetCorrectorSC7cTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('cT'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7cTop')
)

L7PartonJetCorrectorSC7bTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('bT'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7bTop')
)

L7PartonJetCorrectorSC7tTop = cms.ESSource("L7PartonCorrectionService",
    section = cms.string('tT'),
    tagName = cms.string('L7parton_SC7_080301'),
    label = cms.string('L7PartonJetCorrectorSC7tTop')
)



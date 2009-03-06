import FWCore.ParameterSet.Config as cms

paths = ['veryHighEtDQM',
         'singlePhotonRelaxedDQM',
         'singlePhotonDQM',
         'singleElectronRelaxedDQM',
         'singleElectronDQM',
         'singleElectronRelaxedLargeWindowDQM', 
         'singleElectronLargeWindowDQM',
         'highEtDQM',
         'doublePhotonRelaxedDQM',
         'doublePhotonDQM', 
         'doubleElectronRelaxedDQM',
         'doubleElectronDQM']


#define common modules
leptons = cms.EDFilter("PdgIdAndStatusCandViewSelector",
    status = cms.vint32(1),
    src = cms.InputTag("genParticles"),
    pdgId = cms.vint32(11)
)
cut = cms.EDFilter("EtaPtMinCandViewSelector",
    src = cms.InputTag("leptons"),
    etaMin = cms.double(-2.5),
    etaMax = cms.double(2.5),
    ptMin = cms.double(2.0)
)

#define sequences/noncommon modules
selZ = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("cut"),
    minNumber = cms.uint32(2)
)
Zseq='*('

selW = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("cut"),
    minNumber = cms.uint32(1)
)
Wseq='*('

first= True
#load modules
for trig in paths:
    if not first:
        Zseq=Zseq+'+'
        Wseq=Wseq+'+'
    first= False    

    imp = 'from HLTriggerOffline.Egamma.' + trig + '_cfi import *'
    exec imp

    #clone for Z
    clon = trig + '_Z = ' + trig + '.clone()'
    exec clon
    #adjust MC match pid
    mcmatch = trig + '_Z.pdgGen=11'
    exec mcmatch
    Zseq=Zseq + trig + '_Z' 

    #clone for W
    clon = trig + '_W = ' + trig + '.clone()'
    exec clon
    #adjust MC match pid
    mcmatch = trig + '_Z.pdgGen=11'
    exec mcmatch
    Wseq=Wseq + trig + '_W' 

Zseq=Zseq + ')'
Wseq=Wseq + ')'


scom = 'egammavalZee = cms.Sequence(leptons*cut*selZ' + Zseq +')'
exec scom
scom = 'egammavalWenu = cms.Sequence(leptons*cut*selW' + Wseq +')'
exec scom

import FWCore.ParameterSet.Config as cms

from RecoMuon.MuonIsolation.muonPFIsolation_cff import *
muPFIsoDepositChargedForBoostedMuons = muPFIsoDepositCharged.clone(
    src = cms.InputTag('muons')
)    
muPFIsoDepositNeutralForBoostedMuons = muPFIsoDepositNeutral.clone(
    src = cms.InputTag('muons')
)    
muPFIsoDepositGammaForBoostedMuons = muPFIsoDepositGamma.clone(
    src = cms.InputTag('muons')
)    
muPFIsoDepositChargedAllForBoostedMuons = muPFIsoDepositChargedAll.clone(
    src = cms.InputTag('muons')
)
muPFIsoDepositPUforBoostedTauStudy = muPFIsoDepositPU.clone(
    src = cms.InputTag('muons')
)
boostedMuonPFIsolationSequence = cms.Sequence(
    muPFIsoDepositChargedForBoostedMuons
   + muPFIsoDepositNeutralForBoostedMuons
   + muPFIsoDepositGammaForBoostedMuons
   + muPFIsoDepositChargedAllForBoostedMuons
   + muPFIsoDepositPUforBoostedTauStudy
)

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import patMuons
patBoostedMuons = patMuons.clone(
    isoDeposits = cms.PSet(
        # CV: strings for IsoDeposits defined in PhysicsTools/PatAlgos/plugins/PATMuonProducer.cc
        pfChargedHadrons = cms.InputTag("muPFIsoDepositChargedForBoostedMuons"),
        pfNeutralHadrons = cms.InputTag("muPFIsoDepositNeutralForBoostedMuons"),
        pfPhotons = cms.InputTag("muPFIsoDepositGammaForBoostedMuons"),
        user = cms.VInputTag(
            cms.InputTag("muPFIsoDepositChargedAllForBoostedMuons"),
            cms.InputTag("muPFIsoDepositPUforBoostedTauStudy")
       )
    ),
    addGenMatch = cms.bool(False),
    embedHighLevelSelection = cms.bool(True),
    embedCaloMETMuonCorrs = cms.bool(False),
    embedTcMETMuonCorrs = cms.bool(False),
    usePV = cms.bool(False) # compute transverse impact parameter wrt. beamspot (not event vertex)
)

otherSubJetVeto = 'OtherJetConstituentsDeltaRVeto(boostedTauSeeds,0.3,boostedTauSeeds:pfCandAssocMapForIsoDepositVetos,0.01)'

pfChargedHadronVetos = muPFIsoValueCharged04.deposits[0].vetos.value()
pfChargedHadronVetos.append(otherSubJetVeto)
print "pfChargedHadronVetos = %s" % pfChargedHadronVetos

pfNeutralHadronVetos = muPFIsoValueNeutral04.deposits[0].vetos.value()
pfNeutralHadronVetos.append(otherSubJetVeto)
print "pfNeutralHadronVetos = %s" % pfNeutralHadronVetos

pfGammaVetos = muPFIsoValueGamma04.deposits[0].vetos.value()
pfGammaVetos.append(otherSubJetVeto)
print "pfGammaVetos = %s" % pfGammaVetos

userVetos1 = muPFIsoValueChargedAll04.deposits[0].vetos.value()
userVetos1.append(otherSubJetVeto)
print "userVetos1 = %s" % userVetos1

userVetos2 = muPFIsoValuePU04.deposits[0].vetos.value()
userVetos2.append(otherSubJetVeto)
print "userVetos2 = %s" % userVetos2

patBoostedMuons.userIsolation = cms.PSet(
    # CV: strings for Isolation values defined in PhysicsTools/PatAlgos/src/MultiIsolator.cc
    pfChargedHadron = cms.PSet(
        deltaR = cms.double(0.4),
        src = patBoostedMuons.isoDeposits.pfChargedHadrons,
        vetos = cms.vstring(pfChargedHadronVetos),
        skipDefaultVeto = muPFIsoValueCharged04.deposits[0].skipDefaultVeto
    ),
    pfNeutralHadron = cms.PSet(
        deltaR = cms.double(0.4), 
        src = patBoostedMuons.isoDeposits.pfNeutralHadrons,
        vetos = cms.vstring(pfNeutralHadronVetos),
        skipDefaultVeto = muPFIsoValueNeutral04.deposits[0].skipDefaultVeto
    ),
    pfGamma = cms.PSet(
        deltaR = cms.double(0.4), 
        src = patBoostedMuons.isoDeposits.pfPhotons,
        vetos = cms.vstring(pfGammaVetos),
        skipDefaultVeto = muPFIsoValueGamma04.deposits[0].skipDefaultVeto
    ),
    user = cms.VPSet(
        cms.PSet(
            deltaR = cms.double(0.4),
            src = patBoostedMuons.isoDeposits.user[0],
            vetos = cms.vstring(userVetos1),
            skipDefaultVeto = muPFIsoValueChargedAll04.deposits[0].skipDefaultVeto
        ),
        cms.PSet(
            deltaR = cms.double(0.4),
            src = patBoostedMuons.isoDeposits.user[1],
            vetos = cms.vstring(userVetos2),
            skipDefaultVeto = muPFIsoValuePU04.deposits[0].skipDefaultVeto
        )
    )
)

makePatBoostedMuons = cms.Sequence(
    boostedMuonPFIsolationSequence
   + patBoostedMuons
)

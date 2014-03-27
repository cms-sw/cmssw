import FWCore.ParameterSet.Config as cms

from CommonTools.ParticleFlow.Isolation.pfElectronIsolation_cff import *

elPFIsoDepositChargedForBoostedElectrons = elPFIsoDepositCharged.clone(
   src = cms.InputTag('gedGsfElectrons'),    
)
elPFIsoDepositChargedAllForBoostedElectrons = elPFIsoDepositChargedAll.clone(
   src = cms.InputTag('gedGsfElectrons'),   
)
elPFIsoDepositNeutralForBoostedElectrons = elPFIsoDepositNeutral.clone(
   src = cms.InputTag('gedGsfElectrons'),
)
elPFIsoDepositGammaForBoostedElectrons = elPFIsoDepositGamma.clone(
   src = cms.InputTag('gedGsfElectrons'),
)
elPFIsoDepositPUForBoostedElectrons = elPFIsoDepositPU.clone(
   src = cms.InputTag('gedGsfElectrons'),
)
boostedElectronPFIsolationSequence = cms.Sequence(
    elPFIsoDepositChargedForBoostedElectrons
  + elPFIsoDepositChargedAllForBoostedElectrons
  + elPFIsoDepositNeutralForBoostedElectrons
  + elPFIsoDepositGammaForBoostedElectrons
  + elPFIsoDepositPUForBoostedElectrons
)

from PhysicsTools.PatAlgos.producersLayer1.electronProducer_cfi import patElectrons
patBoostedElectrons = patElectrons.clone(
    isoDeposits = cms.PSet(
        # CV: strings for IsoDeposits defined in PhysicsTools/PatAlgos/plugins/PATElectronProducer.cc
        pfChargedHadrons = cms.InputTag("elPFIsoDepositChargedForBoostedElectrons"),
        pfNeutralHadrons = cms.InputTag("elPFIsoDepositNeutralForBoostedElectrons"),
        pfPhotons = cms.InputTag("elPFIsoDepositGammaForBoostedElectrons"),
        user = cms.VInputTag(
            cms.InputTag("elPFIsoDepositChargedAllForBoostedElectrons"),
            cms.InputTag("elPFIsoDepositPUForBoostedElectrons")
        )
    ),
    addGenMatch = cms.bool(False),
    embedHighLevelSelection = cms.bool(True),
    usePV = cms.bool(False) # compute transverse impact parameter wrt. beamspot (not event vertex)      
)

otherSubJetVeto = 'OtherJetConstituentsDeltaRVeto(boostedTauSeeds,0.3,boostedTauSeeds:pfCandAssocMapForIsoDepositVetos,0.01)'

#pfChargedHadronVetos = elPFIsoValueCharged04NoPFId.deposits[0].vetos.value()
pfChargedHadronVetos = elPFIsoValueCharged03NoPFId.deposits[0].vetos.value()
pfChargedHadronVetos.append(otherSubJetVeto)
print "pfChargedHadronVetos = %s" % pfChargedHadronVetos

#pfNeutralHadronVetos = elPFIsoValueNeutral04NoPFId.deposits[0].vetos.value()
pfNeutralHadronVetos = elPFIsoValueNeutral03NoPFId.deposits[0].vetos.value()
pfNeutralHadronVetos.append(otherSubJetVeto)
print "pfNeutralHadronVetos = %s" % pfNeutralHadronVetos

#pfGammaVetos = elPFIsoValueGamma04NoPFId.deposits[0].vetos.value()
pfGammaVetos = elPFIsoValueGamma03NoPFId.deposits[0].vetos.value()
pfGammaVetos.append(otherSubJetVeto)
print "pfGammaVetos = %s" % pfGammaVetos

#userVetos1 = elPFIsoValueChargedAll04NoPFId.deposits[0].vetos.value()
userVetos1 = elPFIsoValueChargedAll03NoPFId.deposits[0].vetos.value()
userVetos1.append(otherSubJetVeto)
print "userVetos1 = %s" % userVetos1

#userVetos2 = elPFIsoValuePU04NoPFId.deposits[0].vetos.value()
userVetos2 = elPFIsoValuePU03NoPFId.deposits[0].vetos.value()
userVetos2.append(otherSubJetVeto)
print "userVetos2 = %s" % userVetos2

patBoostedElectrons.userIsolation = cms.PSet(
   # CV: strings for Isolation values defined in PhysicsTools/PatAlgos/src/MultiIsolator.cc
   pfChargedHadron = cms.PSet(
      #  deltaR = cms.double(0.4),
        deltaR = cms.double(0.3),
        src = patBoostedElectrons.isoDeposits.pfChargedHadrons,
        vetos = cms.vstring(pfChargedHadronVetos),
      #  skipDefaultVeto = elPFIsoValueCharged04NoPFId.deposits[0].skipDefaultVeto
        skipDefaultVeto = elPFIsoValueCharged03NoPFId.deposits[0].skipDefaultVeto
   ),
   pfNeutralHadron = cms.PSet(
      #  deltaR = cms.double(0.4),
        deltaR = cms.double(0.3),
        src = patBoostedElectrons.isoDeposits.pfNeutralHadrons,
        vetos = cms.vstring(pfNeutralHadronVetos),
      #  skipDefaultVeto = elPFIsoValueNeutral04NoPFId.deposits[0].skipDefaultVeto
        skipDefaultVeto = elPFIsoValueNeutral03NoPFId.deposits[0].skipDefaultVeto
   ), 
   pfGamma = cms.PSet(
      #  deltaR = cms.double(0.4),
        deltaR = cms.double(0.3),
        src = patBoostedElectrons.isoDeposits.pfPhotons,
        vetos = cms.vstring(pfGammaVetos),
      #  skipDefaultVeto = elPFIsoValueGamma04NoPFId.deposits[0].skipDefaultVeto
        skipDefaultVeto = elPFIsoValueGamma03NoPFId.deposits[0].skipDefaultVeto 
   ),
   user = cms.VPSet(
        cms.PSet(
      #      deltaR = cms.double(0.4),
            deltaR = cms.double(0.3),
            src = patBoostedElectrons.isoDeposits.user[0],
            vetos = cms.vstring(userVetos1),
      #      skipDefaultVeto = elPFIsoValueChargedAll04NoPFId.deposits[0].skipDefaultVeto
            skipDefaultVeto = elPFIsoValueChargedAll03NoPFId.deposits[0].skipDefaultVeto
        ),
        cms.PSet(
      #      deltaR = cms.double(0.4),
            deltaR = cms.double(0.3),
            src = patBoostedElectrons.isoDeposits.user[1],
            vetos = cms.vstring(userVetos2),
      #      skipDefaultVeto = elPFIsoValuePU04NoPFId.deposits[0].skipDefaultVeto
            skipDefaultVeto = elPFIsoValuePU03NoPFId.deposits[0].skipDefaultVeto
        )
   ) 

)

makePatBoostedElectrons = cms.Sequence(
    boostedElectronPFIsolationSequence
   + patBoostedElectrons
)



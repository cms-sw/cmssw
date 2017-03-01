#from RecoParticleFlow.PFProducer.electronPFIsolationValues_cff import *
#from CommonTools.ParticleFlow.PFBRECO_cff import *
from CommonTools.ParticleFlow.Isolation.tools_cfi import *
import FWCore.ParameterSet.Config as cms
#Now prepare the iso deposits
elPFIsoDepositChargedGsf=isoDepositReplace('gedGsfElectrons','pfAllChargedHadrons')
elPFIsoDepositChargedAllGsf=isoDepositReplace('gedGsfElectrons','pfAllChargedParticles')
elPFIsoDepositNeutralGsf=isoDepositReplace('gedGsfElectrons','pfAllNeutralHadrons')
elPFIsoDepositPUGsf=isoDepositReplace('gedGsfElectrons','pfPileUpAllChargedParticles')
#elPFIsoDepositGamma=isoDepositReplace('pfSelectedElectrons','pfAllPhotons')
elPFIsoDepositGammaGsf= cms.EDProducer("CandIsoDepositProducer",
                                       src = cms.InputTag("gedGsfElectrons"),
                                       MultipleDepositsFlag = cms.bool(False),
                                       trackType = cms.string('candidate'),
                                       ExtractorPSet = cms.PSet(
    Diff_z = cms.double(99999.99),
    ComponentName = cms.string('PFCandWithSuperClusterExtractor'),
    DR_Max = cms.double(1.0),
    Diff_r = cms.double(99999.99),
    inputCandView = cms.InputTag("pfAllPhotons"),
    DR_Veto = cms.double(0),
    SCMatch_Veto = cms.bool(False),
    MissHitSCMatch_Veto = cms.bool(True),
    DepositLabel = cms.untracked.string('')
    )
                                       )

# elPFIsoDepositChargedGsf= elPFIsoDepositCharged.clone()
# elPFIsoDepositChargedGsf.src = 'gedGsfElectrons'
# elPFIsoDepositChargedAllGsf = elPFIsoDepositChargedAll.clone()
# elPFIsoDepositChargedAllGsf.src = 'gedGsfElectrons'
# elPFIsoDepositNeutralGsf = elPFIsoDepositNeutral.clone()
# elPFIsoDepositNeutralGsf.src = 'gedGsfElectrons'
# elPFIsoDepositGammaGsf = elPFIsoDepositGamma.clone()
# elPFIsoDepositGammaGsf.src = 'gedGsfElectrons'
# elPFIsoDepositPUGsf = elPFIsoDepositPU.clone()
# elPFIsoDepositPUGsf.src = 'gedGsfElectrons'

elPFIsoValueCharged03PFIdGsf = cms.EDProducer("PFCandIsolatorFromDeposits",
                                           deposits = cms.VPSet(
    cms.PSet(
    src = cms.InputTag("elPFIsoDepositChargedGsf"),
    deltaR = cms.double(0.3),
    weight = cms.string('1'),
    vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
    skipDefaultVeto = cms.bool(True),
    mode = cms.string('sum'),
    PivotCoordinatesForEBEE = cms.bool(True)
    )
    )
                                           )
elPFIsoValueChargedAll03PFIdGsf = cms.EDProducer("PFCandIsolatorFromDeposits",
                                                 deposits = cms.VPSet(
          cms.PSet(
          src = cms.InputTag("elPFIsoDepositChargedAllGsf"),
          deltaR = cms.double(0.3),
          weight = cms.string('1'),
          vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
          skipDefaultVeto = cms.bool(True),
          mode = cms.string('sum'),
          PivotCoordinatesForEBEE = cms.bool(True)
   )
 )
                                              )

elPFIsoValueGamma03PFIdGsf = cms.EDProducer("PFCandIsolatorFromDeposits",
                                         deposits = cms.VPSet(
           cms.PSet(
           src = cms.InputTag("elPFIsoDepositGammaGsf"),
           deltaR = cms.double(0.3),
           weight = cms.string('1'),
           vetos = cms.vstring('EcalEndcaps:ConeVeto(0.08)'),
           skipDefaultVeto = cms.bool(True),
           mode = cms.string('sum'),
           PivotCoordinatesForEBEE = cms.bool(True)
     )
  )
)

elPFIsoValueNeutral03PFIdGsf = cms.EDProducer("PFCandIsolatorFromDeposits",
                                           deposits = cms.VPSet(
             cms.PSet(
             src = cms.InputTag("elPFIsoDepositNeutralGsf"),
             deltaR = cms.double(0.3),
             weight = cms.string('1'),
             vetos = cms.vstring(),
             skipDefaultVeto = cms.bool(True),
             mode = cms.string('sum'),
             PivotCoordinatesForEBEE = cms.bool(True)
             )
         )
                                           )

elPFIsoValuePU03PFIdGsf = cms.EDProducer("PFCandIsolatorFromDeposits",
                                      deposits = cms.VPSet(
           cms.PSet(
           src = cms.InputTag("elPFIsoDepositPUGsf"),
           deltaR = cms.double(0.3),
           weight = cms.string('1'),
           vetos = cms.vstring('EcalEndcaps:ConeVeto(0.015)'),
           skipDefaultVeto = cms.bool(True),
           mode = cms.string('sum'),
           PivotCoordinatesForEBEE = cms.bool(True)
           )
  )
                                         )


# elPFIsoValueCharged03PFIdGsf = elPFIsoValueCharged03PFId.clone()
# #elPFIsoValueCharged03PFIdGsf.deposits[0].src = cms.InputTag(elPFIsoDepositChargedGsf)

# elPFIsoValueChargedAll03PFIdGsf = elPFIsoValueChargedAll03PFId.clone()
# #elPFIsoValueChargedAll03PFIdGsf.deposits[0].src = cms.InputTag(elPFIsoDepositChargedAllGsf)

# elPFIsoValueGamma03PFIdGsf = elPFIsoValueGamma03PFId.clone()
# #elPFIsoValueGamma03PFIdGsf.deposits[0].src = cms.InputTag(elPFIsoDepositGammaGsf)


# elPFIsoValueNeutral03PFIdGsf = elPFIsoValueNeutral03PFId.clone()
# #elPFIsoValueNeutral03PFIdGsf.deposits[0].src = cms.InputTag(elPFIsoDepositNeutralGsf)

# elPFIsoValuePU03PFIdGsf = elPFIsoValuePU03PFId.clone()
# #elPFIsoValuePU03PFIdGsf.deposits[0].src = cms.InputTag(elPFIsoDepositPUGsf)

eleIsoSequence = cms.Sequence((elPFIsoDepositChargedGsf + elPFIsoDepositChargedAllGsf + elPFIsoDepositNeutralGsf + elPFIsoDepositGammaGsf + elPFIsoDepositPUGsf))
eleIsoSequence *= cms.Sequence(elPFIsoValueCharged03PFIdGsf+elPFIsoValueChargedAll03PFIdGsf+elPFIsoValueGamma03PFIdGsf+elPFIsoValueNeutral03PFIdGsf+elPFIsoValuePU03PFIdGsf )
pfisoALCARECO = cms.Sequence(eleIsoSequence) #pfParticleSelectionSequence + eleIsoSequence)

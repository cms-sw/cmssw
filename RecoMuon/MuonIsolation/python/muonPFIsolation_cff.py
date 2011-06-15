import FWCore.ParameterSet.Config as cms

#Import sequence frokm ParticleFlow
from CommonTools.ParticleFlow.ParticleSelectors.pfCandsForIsolation_cff import *


#Create some additional collection (NEEDS TO BE DISCUSSED WITH E/G friends to not double create) 


#Create the PU candidates
pfPileUpCandidates = cms.EDProducer(
         "TPPFCandidatesOnPFCandidates",
         enable =  cms.bool( True ),
         verbose = cms.untracked.bool( False ),
         name = cms.untracked.string("pileUpCandidates"),
         topCollection = cms.InputTag("pfNoPileUp"),
         bottomCollection = cms.InputTag("particleFlowTmp"),
)


#Take the PU charged particles
pfPUChargedCandidates = cms.EDFilter("PdgIdPFCandidateSelector",
                                          src = cms.InputTag("pfPileUpCandidates"),
                                          pdgId = cms.vint32(211,-211,321,-321,999211,2212,-2212,11,-11,13,-13)
)
                                                                                                                        

#Create All Charged Particles
pfAllChargedCandidates = cms.EDFilter("PdgIdPFCandidateSelector",
                              src = cms.InputTag("pfNoPileUp"),
                              pdgId = cms.vint32(211,-211,321,-321,999211,2212,-2212,11,-11,13,-13)
                          )


muonPrePFIsolationSequence =  cms.Sequence(pfCandsForIsolationSequence+
                                           pfPileUpCandidates+
                                           pfPUChargedCandidates+
                                           pfAllChargedCandidates
)                                         

#######################################################################################################
#######################################################################################################
#######################################################################################################
from CommonTools.ParticleFlow.Isolation.tools_cfi import *


#Now prepare the iso deposits
muPFIsoDepositCharged=isoDepositReplace('muons1stStep','pfAllChargedHadrons')
muPFIsoDepositChargedAll=isoDepositReplace('muons1stStep','pfAllChargedCandidates')
muPFIsoDepositNeutral=isoDepositReplace('muons1stStep','pfAllNeutralHadrons')
muPFIsoDepositGamma=isoDepositReplace('muons1stStep','pfAllPhotons')
muPFIsoDepositPU=isoDepositReplace('muons1stStep','pfPUChargedCandidates')



#Now create isolation values for those isolation deposits 

muPFIsoValueCharged03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)

muPFIsoValueChargedAll03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

muPFIsoValueGamma03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)

muPFIsoValueNeutral03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )
)

muPFIsoValuePU03 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.3),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)



muPFIsoValueCharged04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositCharged"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
            )
     )
)




muPFIsoValueChargedAll04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositChargedAll"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.0001','Threshold(0.0)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
     )
   )
)

muPFIsoValueGamma04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositGamma"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)


muPFIsoValueNeutral04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositNeutral"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
    )
 )

)
muPFIsoValuePU04 = cms.EDProducer("CandIsolatorFromDeposits",
    deposits = cms.VPSet(
            cms.PSet(
            src = cms.InputTag("muPFIsoDepositPU"),
            deltaR = cms.double(0.4),
            weight = cms.string('1'),
            vetos = cms.vstring('0.01','Threshold(0.5)'),
            skipDefaultVeto = cms.bool(True),
            mode = cms.string('sum')
      )
   )
)


muonPFIsolationSequence =  cms.Sequence(muPFIsoDepositCharged+
                                        muPFIsoDepositChargedAll+
                                        muPFIsoDepositGamma+
                                        muPFIsoDepositNeutral+
                                        muPFIsoDepositPU+
                                        ##############################
                                        muPFIsoValueCharged03+
                                        muPFIsoValueChargedAll03+
                                        muPFIsoValueGamma03+
                                        muPFIsoValueNeutral03+
                                        muPFIsoValuePU03+
                                       ############################## 
                                        muPFIsoValueCharged04+
                                        muPFIsoValueChargedAll04+
                                        muPFIsoValueGamma04+
                                        muPFIsoValueNeutral04+
                                        muPFIsoValuePU04

)                                         





                 


import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

# # PHISYM producer
from Calibration.EcalCalibAlgos.EcalPhiSymRecHitProducers_cfi import EcalPhiSymRecHitProducerRun, EcalPhiSymRecHitProducerLumi

# Sum info by run or lumi sequences
ALCARECOEcalPhiSymRecHitProducerRun = EcalPhiSymRecHitProducerRun.clone()
ALCARECOEcalPhiSymRecHitProducerLumi = EcalPhiSymRecHitProducerLumi.clone()

# # NANOAOD flat table producers
import Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi
ALCARECOecalPhiSymRecHitRunTableEB = Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi.ecalPhiSymRecHitRunTableEB
ALCARECOecalPhiSymRecHitRunTableEE = Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi.ecalPhiSymRecHitRunTableEE 
ALCARECOecalPhiSymInfoRunTable = Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi.ecalPhiSymInfoRunTable
ALCARECOecalPhiSymRecHitLumiTableEB = Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi.ecalPhiSymRecHitLumiTableEB
ALCARECOecalPhiSymRecHitLumiTableEE = Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi.ecalPhiSymRecHitLumiTableEE 
ALCARECOecalPhiSymInfoLumiTable = Calibration.EcalCalibAlgos.EcalPhiSymFlatTableProducers_cfi.ecalPhiSymInfoLumiTable

ALCARECOecalPhiSymRecHitRunTableEB.src = cms.InputTag("ALCARECOEcalPhiSymRecHitProducerRun", "EB")
ALCARECOecalPhiSymRecHitRunTableEE.src = cms.InputTag("ALCARECOEcalPhiSymRecHitProducerRun", "EE")
ALCARECOecalPhiSymInfoRunTable.src = cms.InputTag("ALCARECOEcalPhiSymRecHitProducerRun")
ALCARECOecalPhiSymRecHitLumiTableEB.src = cms.InputTag("ALCARECOEcalPhiSymRecHitProducerLumi", "EB")
ALCARECOecalPhiSymRecHitLumiTableEE.src = cms.InputTag("ALCARECOEcalPhiSymRecHitProducerLumi", "EE")
ALCARECOecalPhiSymInfoLumiTable.src = cms.InputTag("ALCARECOEcalPhiSymRecHitProducerLumi")

nmis = ALCARECOEcalPhiSymRecHitProducerRun.nMisCalib.value()
for imis in range(1, nmis+1):
    # get the naming and indexing right.
    if imis<nmis/2+1:
        var_name = 'sumEt_m'+str(abs(int(imis-(nmis/2)-1)))
        var = Var(f'sumEt({imis})', float, doc='ECAL PhiSym rechits: '+str(imis-(nmis/2)-1)+'*miscalib et', precision=23)
    else:
        var_name = 'sumEt_p'+str(int(imis-(nmis/2)))
        var = Var(f'sumEt({imis})', float, doc='ECAL PhiSym rechits: '+str(imis-(nmis/2))+'*miscalib et', precision=23)
        
    setattr(ALCARECOecalPhiSymRecHitRunTableEB.variables, var_name, var)
    setattr(ALCARECOecalPhiSymRecHitRunTableEE.variables, var_name, var)
    setattr(ALCARECOecalPhiSymRecHitLumiTableEB.variables, var_name, var)
    setattr(ALCARECOecalPhiSymRecHitLumiTableEE.variables, var_name, var)


seqALCARECOEcalPhiSymByRun = cms.Sequence( ALCARECOEcalPhiSymRecHitProducerRun *
                                           ALCARECOecalPhiSymRecHitRunTableEB * 
                                           ALCARECOecalPhiSymRecHitRunTableEE *
                                           ALCARECOecalPhiSymInfoRunTable )
seqALCARECOEcalPhiSymByLumi = cms.Sequence( ALCARECOEcalPhiSymRecHitProducerLumi *
                                            ALCARECOecalPhiSymRecHitLumiTableEB * 
                                            ALCARECOecalPhiSymRecHitLumiTableEE *
                                            ALCARECOecalPhiSymInfoLumiTable )


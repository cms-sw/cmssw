import FWCore.ParameterSet.Config as cms

from Calibration.LumiAlCaRecoProducers.alcaRawPCCProducer_cfi import *
ALCARECORawPCCProd = rawPCCProd.clone()
ALCARECORawPCCProd.RawPCCProducerParameters.inputPccLabel="alcaPCCIntegratorZeroBias"
ALCARECORawPCCProd.RawPCCProducerParameters.ProdInst="alcaPCCZeroBias"
ALCARECORawPCCProd.RawPCCProducerParameters.outputProductName="rawPCCProd"
ALCARECORawPCCProd.RawPCCProducerParameters.OutputValue="Average"
ALCARECORawPCCProd.RawPCCProducerParameters.ApplyCorrections=True
ALCARECORawPCCProd.RawPCCProducerParameters.saveCSVFile=False  # .csv file may be retrived from PromtReco Crab jobs (saved into dataset)

#additional instance with no background corrections, useful for low lumi runs where corrections are invalid
ALCARECORawPCCProdUnCorr = ALCARECORawPCCProd.clone() 
ALCARECORawPCCProdUnCorr.RawPCCProducerParameters.ApplyCorrections=False

seqALCARECORawPCCProducer = cms.Sequence(ALCARECORawPCCProd+ALCARECORawPCCProdUnCorr)

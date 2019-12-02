###
## author: J. F. Benitez (benitezj@cern.ch)
###
## description: unit test for the LumiAlCaRecoProducers 

import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
DQMStore = cms.Service("DQMStore")
process = cms.Process("TEST")
process.load("FWCore.Framework.test.cmsExceptionsFatal_cff")
process.load("DQMServices.Core.DQM_cfg")
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(5))

process.source = cms.Source("PoolSource",
                            fileNames =  cms.untracked.vstring(
                                'root://xrootd-cms.infn.it//store/data/Commissioning2018/AlCaLumiPixels0/ALCARECO/AlCaPCCRandom-02May2018-v1/70000/6CE65B93-D657-E811-BBB5-FA163E8006EF.root',
                                #'file:/eos/cms/store/data/Run2017G/AlCaLumiPixels/ALCARECO/AlCaPCCRandom-17Nov2017-v1/20000/229B17F0-2B93-E811-9506-0025905B8568.root'
                            )
)


process.raw = cms.EDProducer("RawPCCProducer",
                             RawPCCProducerParameters = cms.PSet(inputPccLabel = cms.string("alcaPCCProducerRandom"),
                                                                 ProdInst = cms.string("alcaPCCRandom"),
                                                                 modVeto=cms.vint32(), 
                                                                 outputProductName = cms.untracked.string("raw"),
                                                             )
                         )

process.cor = DQMEDAnalyzer("CorrPCCProducer",
                            CorrPCCProducerParameters=cms.PSet(inLumiObLabel = cms.string("raw"),
                                                               ProdInst = cms.string("raw"),
                                                               approxLumiBlockSize=cms.int32(50),
                                                               type2_a= cms.double(0.00072),
                                                               type2_b= cms.double(0.014),))



process.load("CondCore.CondDB.CondDB_cfi")
process.CondDB.connect = "sqlite_file:PCC_Corr.db"
process.PoolDBOutputService = cms.Service("PoolDBOutputService", process.CondDB,
                                          toPut = cms.VPSet(cms.PSet(record = cms.string('LumiCorrectionsRcd'),
                                                                     tag = cms.string('TestCorrections'))),
                                          loadBlobStreamer = cms.untracked.bool(False),
                                          timetype   = cms.untracked.string('lumiid'))


process.p1 = cms.Path(process.raw+process.cor)

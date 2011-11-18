import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing
import os

process = cms.Process("V2Analyzer")
# import of standard configurations
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/MagneticField_38T_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load('RecoHI.HiEvtPlaneAlgos.HiEvtPlane_cfi')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('RecoHI.HiEvtPlaneAlgos.hievtplaneflatproducer_cfi')


process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'GR_R_41_V0::All'

process.load('Configuration/EventContent/EventContentHeavyIons_cff')


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound')
)

process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(
        'file:myfile.root'
    )
)
fileName = 'file:/store/user/appeltel/HIAllPhysics/Flow_Skim_Run2010_HIAllPhysics-batch2/2337c3cdf19221b3c4af10fbcbd13096/flowskim_9_1_qQX.root'

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(fileName),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
                            inputCommands = cms.untracked.vstring('keep *'),
                            dropDescendantsOfDroppedBranches=cms.untracked.bool(False)
)



process.TFileService = cms.Service("TFileService",
    fileName = cms.string("rpflat.root")
)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.load("CondCore.DBCommon.CondDBCommon_cfi");


process.HeavyIonGlobalParameters = cms.PSet(
    centralitySrc = cms.InputTag("hiCentrality"),
    centralityVariable = cms.string("HFtowers"),
    nonDefaultGlauberModel = cms.string("")
    )
from CmsHi.Analysis2010.CommonFunctions_cff import *
overrideCentrality(process)

process.dump = cms.EDAnalyzer("EventContentAnalyzer")

process.CondDBCommon.connect = "sqlite_file:/gpfs20/home/sanders5/CMSSW_4_1_4/src/RecoHI/HiEvtPlaneAlgos/data/flatparms.db"
process.PoolDBESSource2 = cms.ESSource("PoolDBESSource",
                                      process.CondDBCommon,
                                      toGet = cms.VPSet(cms.PSet(record = cms.string('HeavyIonRPRcd'),
                                                                 tag = cms.string('flatParamtest')
                                                                 )
                                                        )
                                      )

process.hiEvtPlane.useTrackPtWeight_ = cms.untracked.bool(True)
process.hiEvtPlane.minpt_ = cms.untracked.double(0.3)
process.hiEvtPlane.maxpt_ = cms.untracked.double(2.6)
process.hiEvtPlane.biggap_ = cms.untracked.double(0.3)
process.hiEvtPlane.minvtx_ = cms.untracked.double(-10.)
process.hiEvtPlane.maxvtx_ = cms.untracked.double(10.)
process.hiEvtPlane.dzerr_ = cms.untracked.double(10.)
process.hiEvtPlane.chi2_ = cms.untracked.double(40.)
process.v2analyzer.dzerr_ = cms.untracked.double(10.)
process.v2analyzer.chi2_ = cms.untracked.double(40.)
process.v2analyzer.jetAnal_ = cms.untracked.bool(True)

process.p = cms.Path(process.hiEvtPlane*process.hiEvtPlaneFlat)

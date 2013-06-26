# The following comments couldn't be translated into the new config version:

#read constants from DB

import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
process.load("CondCore.DBCommon.CondDBSetup_cfi")

process.PoolDBESSource = cms.ESSource("PoolDBESSource",
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb'),
        authenticationMethod = cms.untracked.uint32(1)
    ),
#    loadAll = cms.bool(True),
    timetype = cms.string('runnumber'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('CSCChamberTimeCorrectionsRcd'),
        tag = cms.string('CSCChamberTimeCorrections_testanode')
    )),
    #connect = cms.string('frontier://FrontierDev/CMS_COND_CSC')
    #connect = cms.string('oracle://cms_orcoff_prep/CMS_COND_CSC')
    #connect = cms.string('sqlite_file:/afs/cern.ch/user/d/deisher/CMSSW_3_7_0_pre3/src/OnlineDB/CSCCondDB/test/CSCChamberTimeCorrections_before_133826.db')
     connect = cms.string('sqlite_file:/afs/cern.ch/user/d/deisher/public/test_anode_offsets_mc.db')                                 
    #connect = cms.string('sqlite_file:/afs/cern.ch/user/d/deisher/CMSSW_3_7_0_pre3/src/CalibMuon/CSCCalibration/test/test_chipCorr_and_chamberCorr_after_133826.db')
                                   )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)
process.source = cms.Source("EmptySource",
                          firstRun = cms.untracked.uint32(150000))


process.prod1 = cms.EDAnalyzer("CSCReadChamberTimeCorrAnalyzer")

process.printEventNumber = cms.OutputModule("AsciiOutputModule")

process.p = cms.Path(process.prod1)
process.ep = cms.EndPath(process.printEventNumber)


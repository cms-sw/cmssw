
import FWCore.ParameterSet.Config as cms

from datetime import date

processName = "writeMVAs"
print "<%s>:" % processName
today = date.today()
today_string = "%s%s%s" % (today.strftime("%Y"), today.strftime("%B")[0:3], today.strftime("%d"))
print " date = %s" % today_string

process = cms.Process(processName)

process.maxEvents = cms.untracked.PSet(            
    input = cms.untracked.int32(1) # CV: needs to be set to 1 so that GBRForestWriter::analyze method gets called exactly once         
)

process.source = cms.Source("EmptySource")

process.load('Configuration/StandardSequences/Services_cff')

#--------------------------------------------------------------------------------
# enable/disable update of anti-electron discriminator MVA training
updateAntiElectronDiscrMVA = False

# enable/disable update of anti-muon discriminator MVA training
updateAntiMuonDiscrMVA = True

# enable/disable update of tau ID (= isolation+lifetime) discriminator MVA training
updateTauIdDiscrMVA = True
#--------------------------------------------------------------------------------

process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:RecoTauTag_MVAs_%s.db' % today_string

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    timetype = cms.untracked.string('runnumber'),
    toPut = cms.VPSet()
)                                          

process.writeMVAsSequence = cms.Sequence()

if updateAntiElectronDiscrMVA:
    process.load('RecoTauTag/TauTagTools/writeAntiElectronDiscrMVA_cfi')
    for category in process.writeAntiElectronDiscrMVAs.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = category.outputRecord,
                tag = category.outputRecord
            )
        )
    for WP in process.writeAntiElectronDiscrWPs.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = WP.outputRecord,
                tag = WP.outputRecord
            )
        )        
    process.writeMVAsSequence += process.writeAntiElectronDiscrSequence    
if updateAntiMuonDiscrMVA:
    process.load('RecoTauTag/TauTagTools/writeAntiMuonDiscrMVA_cfi')
    for training in process.writeAntiMuonDiscrMVAs.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = training.outputRecord,
                tag = training.outputRecord
            )
        )
    for WP in process.writeAntiMuonDiscrWPs.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = WP.outputRecord,
                tag = WP.outputRecord
            )
        )
    for WP in process.writeAntiMuonDiscrMVAoutputNormalizations.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = WP.outputRecord,
                tag = WP.outputRecord
            )
        )
    process.writeMVAsSequence += process.writeAntiMuonDiscrSequence
if updateTauIdDiscrMVA:
    process.load('RecoTauTag/TauTagTools/writeTauIdDiscrMVA_cfi')
    for training in process.writeTauIdDiscrMVAs.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = training.outputRecord,
                tag = training.outputRecord
            )
        )
    for WP in process.writeTauIdDiscrWPs.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = WP.outputRecord,
                tag = WP.outputRecord
            )
        )
    for WP in process.writeTauIdDiscrMVAoutputNormalizations.jobs:
        process.PoolDBOutputService.toPut.append(
            cms.PSet(
                record = WP.outputRecord,
                tag = WP.outputRecord
            )
        )
    process.writeMVAsSequence += process.writeTauIdDiscrSequence

##print "PoolDBOutputService:"
##print process.PoolDBOutputService
    
process.p = cms.Path(process.writeMVAsSequence)

import FWCore.ParameterSet.Config as cms

EDMtoMEConverter = cms.EDProducer("EDMtoMEConverter",
    Name = cms.untracked.string('EDMtoMEConverter'),
    Verbosity = cms.untracked.int32(0), # 0 provides no output
                                        # 1 provides basic output
    Frequency = cms.untracked.int32(50),
    convertOnEndLumi = cms.untracked.bool(True),
    convertOnEndRun = cms.untracked.bool(True),
    # convert everything into JOB histograms so endJob harvesting works.
    reScope = cms.untracked.string("JOB"),
    runInputTag = cms.InputTag('MEtoEDMConverter', 'MEtoEDMConverterRun'),
    lumiInputTag = cms.InputTag('MEtoEDMConverter', 'MEtoEDMConverterLumi')
)

import FWCore.ParameterSet.Config as cms

process = cms.Process("EwkZMuMuGolden")

process.load("ElectroWeakAnalysis.ZMuMu.ZMuMuGolden_cfi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.threshold = ''
process.MessageLogger.cerr.FwkReport.reportEvery = 1
#process.load("Configuration.StandardSequences.Geometry_cff")
#process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = cms.string('')
#process.load("Configuration.StandardSequences.MagneticField_cff")


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("ElectroWeakAnalysis.ZMuMu.2010data_cfi")

#process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring(
#    
#'rfio:/dpm/na.infn.it/home/cms/store/user/degruttola/2010/HLTMu9_run2010B_Nov4ReReco/testZMuMuSubskim_9_1_fQk.root',


#'file:/scratch2/users/fabozzi/spring10/powheg_zmm_cteq66/1CD6D0A6-1E64-DF11-BB60-001D09FD0D10.root',',
   # 'rfio:/castor/cern.ch/cms/store/relval/CMSSW_3_4_0_pre1/RelValZMM/GEN-SIM-RECO/STARTUP31X_V8-v1/0007/CAE2081C-48B5-DE11-9161-001D09F29321.root',',
#    )
#)


process.poolDBESSource = cms.ESSource("PoolDBESSource",
                                      BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
                                      DBParameters = cms.PSet(
    messageLevel = cms.untracked.int32(2),
    authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb/')
    ),
                                      timetype = cms.untracked.string('runnumber'),
                                      connect = cms.string('frontier://FrontierProd/CMS_COND_31X_PHYSICSTOOLS'),
                                      toGet = cms.VPSet(cms.PSet(
    record = cms.string('MuScleFitDBobjectRcd'),
    tag = cms.string('MuScleFit_Scale_Z_36_invPb_innerTrack_Dec22_v1')
    ))
                                      )

process.MuScleFitMuonProducer = cms.EDProducer(
        'MuScleFitMuonProducer',
            MuonLabel = cms.InputTag("muons"),
            DbObjectLabel = cms.untracked.string(""),
            PatMuons = cms.bool(False)
        )

process.goodGlobalMuons.src =cms.InputTag("MuScleFitMuonProducer")
                                

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("ewkZMuMuGolden_corr.root")
)

zPlots = cms.PSet(
    histograms = cms.VPSet(
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("zMass"),
    description = cms.untracked.string("Z mass [GeV/c^{2}]"),
    plotquantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu1Pt"),
    description = cms.untracked.string("Highest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("max(daughter(0).pt,daughter(1).pt)")
    ),
    cms.PSet(
    min = cms.untracked.double(0.0),
    max = cms.untracked.double(200.0),
    nbins = cms.untracked.int32(200),
    name = cms.untracked.string("mu2Pt"),
    description = cms.untracked.string("Lowest muon p_{t} [GeV/c]"),
    plotquantity = cms.untracked.string("min(daughter(0).pt,daughter(1).pt)")
    )
    )
)

process.goodZToMuMuNtuple = cms.EDProducer(
    "CandViewNtpProducer",
    src = cms.InputTag("zmmCands"),
    variables = cms.VPSet(
    cms.PSet(
    tag = cms.untracked.string("mass"),
    quantity = cms.untracked.string("mass")
    ),
    cms.PSet(
    tag = cms.untracked.string("pt"),
    quantity = cms.untracked.string("pt")
    ),
    cms.PSet(
    tag = cms.untracked.string("muMaxPt"),
    quantity = cms.untracked.string("max(daughter(0).pt,daughter(1).pt)")
    ),
    cms.PSet(
    tag = cms.untracked.string("muMinPt"),
    quantity = cms.untracked.string("min(daughter(0).pt,daughter(1).pt)")
    ),
    )
    )
        



process.goodZToMuMuPlots = cms.EDAnalyzer(
    "CandViewHistoAnalyzer",
    zPlots,
#    src = cms.InputTag("goodZToMuMuAtLeast1HLT"),
    src = cms.InputTag("zmmCands"),
    filter = cms.bool(False)
)




process.eventInfo = cms.OutputModule (
    "AsciiOutputModule"
)

process.ewkZMuMuGoldenPath = cms.Path(
    process.MuScleFitMuonProducer *
    process.ewkZMuMuGoldenSequence *
    process.goodZToMuMuPlots *
    process.goodZToMuMuNtuple
)

process.out = cms.OutputModule(
        "PoolOutputModule",
            fileName = cms.untracked.string('Ntuple_test_corr.root'),
        outputCommands = cms.untracked.vstring(
    "drop *",
                "keep *_goodZToMuMuNtuple_*_*",
    ),
            SelectEvents = cms.untracked.PSet(
          SelectEvents = cms.vstring(
            "ewkZMuMuGoldenPath",
            )
          )
        )

process.endPath = cms.EndPath(
    process.eventInfo
    + process.out
    )



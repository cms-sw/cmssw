import FWCore.ParameterSet.Config as cms

process = cms.Process("Analyzer")
process.load("FWCore.MessageService.MessageLogger_cfi")

# process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')

process.maxEvents = cms.untracked.PSet( 
    input           = cms.untracked.int32(-1),
    # eventsToProcess = cms.untracked.VEventRange('1:26043:618522'), # Run 1 LS 26043 Evt 618522
    # eventsToProcess = cms.untracked.VEventRange('1:26044:618528'), # Run 1 LS 26044 Evt 618528
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # 'file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_116.root'
        'file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_100.root'
    )
)

process.me0timeanalyzer = cms.EDAnalyzer('MyME0InTimePUAnalyzer',
                              # ----------------------------------------------------------------------
                              # RootFileName = cms.untracked.string("TestGEMSegmentHistograms.root"),
                              # ----------------------------------------------------------------------
                              preDigiSmearX   = cms.untracked.double(0.05), # [in cm]
                              preDigiSmearY   = cms.untracked.double(1.0),  # [in cm]
                              nMatchedHits    = cms.untracked.int32(3),
                              # ----------------------------------------------------------------------
                              printInfoHepMC    = cms.untracked.bool(False),
                              printInfoSignal   = cms.untracked.bool(True),
                              printInfoPU       = cms.untracked.bool(False),
                              printInfoAll      = cms.untracked.bool(False),
                              printInfoME0Match = cms.untracked.bool(False),
                              # ----------------------------------------------------------------------

)

process.p = cms.Path(process.me0timeanalyzer)

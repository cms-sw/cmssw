import FWCore.ParameterSet.Config as cms

process = cms.Process("Analyzer")
process.load("FWCore.MessageService.MessageLogger_cfi")

# process.load('Configuration.Geometry.GeometryExtended2015MuonGEMDevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023HGCalMuonReco_cff')
# CSCGeometry depends on alignment ==> necessary to provide GlobalPositionRecord
process.load("Alignment.CommonAlignmentProducer.FakeAlignmentSource_cfi") 
# process.load("Geometry.CSCGeometry.cscGeometry_cfi")

# process.load("Geometry.CSCGeometry.cscGeometry_cfi")
# process.load("Geometry.DTGeometry.dtGeometry_cfi")
# process.load("Geometry.GEMGeometry.gemGeometry_cfi")


process.maxEvents = cms.untracked.PSet( 
    input           = cms.untracked.int32(-1),
    # eventsToProcess = cms.untracked.VEventRange('1:26043:618522'), # Run 1 LS 26043 Evt 618522
    # eventsToProcess = cms.untracked.VEventRange('1:26044:618528'), # Run 1 LS 26044 Evt 618528
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        # 'file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_116.root'
        # 'file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_100.root'
        # 'file:DYToMuMu_M-20_HGCALGS_PU140_ME0_RECO_100ps_amandeep_100_v2.root'
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_2_1_vGt.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_24_1_GJ6.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_18_1_uME.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_5_1_799.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_9_1_ijp.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_40_1_sCu.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_35_1_VJM.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_21_1_H5S.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_30_1_2Xy.root',
        '/store/user/piet/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola/DYToMuMu_M-20_TuneZ2star_14TeV-pythia6-tauola_HGCALGS_PU140_100ps_1p5ns_RECO/a6c1ab73bd1959e4a7fbbca874362562/out_reco_3_1_6rT.root',
    )
)

process.me0timeanalyzer = cms.EDAnalyzer('MyME0InTimePUAnalyzer',
                              # ----------------------------------------------------------------------
                              RootFileName       = cms.untracked.string("ME0InTimeOutOfTimePUtHistograms.root"),
                              # ----------------------------------------------------------------------
                              preDigiSmearX      = cms.untracked.double(0.0500), # [in cm] (single layer resolution)
                              preDigiSmearY      = cms.untracked.double(1.0),    # [in cm] (single layer resolution)
                              cscDetResX         = cms.untracked.double(0.0150), # [in cm] (chamber resolution :: 75-150um, take here 150um)
                              cscDetResY         = cms.untracked.double(5.0),    # [in cm]
                              dtDetResX          = cms.untracked.double(0.0125), # [in cm] (chamber resolution ::  75-125um in r-phi, take here 125um)
                              dtDetResY          = cms.untracked.double(0.0400), # [in cm] (chamber resolution :: 150-400um in r-z  , take here 400um)
                              nMatchedHitsME0Seg = cms.untracked.int32(3),
                              nMatchedHitsCSCSeg = cms.untracked.int32(3),
                              nMatchedHitsDTSeg  = cms.untracked.int32(6),
                              # ----------------------------------------------------------------------
                              printInfoHepMC     = cms.untracked.bool(False),
                              printInfoSignal    = cms.untracked.bool(False),
                              printInfoPU        = cms.untracked.bool(False),
                              printInfoAll       = cms.untracked.bool(False),
                              printInfoME0Match  = cms.untracked.bool(False),
                              printInfoMuonMatch = cms.untracked.bool(False),
                              # ----------------------------------------------------------------------

)

process.p = cms.Path(process.me0timeanalyzer)

import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.common_cff import *
from DPGAnalysis.MuonTools.common_cff import *

from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

from PhysicsTools.PatAlgos.producersLayer1.muonProducer_cfi import *

muonFlatTableProducer = simpleCandidateFlatTableProducer.clone(
    src = cms.InputTag("patMuons"),
    name = cms.string("muon"),
    doc  = cms.string("RECO muon information"),

    variables = cms.PSet(CandVars, # CB is this including eta
        isGlobal = Var("isGlobalMuon", bool, doc="muon is a global muon"),
        isTracker = Var("isTrackerMuon", bool, doc="muon is a tracker muon"),
        # isTrackerArb = Var("muon::isGoodMuon(muon, muon::TrackerMuonArbitrated)", bool, doc="muon is tracker muon arbitrated"),
        isStandalone = Var("isStandAloneMuon", bool, doc="muon is a standalone muon"),
        isRPC = Var("isRPCMuon", bool, doc="muon is an RPC muon"),
        isGEM = Var("isGEMMuon", bool, doc="muon is a GEM muon"),
        
        isLoose = Var("passed('CutBasedIdLoose')", bool, doc="Loose muon ID"),
        isMedium = Var("passed('CutBasedIdMedium')", bool, doc="Medium muon ID"),
        isTight = Var("passed('CutBasedIdTight')", bool, doc="Tight muon ID"),

        pfIso04 = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt", float, doc="relative PF-isolation (delta beta corrected, 0.4 cone)", precision=6),
        trkIso03 = Var("isolationR03().sumPt/tunePMuonBestTrack().pt", float, doc="relative tracker isolation (0.3 cone)",  precision=6),        
        
        trk_dz = Var(f"?!innerTrack().isNull()? dB('PVDZ') : {defaults.FLOAT}", float, doc="dz (with sign) wrt first PV - cm", precision=10),
        trk_dxy = Var(f"?!innerTrack().isNull()? dB('PV2D') : {defaults.FLOAT}", float, doc="dxy (with sign) wrt first PV - cm", precision=10),

        trk_algo = Var(f"?!innerTrack().isNull()? innerTrack().algo() : {defaults.INT_POS}", "int8", doc="iterative tracking algorithm used to build the inner track"),
        trk_origAlgo = Var(f"?!innerTrack().isNull()? innerTrack().originalAlgo() : {defaults.INT_POS}", "int8", doc="original (pre muon iterations) iterative tracking algorithm used to build the inner track"),
        
        trk_numberOfValidPixelHits = Var(f"?!innerTrack().isNull()? innerTrack().hitPattern().numberOfValidPixelHits() : {defaults.INT_POS}", "int8", doc="number of valid pixel hits"),
        trk_numberOfValidTrackerLayers = Var(f"?!innerTrack().isNull()? innerTrack().hitPattern().trackerLayersWithMeasurement() : {defaults.INT_POS}", "int8", doc="number of valid tracker layers"),
        trk_validFraction = Var(f"?!innerTrack().isNull()? innerTrack().validFraction() : {defaults.FLOAT_POS}", "int8", doc="fraction of tracker layer with muon hits"),
        
        trkMu_stationMask = Var("stationMask()", "uint8", doc="bit map of stations with tracks within given distance (in cm) of chamber edges"),
        trkMu_numberOfMatchedStations = Var("numberOfMatchedStations()", "int8", doc="number of matched DT/CSC stations"),
        rpcMu_numberOfMatchedRPCLayers = Var("numberOfMatchedRPCLayers()", "int8", doc="number of matched RPC layers"),
        
        staMu_numberOfValidMuonHits = Var(f"?isStandAloneMuon()? outerTrack().hitPattern().numberOfValidMuonHits() : {defaults.INT_POS}", "int8", doc="Number of valid muon hits"),

        staMu_normChi2 = Var(f"?isStandAloneMuon()? outerTrack().chi2()/outerTrack().ndof() : {defaults.FLOAT_POS}", float, doc="chi2/ndof (standalone track)", precision=10),
        glbMu_normChi2 = Var(f"?isGlobalMuon()? globalTrack().chi2()/globalTrack().ndof() : {defaults.FLOAT_POS}", float, doc="chi2/ndof (global track)", precision=10)
        )
)

from DPGAnalysis.MuonTools.muDTMuonExtTableProducer_cfi import muDTMuonExtTableProducer

from RecoMuon.TrackingTools.MuonServiceProxy_cff import MuonServiceProxy

from DPGAnalysis.MuonTools.muGEMMuonExtTableProducer_cfi import muGEMMuonExtTableProducer
muGEMMuonExtTableProducer.ServiceParameters = MuonServiceProxy.ServiceParameters

from DPGAnalysis.MuonTools.muCSCTnPFlatTableProducer_cfi import muCSCTnPFlatTableProducer
muCSCTnPFlatTableProducer.ServiceParameters = MuonServiceProxy.ServiceParameters

muRecoProducers = cms.Sequence(patMuons
                               + muonFlatTableProducer
                               + muDTMuonExtTableProducer
                               + muGEMMuonExtTableProducer
                               + muCSCTnPFlatTableProducer
                              )

import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *

from PhysicsTools.NanoAOD.common_cff import Var, P3Vars
from PhysicsTools.NanoAOD.triggerObjects_cff import mksel

def Custom_Muon_Task(process):
    process.nanoTableTaskCommon.remove(process.electronTablesTask)
    process.nanoTableTaskCommon.remove(process.lowPtElectronTablesTask)
    process.nanoTableTaskCommon.remove(process.photonTablesTask)
    process.nanoTableTaskCommon.remove(process.metTablesTask)
    process.nanoTableTaskCommon.remove(process.tauTablesTask)
    process.nanoTableTaskCommon.remove(process.boostedTauTablesTask)
    process.nanoTableTaskCommon.remove(process.jetPuppiTablesTask)
    process.nanoTableTaskCommon.remove(process.jetAK8TablesTask)

    process.nanoTableTaskFS.remove(process.electronMCTask)
    process.nanoTableTaskFS.remove(process.lowPtElectronMCTask)
    process.nanoTableTaskFS.remove(process.photonMCTask)
    process.nanoTableTaskFS.remove(process.jetMCTask)
    process.nanoTableTaskFS.remove(process.tauMCTask)
    process.nanoTableTaskFS.remove(process.boostedTauMCTask)
    process.nanoTableTaskFS.remove(process.metMCTable)
    process.nanoTableTaskFS.remove(process.ttbarCatMCProducersTask)
    process.nanoTableTaskFS.remove(process.ttbarCategoryTableTask)
    
    return process

def AddPFTracks(proc):
    pfTracks = "pfTracks"
    setattr(proc, pfTracks, cms.EDProducer("pfTracksProducer",
                    PFCands=cms.InputTag("packedPFCandidates"),
                    lostTracks=cms.InputTag("lostTracks"),
                    TrkHPurity = cms.bool(False),
                    trkSelection = cms.string("bestTrack.pt()>5 && abs(bestTrack.eta())<2.4 "),
                   )
    )
    
    pfTracksTable = "pfTracksTable"
    setattr(proc, pfTracksTable, cms.EDProducer("SimpleTrackFlatTableProducer",
                        src = cms.InputTag("pfTracks"),
                        cut = cms.string("pt > 15"), # filtered already above
                        name = cms.string("Track"),
                        doc  = cms.string("General tracks with pt > 15 GeV"),
                        singleton = cms.bool(False), # the number of entries is variable
                        extension = cms.bool(False), # this is the main table for the muons
                        variables = cms.PSet(P3Vars,
                            dz = Var("dz",float,doc="dz (with sign) wrt first PV, in cm",precision=10),
                            dxy = Var("dxy",float,doc="dxy (with sign) wrt first PV, in cm",precision=10),
                            charge = Var("charge", int, doc="electric charge"),
                            normChiSq = Var("normalizedChi2", float, precision=14, doc="Chi^2/ndof"),
                            numberOfValidHits = Var('numberOfValidHits()', 'int', precision=-1, doc='Number of valid hits in track'),
                            numberOfLostHits = Var('numberOfLostHits()', 'int', precision=-1, doc='Number of lost hits in track'),
                            trackAlgo = Var('algo()', 'int', precision=-1, doc='Track algo enum, check DataFormats/TrackReco/interface/TrackBase.h for details.'),
                            trackOriginalAlgo = Var('originalAlgo()', 'int', precision=-1, doc='Track original algo enum'),
                            qualityMask = Var('qualityMask()', 'int', precision=-1, doc='Quality mask of the track.'),
                            extraIdx = Var('extra().key()', 'int', precision=-1, doc='Index of the TrackExtra in the original collection'),
                            vx = Var('vx', 'float', precision=-1, doc='Track X position'),
                            vy = Var('vy', 'float', precision=-1, doc='Track Y position'),
                            vz = Var('vz', 'float', precision=-1, doc='Track Z position'),
                           ),
                 )
    )
    
    pfTracksTask = "pfTracksTask"
    setattr(proc,pfTracksTask, cms.Task(
        getattr(proc,pfTracks)
       )
    )
  
    pfTracksTableTask = "pfTracksTableTask"
    setattr(proc,pfTracksTableTask, cms.Task(
        getattr(proc,pfTracksTable)
      ) 
    )
    proc.nanoTableTaskCommon.add(getattr(proc,pfTracksTask))
    proc.nanoTableTaskCommon.add(getattr(proc,pfTracksTableTask))
  
    return proc

    

def AddVariablesForMuon(proc):
    
    muonWithVariables = "muonWithVariables"
    setattr(proc, muonWithVariables, cms.EDProducer("MuonSpecialVariables",
                    muonSrc=cms.InputTag("slimmedMuons"),
                    vertexSrc=cms.InputTag("offlineSlimmedPrimaryVertices"),
                    trkSrc=cms.InputTag("pfTracks"),
                    )
    )
    getattr(proc,"muonTask").add(getattr(proc,muonWithVariables))
    
    proc.slimmedMuonsUpdated.src = cms.InputTag("muonWithVariables")
    #proc.muonMVATTH.src = cms.InputTag("muonWithVariables") 
    #proc.muonMVALowPt.src = cms.InputTag("muonWithVariables") 
    #proc.muonTable.src = cms.InputTag("muonWithVariables") 
    #proc.muonMCTable.src = cms.InputTag("muonWithVariables") 
    #proc.muonsMCMatchForTable.src = cms.InputTag("muonWithVariables") 
  
 
    #SandAlone Variables
    proc.muonTable.variables.standalonePt = Var("? standAloneMuon().isNonnull() ? standAloneMuon().pt() : -1", float, doc = "pt of the standalone muon", precision=14)
    proc.muonTable.variables.standaloneEta = Var("? standAloneMuon().isNonnull() ? standAloneMuon().eta() : -99", float, doc = "eta of the standalone muon", precision=14)
    proc.muonTable.variables.standalonePhi = Var("? standAloneMuon().isNonnull() ? standAloneMuon().phi() : -99", float, doc = "phi of the standalone muon", precision=14)
    proc.muonTable.variables.standaloneCharge = Var("? standAloneMuon().isNonnull() ? standAloneMuon().charge() : -99", float, doc = "phi of the standalone muon", precision=14)
    
    # Inner Track Algo variables
    proc.muonTable.variables.innerTrackAlgo = Var('? innerTrack().isNonnull() ? innerTrack().algo() : -99', 'int', precision=-1, doc='Track algo enum, check DataFormats/TrackReco/interface/TrackBase.h for details.')
    proc.muonTable.variables.innerTrackOriginalAlgo = Var('? innerTrack().isNonnull() ? innerTrack().originalAlgo() : -99', 'int', precision=-1, doc='Track original algo enum')

    #Spark Tool Iso 03 variables
    proc.muonTable.variables.pfAbsIso03_neu = Var("pfIsolationR03().sumNeutralHadronEt",float,doc="PF absolute isolation dR=0.3, neutral component")
    proc.muonTable.variables.pfAbsIso03_pho = Var("pfIsolationR03().sumPhotonEt",float,doc="PF absolute isolation dR=0.3, photon component")
    proc.muonTable.variables.pfAbsIso03_sumPU = Var("pfIsolationR03().sumPUPt",float,doc="PF absolute isolation dR=0.3, pu component (no deltaBeta corrections)")
    proc.muonTable.variables.absTrkIso03 = Var("userFloat('absTrkiso03')",float,doc="Realtive Tracker Iso with cone size 0.3")
    
    # Spark Tool Iso 04 variables
    proc.muonTable.variables.pfAbsIso04_chg = Var("pfIsolationR04().sumChargedHadronPt",float,doc="PF absolute isolation dR=0.4, charged component")
    proc.muonTable.variables.pfAbsIso04_neu = Var("pfIsolationR04().sumNeutralHadronEt",float,doc="PF absolute isolation dR=0.4, neutral component")
    proc.muonTable.variables.pfAbsIso04_pho = Var("pfIsolationR04().sumPhotonEt",float,doc="PF absolute isolation dR=0.4, photon component")
    proc.muonTable.variables.pfAbsIso04_sumPU = Var("pfIsolationR04().sumPUPt",float,doc="PF absolute isolation dR=0.4, pu component (no deltaBeta corrections)")
    proc.muonTable.variables.absTrkIso04 = Var("userFloat('absTrkiso04')",float,doc="Realtive Tracker Iso with cone size 0.4")

    #Mini PF Isolation
    proc.muonTable.variables.miniPFAbsIso_chg = Var("userFloat('miniIsoChg')",float,doc="mini PF absolute isolation, charged component")
    proc.muonTable.variables.miniPFAbsIso_all = Var("userFloat('miniIsoAll')",float,doc="mini PF absolute isolation, total (with scaled rho*EA PU corrections)")
    proc.muonTable.variables.miniPFAbsIso_neu = Var("miniPFIsolation().neutralHadronIso()",float,doc="mini PF absolute isolation, neutral component")
    proc.muonTable.variables.miniPFAbsIso_pho = Var("miniPFIsolation().photonIso()", float, doc="mini PF absolute isolation, photon component")
    
    # Absolute Isolations for variables already present in Standard NanoAOD as Relative Isolation
    proc.muonTable.variables.tkAbsIso = Var("isolationR03().sumPt",float,doc="Tracker-based absolute isolation dR=0.3 for highPt, trkIso",precision=6)
    proc.muonTable.variables.pfAbsIso03_chg = Var("pfIsolationR03().sumChargedHadronPt",float,doc="PF absolute isolation dR=0.3, charged component")
    proc.muonTable.variables.pfAbsIso03_all = Var("(pfIsolationR03().sumChargedHadronPt + max(pfIsolationR03().sumNeutralHadronEt + pfIsolationR03().sumPhotonEt - pfIsolationR03().sumPUPt/2,0.0))",float,doc="PF absolute isolation dR=0.3, total (deltaBeta corrections)")
    proc.muonTable.variables.pfAbsIso04_all = Var("(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))",float,doc="PF absolute isolation dR=0.4, total (deltaBeta corrections)")
    proc.muonTable.variables.jetAbsIso = Var("?userCand('jetForLepJetVar').isNonnull()?(1./userFloat('ptRatio'))-1.:(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))",float,doc="Absolute isolation in matched jet (1/ptRatio-1, pfRelIso04_all if no matched jet)",precision=8)

    # Muon Quality Variables
    proc.muonTable.variables.expectedMatchedStations = Var("expectedNnumberOfMatchedStations()",int,doc="Expected Number of Matched stations")
    proc.muonTable.variables.RPCLayers = Var("numberOfMatchedRPCLayers()",int,doc="Number of RPC Layers")
    proc.muonTable.variables.stationMask = Var("stationMask()","uint8",doc="Number of masked station")
    proc.muonTable.variables.nShowers = Var("numberOfShowers()",int,doc="Number of Showers")
    ##
   

    # Hits related variables 
    proc.muonTable.variables.pixelLayers = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().pixelLayersWithMeasurement() : -99", float,doc="Number of Pixel Layers") # No of tracker layers are already saved in the standard NanoAODs
    proc.muonTable.variables.pixelHits = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().numberOfValidPixelHits() : -99", float, doc="Numbr of valid pixel hits")
    proc.muonTable.variables.muonStations = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().muonStationsWithValidHits() : -99", float, doc="No of valid hits in muon stations")
    proc.muonTable.variables.DTHits = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().numberOfValidMuonDTHits() : -99", float, doc="No of valid hits in DT")
    proc.muonTable.variables.CSCHits = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().numberOfValidMuonCSCHits() : -99", float, doc="No of valid hits in CSC")
    proc.muonTable.variables.RPCHits = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().numberOfValidMuonRPCHits() : -99", float, doc="No of valid hits in RPC")
    
    
    
    # Chi2 related to different tracks
    proc.muonTable.variables.trkChi2 = Var("? globalTrack().isNonnull() ? globalTrack().normalizedChi2() : ? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().normalizedChi2() : -99",float,doc="Normalized Chi Square from either globalTrack or innerTrack ")
    proc.muonTable.variables.trkChi2_outerTrack = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().normalizedChi2() : -99",float,doc="Normalized Chi Square from outerTrack ")
    proc.muonTable.variables.trkChi2_innerTrack = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().normalizedChi2() : -99",float,doc="Normalized Chi Square from outerTrack ")
   

    #Inner Track related variables
    proc.muonTable.variables.innerTrack_ptErr = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().ptError()/innerTrack().pt() : -99", float, doc="InnerTrack Pt Error")
    proc.muonTable.variables.innerTrack_pt = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().pt(): -99", float, doc="InnerTrack Pt")
    proc.muonTable.variables.innerTrack_eta = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().eta(): -99", float, doc="InnerrTrack Eta")
    proc.muonTable.variables.innerTrack_phi = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().phi(): -99", float, doc="InnerTrack Phi")
    proc.muonTable.variables.innerTrack_charge = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().charge(): -99", float, doc="InnerTrack charge")
    proc.muonTable.variables.innerTrack_MuonHits = Var("? innerTrack().isNonnull() ? innerTrack().hitPattern().numberOfValidMuonHits() : -99", float, doc = "Number of valid Muon Hits from InnerTrack")
    proc.muonTable.variables.innerTrack_validFraction = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().validFraction() : -99", float, doc="Inner Track Valid Fraction")
    ## Pixellayers and PixelHits are already defined in the section related to the hits
    ## TrackerLayers are already defined in standard NanoAOD
    ## Chi2 is already defined in the Chi2 related section
    
    #Dxy Dz variables as of Spark tool
    proc.muonTable.variables.innerTrackDxy = Var("? userInt('isGoodVertex') ? userFloat('innerTrackDxy') : -99.9",float,doc = "dxy from Primary Vertex calculated with Inner Track")
    proc.muonTable.variables.innerTrackDz = Var("? userInt('isGoodVertex') ? userFloat('innerTrackDz') : -99.9",float,doc= "dz from Primary Vertex calculated with Inner Track")


    # Outer Track related variables
    proc.muonTable.variables.outerTrack_ptErr = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().ptError()/outerTrack().pt() : -99", float, doc="OuterTrack Pt Error")
    proc.muonTable.variables.outerTrack_pt = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().pt(): -99", float, doc="OuterTrack Pt")
    proc.muonTable.variables.outerTrack_eta = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().eta(): -99", float, doc="OuterTrack Eta")
    proc.muonTable.variables.outerTrack_phi = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().phi(): -99", float, doc="OuterTrack Phi")
    proc.muonTable.variables.outerTrack_charge = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().charge(): -99", float, doc="OuterTrack charge")
    proc.muonTable.variables.outerTrack_MuonHits = Var("? outerTrack().isNonnull() ? outerTrack().hitPattern().numberOfValidMuonHits() : -99", float, doc = "Number of valid Muon Hits from OuterTrack")
    ## Muonstations, DTHits and CSCHits are already defined in the Hits related section
    ## Chi2 is already defined in the Chi2 related section

    #Global track realted variables
    proc.muonTable.variables.muonHits = Var("? globalTrack().isNonnull() ? globalTrack().hitPattern().numberOfValidMuonHits() : ?  innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().numberOfValidMuonHits() :-99",float,doc="Number of valid Muon Hits from either globalTrack or innerTrack")
    proc.muonTable.variables.globalTrack_ptErr = Var("? globalTrack().isNonnull() ? globalTrack().ptError()/globalTrack().pt() : -99", float, doc="GlobalTrack Pt Error")
    proc.muonTable.variables.globalTrack_pt = Var("? globalTrack().isNonnull() ? globalTrack().pt(): -99", float, doc="GlobalTrack Pt")
    proc.muonTable.variables.globalTrack_eta = Var("? globalTrack().isNonnull() ? globalTrack().eta(): -99", float, doc="GlobalTrack Eta")
    proc.muonTable.variables.globalTrack_phi = Var("? globalTrack().isNonnull() ? globalTrack().phi(): -99", float, doc="GlobalTrack Phi")
    proc.muonTable.variables.globalTrack_charge = Var("? globalTrack().isNonnull() ? globalTrack().charge(): -99", float, doc="GlobalTrack charge")

    #muonBestTrack related varaibles
    ## tightCharge from muonBestTrack is already saved in deafult nanoAOD
    proc.muonTable.variables.best_pt = Var("? muonBestTrack().isNonnull() && muonBestTrack().isAvailable() ? muonBestTrack().pt(): -99", float, doc="MuonBestTrack Pt")
    proc.muonTable.variables.best_pterr = Var("? muonBestTrack().isNonnull() && muonBestTrack().isAvailable() ? muonBestTrack().ptError() : -99", float, doc = "pTerr from MuonBestTrack")
    proc.muonTable.variables.best_eta = Var("? muonBestTrack().isNonnull() && muonBestTrack().isAvailable() ? muonBestTrack().eta(): -99", float, doc="MuonBestrack Eta")
    proc.muonTable.variables.best_phi = Var("? muonBestTrack().isNonnull() && muonBestTrack().isAvailable() ? muonBestTrack().phi(): -99", float, doc="MuonBestTrack Phi")
    proc.muonTable.variables.best_charge = Var("? muonBestTrack().isNonnull() && muonBestTrack().isAvailable() ? muonBestTrack().charge(): -99", float, doc="MuonBestTrack charge")

    # TuneP related variables
    proc.muonTable.variables.tuneP_pt = Var("? tunePMuonBestTrack().isNonnull() && tunePMuonBestTrack().isAvailable() ? tunePMuonBestTrack().pt() : -99", float, doc = "pT from tunePMuonBestTrack")
    # tuneP_ptErr is now moved to standard NanoAODs
    proc.muonTable.variables.tuneP_eta = Var("? tunePMuonBestTrack().isNonnull() && tunePMuonBestTrack().isAvailable() ? tunePMuonBestTrack().eta(): -99", float, doc="tunePMuonBestTrack Eta")
    proc.muonTable.variables.tuneP_phi = Var("? tunePMuonBestTrack().isNonnull() && tunePMuonBestTrack().isAvailable() ? tunePMuonBestTrack().phi(): -99", float, doc="tunePMuonBestTrack Phi")
    proc.muonTable.variables.tuneP_charge = Var("? tunePMuonBestTrack().isNonnull() && tunePMuonBestTrack().isAvailable() ? tunePMuonBestTrack().charge(): -99", float, doc="tunePMuonBestTrack() charge")
    proc.muonTable.variables.tuneP_muonHits = Var("? tunePMuonBestTrack().isNonnull() && tunePMuonBestTrack().isAvailable() ? tunePMuonBestTrack().hitPattern().numberOfValidMuonHits() : -99", int, doc="No of valid muon hists from tunePMuonBestTrack")

    # tpfms Track
    proc.muonTable.variables.tpfms_pt = Var("? tpfmsTrack().isNonnull() && tpfmsTrack().isAvailable() ? tpfmsTrack().pt() : -99", float, doc = "pT from tpfmsTrack")
    proc.muonTable.variables.tpfms_pterr = Var("? tpfmsTrack().isNonnull() && tpfmsTrack().isAvailable() ? tpfmsTrack().ptError() : -99", float, doc = "pTerr from tpfmsTrack")
    proc.muonTable.variables.tpfms_eta = Var("? tpfmsTrack().isNonnull() && tpfmsTrack().isAvailable() ? tpfmsTrack().eta(): -99", float, doc="tpfmsTrack Eta")
    proc.muonTable.variables.tpfms_phi = Var("? tpfmsTrack().isNonnull() && tpfmsTrack().isAvailable() ? tpfmsTrack().phi(): -99", float, doc="tpfmsTrack Phi")
    proc.muonTable.variables.tpfms_charge = Var("? tpfmsTrack().isNonnull() && tpfmsTrack().isAvailable() ? tpfmsTrack().charge(): -99", float, doc="tpfmsTrack charge")
    proc.muonTable.variables.tpfms_muonHits = Var("? tpfmsTrack().isNonnull() && tpfmsTrack().isAvailable() ? tpfmsTrack().hitPattern().numberOfValidMuonHits() : -99", int, doc="No of valid muon hists from tpfmsTrack")


    #picky Track
    proc.muonTable.variables.picky_pt = Var("? pickyTrack().isNonnull() && pickyTrack().isAvailable() ? pickyTrack().pt() : -99", float, doc = "pT from pickyTrack")
    proc.muonTable.variables.picky_pterr = Var("? pickyTrack().isNonnull() && pickyTrack().isAvailable() ? pickyTrack().ptError() : -99", float, doc = "pTerr from pickyTrack")
    proc.muonTable.variables.picky_eta = Var("? pickyTrack().isNonnull() && pickyTrack().isAvailable() ? pickyTrack().eta(): -99", float, doc="pickyTrack Eta")
    proc.muonTable.variables.picky_phi = Var("? pickyTrack().isNonnull() && pickyTrack().isAvailable() ? pickyTrack().phi(): -99", float, doc="pickyTrack Phi")
    proc.muonTable.variables.picky_charge = Var("? pickyTrack().isNonnull() && pickyTrack().isAvailable() ? pickyTrack().charge(): -99", float, doc="pickyTrack charge")
    proc.muonTable.variables.picky_muonHits = Var("? pickyTrack().isNonnull() && pickyTrack().isAvailable() ? pickyTrack().hitPattern().numberOfValidMuonHits() : -99", int, doc="No of valid muon hists from pickyTrack")
    
    #dyt Track
    proc.muonTable.variables.dyt_pt = Var("? dytTrack().isNonnull() && dytTrack().isAvailable() ? dytTrack().pt() : -99", float, doc = "pT from dytTrack")
    proc.muonTable.variables.dyt_pterr = Var("? dytTrack().isNonnull() && dytTrack().isAvailable() ? dytTrack().ptError() : -99", float, doc = "pTerr from dytTrack")
    proc.muonTable.variables.dyt_eta = Var("? dytTrack().isNonnull() && dytTrack().isAvailable() ? dytTrack().eta(): -99", float, doc="dytTrack Eta")
    proc.muonTable.variables.dyt_phi = Var("? dytTrack().isNonnull() && dytTrack().isAvailable() ? dytTrack().phi(): -99", float, doc="dytTrack Phi")
    proc.muonTable.variables.dyt_charge = Var("? dytTrack().isNonnull() && dytTrack().isAvailable() ? dytTrack().charge(): -99", float, doc="dytTrack charge")
    proc.muonTable.variables.dyt_muonHits = Var("? dytTrack().isNonnull() && dytTrack().isAvailable() ? dytTrack().hitPattern().numberOfValidMuonHits() : -99", int, doc="No of valid muon hists from dytTrack")  

    #CombinedQuality Variables
    proc.muonTable.variables.positionChi2 = Var("combinedQuality().chi2LocalPosition", float, doc="chi2 Local Position")
    proc.muonTable.variables.momentumChi2 = Var("combinedQuality().chi2LocalMomentum", float, doc="chi2 Local Momentum")
    proc.muonTable.variables.trkKink = Var("combinedQuality().trkKink", float, doc="Track Kink")
    proc.muonTable.variables.glbKink = Var("combinedQuality().glbKink", float, doc="Glb Kink")
    proc.muonTable.variables.glbTrackProbability = Var("combinedQuality().glbTrackProbability", float, doc="Glb Track Probability")
    proc.muonTable.variables.trkRelChi2 = Var("combinedQuality().trkRelChi2",float,doc="Track Rel Chi2")
    
    #timAtIpInOutErr
    proc.muonTable.variables.timAtIpInOutErr = Var("time().timeAtIpInOutErr",float,doc="timAtIpInOutErr")

    #isArbitratedTracker
    proc.muonTable.variables.isArbitratedTracker = Var("userInt('isArbitratedTracker')", bool, doc = "s Arbitrated Tracker")

    #ExtraidX
    proc.muonTable.variables.standaloneExtraIdx = Var('? standAloneMuon().isNonnull() ? standAloneMuon().extra().key() : -99', 'int', precision=-1, doc='Index of the StandAloneTrack TrackExtra in the original collection')
    proc.muonTable.variables.innerTrackExtraIdx = Var('? innerTrack().isNonnull() ? innerTrack().extra().key() : -99', 'int', precision=-1, doc='Index of the innerTrack TrackExtra in the original collection')

    #Jet Related Variables
    # lazyEval=True: userCand() returns the base type `reco::CandidatePtr`, needs to be dynamically casted to pat::Jet to call the userFloat() / bDiscriminator() methods
    proc.muonTable.variables.jetPtRatio = Var("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt)", float, doc="ptRatio using the LepAware JEC approach, for muon MVA", lazyEval=True)
    proc.muonTable.variables.jetDF = Var("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0",float,doc="b-tagging discriminator of the jet matched to the lepton, for muon MVA", lazyEval=True)
    proc.muonTable.variables.jetCSVv2 = Var("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfCombinedSecondaryVertexV2BJetTags'),0.0):0.0",float,doc="CSVv2 b-tagging discriminator of the jet matched to the lepton, for muon MVA", lazyEval=True)

    #nSegments
    proc.muonTable.variables.nsegments = Var("userInt('nsegments')", int, doc = "nsegments as of Spark-tool")

    #Sim Variables
    proc.muonTable.variables.simType = Var("? simType() ? simType() : -99",int,doc="simType")
    proc.muonTable.variables.simExtType = Var("? simExtType() ? simExtType() : -99",int,doc="simExtType")
    proc.muonTable.variables.simFlavour = Var("? simFlavour() ? simFlavour() : -99",int,doc="simFlavour")
    proc.muonTable.variables.simHeaviestMotherFlavour = Var(" ? simHeaviestMotherFlavour() ? simHeaviestMotherFlavour() : -99",int,doc="simHeaviestMotherFlavour")
    proc.muonTable.variables.simPdgId = Var("? simPdgId() ? simPdgId() : -99",int,doc="simPdgId")
    proc.muonTable.variables.simMotherPdgId = Var("? simMotherPdgId() ? simMotherPdgId() : -99",int,doc="simMotherPdgId")
    proc.muonTable.variables.simBX = Var("? simBX() ? simBX() : -99",int,doc="simBX")
    proc.muonTable.variables.simProdRho = Var("? simProdRho() ? simProdRho(): -99",float,doc="simProdRho")
    proc.muonTable.variables.simProdZ = Var("? simProdZ() ? simProdZ(): -99",float,doc="simProdZ")
    proc.muonTable.variables.simPt = Var("? simPt() ? simPt(): -99",float,doc="simPt")
    proc.muonTable.variables.simEta = Var("? simEta() ? simEta(): -99",float,doc="simEta")
    proc.muonTable.variables.simPhi = Var("? simPhi() ? simPhi(): -99",float,doc='simPhi')
   
    #ID Variables
    proc.muonTable.variables.isRPC = Var("isRPCMuon",bool,doc="muon is RPC muon") 
    
    return proc

def AddTriggerObjectBits(process): 
    process.triggerObjectTable.selections.Muon_POG = cms.PSet(
            id = cms.int32(1313),
            sel = cms.string("type(83) && pt > 5 && (coll('hltIterL3MuonCandidates') || (pt > 45 && coll('hltHighPtTkMuonCands')) || (pt > 95 && coll('hltOldL3MuonCandidates')))"),
            l1seed = cms.string("type(-81)"), l1deltaR = cms.double(0.5),
            l2seed = cms.string("type(83) && coll('hltL2MuonCandidates')"),  l2deltaR = cms.double(0.3),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.VPSet(
                mksel("filter('hltTripleMuonL2PreFiltered0')","hltTripleMuonL2PreFiltered0"), #0
                mksel("filter('hltTripleMuL3PreFiltered222')","hltTripleMuL3PreFiltered222"), #1
                mksel("filter('hltJpsiMuonL3Filtered3p5')","hltJpsiMuonL3Filtered3p5"), #2 
                mksel("filter('hltVertexmumuFilterJpsiMuon3p5')","hltVertexmumuFilterJpsiMuon3p5"), #3
                mksel("filter('hltL2fL1sDoubleMu0er15OSIorDoubleMu0er14OSIorDoubleMu4OSIorDoubleMu4p5OSL1Filtered0')","hltL2fL1sDoubleMu0er15OSIorDoubleMu0er14OSIorDoubleMu4OSIorDoubleMu4p5OSL1Filtered0"), #4
                mksel("filter('hltDoubleMu4JpsiDisplacedL3Filtered')","hltDoubleMu4JpsiDisplacedL3Filtered"), #5
                mksel("filter('hltDisplacedmumuFilterDoubleMu4Jpsi')","hltDisplacedmumuFilterDoubleMu4Jpsi"), #6
                mksel("filter('hltJpsiTkVertexFilter')","hltJpsiTkVertexFilter"), #7
                mksel("filter('hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered')","hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered"), #8
                mksel("filter('hltL3fL1DoubleMu155fPreFiltered8')","hltL3fL1DoubleMu155fPreFiltered8"), #9
                mksel("filter('hltL3fL1DoubleMu155fFiltered17')","hltL3fL1DoubleMu155fFiltered17"), #10
                mksel("filter('hltDiMuon178RelTrkIsoFiltered0p4')","hltDiMuon178RelTrkIsoFiltered0p4"), #11
                mksel("filter('hltDiMuon178RelTrkIsoFiltered0p4DzFiltered0p2')","hltDiMuon178RelTrkIsoFiltered0p4DzFiltered0p2"), #12
                mksel("filter('hltDiMuon178RelTrkIsoVVLFiltered')","hltDiMuon178RelTrkIsoVVLFiltered"), #13
                mksel("filter('hltDiMuon178RelTrkIsoVVLFilteredDzFiltered0p2')","hltDiMuon178RelTrkIsoVVLFilteredDzFiltered0p2"), #14
                mksel("filter('hltDiMuon178Mass3p8Filtered')","hltDiMuon178Mass3p8Filtered"), #15
                mksel("filter('hltL3fL1sMu22Or25L1f0L2f10QL3Filtered50Q')","hltL3fL1sMu22Or25L1f0L2f10QL3Filtered50Q"), #16
                mksel("filter('hltL2fOldL1sMu22or25L1f0L2Filtered10Q')","hltL2fOldL1sMu22or25L1f0L2Filtered10Q"), #17
                mksel("filter('hltL3fL1sMu22Or25L1f0L2f10QL3Filtered100Q')","hltL3fL1sMu22Or25L1f0L2f10QL3Filtered100Q"), #18
                mksel("filter('hltL3fL1sMu25f0TkFiltered100Q')","hltL3fL1sMu25f0TkFiltered100Q"), #19
                mksel("filter('hltL3fL1sMu15DQlqL1f0L2f10L3Filtered17')","hltL3fL1sMu15DQlqL1f0L2f10L3Filtered17"), #20
                mksel("filter('hltL3fL1sMu1lqL1f0L2f10L3Filtered17TkIsoFiltered0p4')","hltL3fL1sMu1lqL1f0L2f10L3Filtered17TkIsoFiltered0p4"), #21
                mksel("filter('hltL3fL1sMu1lqL1f0L2f10L3Filtered17TkIsoVVLFiltered')","hltL3fL1sMu1lqL1f0L2f10L3Filtered17TkIsoVVLFiltered"), #22
                mksel("filter('hltL3fL1sMu5L1f0L2f5L3Filtered8')","hltL3fL1sMu5L1f0L2f5L3Filtered8"), #23
                mksel("filter('hltL3fL1sMu5L1f0L2f5L3Filtered8TkIsoFiltered0p4')","hltL3fL1sMu5L1f0L2f5L3Filtered8TkIsoFiltered0p4"), #24
                mksel("filter('hltL3fL1sMu5L1f0L2f5L3Filtered8TkIsoVVLFiltered')","hltL3fL1sMu5L1f0L2f5L3Filtered8TkIsoVVLFiltered"), #25
                mksel("filter('hltL3fL1sMu15DQlqL1f0L2f10L3Filtered12')","hltL3fL1sMu15DQlqL1f0L2f10L3Filtered12"), #26
                mksel("filter('hltL3fL1sMu15DQlqL1f0L2f10L3Filtered15')","hltL3fL1sMu15DQlqL1f0L2f10L3Filtered15"), #27
                mksel("filter('hltL3fL1sMu15DQlqL1f0L2f10L3Filtered19')","hltL3fL1sMu15DQlqL1f0L2f10L3Filtered19"), #28
                mksel("filter('hltL3fL1sMu15DQlqL1f0L2f10L3Filtered19')","hltL3fL1sMu15DQlqL1f0L2f10L3Filtered19") #29
            )
        )

    process.triggerObjectTable.selections.Muon_POG_v2 = cms.PSet(
            id = cms.int32(131313),
            sel = cms.string("type(83) && pt > 5 && (coll('hltIterL3MuonCandidates') || (pt > 45 && coll('hltHighPtTkMuonCands')) || (pt > 95 && coll('hltOldL3MuonCandidates')))"),
            l1seed = cms.string("type(-81)"), l1deltaR = cms.double(0.5),
            l2seed = cms.string("type(83) && coll('hltL2MuonCandidates')"),  l2deltaR = cms.double(0.3),
            skipObjectsNotPassingQualityBits = cms.bool(True),
            qualityBits = cms.VPSet(
                mksel("filter('hltL3fL1sSingleMuOpenCandidateL1f0L2f3QL3Filtered50Q')","hltL3fL1sSingleMuOpenCandidateL1f0L2f3QL3Filtered50Q"), #0
                mksel("filter('hltTrk200MuonEndcapFilter')","hltTrk200MuonEndcapFilter") #1
            )
        )

    return process

def IncreaseGenPrecesion(process):
    
    process.genParticleTable.variables.pt = Var("pt",  float, precision=16)
    process.genParticleTable.variables.eta = Var("eta",  float,precision=16)
    process.genParticleTable.variables.phi = Var("phi", float,precision=16)
    
    return process

def PrepMuonCustomNanoAOD(process):
    
    process = Custom_Muon_Task(process)
    process = AddPFTracks(process)
    process = AddVariablesForMuon(process)
    process = AddTriggerObjectBits(process)
    process = IncreaseGenPrecesion(process)


    return process

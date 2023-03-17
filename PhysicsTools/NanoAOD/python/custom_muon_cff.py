import FWCore.ParameterSet.Config as cms

from PhysicsTools.NanoAOD.nano_eras_cff import *
from PhysicsTools.NanoAOD.simpleCandidateFlatTableProducer_cfi import simpleCandidateFlatTableProducer

from PhysicsTools.NanoAOD.common_cff import Var, P3Vars, P4Vars
from PhysicsTools.NanoAOD.muons_cff import muonTable, finalMuons

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
    
    # Spark Tool Iso 04 variables
    proc.muonTable.variables.pfAbsIso04_chg = Var("pfIsolationR04().sumChargedHadronPt",float,doc="PF absolute isolation dR=0.4, charged component")
    proc.muonTable.variables.pfAbsIso04_neu = Var("pfIsolationR04().sumNeutralHadronEt",float,doc="PF absolute isolation dR=0.4, neutral component")
    proc.muonTable.variables.pfAbsIso04_pho = Var("pfIsolationR04().sumPhotonEt",float,doc="PF absolute isolation dR=0.4, photon component")
    proc.muonTable.variables.pfAbsIso04_sumPU = Var("pfIsolationR04().sumPUPt",float,doc="PF absolute isolation dR=0.4, pu component (no deltaBeta corrections)")

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
    proc.muonTable.variables.relTrkiso4 = Var("userFloat('relTrkiso4')",float,doc="Realtive Tracker Iso with cone size 0.4")

    # Muon Quality Variables
    proc.muonTable.variables.expectedMatchedStations = Var("expectedNnumberOfMatchedStations()",int,doc="Expected Number of Matched stations")
    proc.muonTable.variables.RPCLayers = Var("numberOfMatchedRPCLayers()",int,doc="Number of RPC Layers")
    proc.muonTable.variables.stationMask = Var("stationMask()","uint8",doc="Number of masked station")
    proc.muonTable.variables.nShowers = Var("numberOfShowers()",int,doc="Number of Showers")
    proc.muonTable.variables.muonHits = Var("? globalTrack().isNonnull() ? globalTrack().hitPattern().numberOfValidMuonHits() : ?  innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().numberOfValidMuonHits() :-99",float,doc="Number of valid Muon Hits from either globalTrack or innerTrack")
    ## For completeness I save here also the muonHits for the outer tracker also
    proc.muonTable.variables.outerTrackMuonHits = Var("? outerTrack().isNonnull() ? outerTrack().hitPattern().numberOfValidMuonHits() : -99", float, doc = "Number of valid Muon Hits from OuterTrack")
    ##
    proc.muonTable.variables.pixelLayers = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().pixelLayersWithMeasurement() : -99", float,doc="Number of Pixel Layers") # No of tracker layers are already saved in the standard NanoAODs
    proc.muonTable.variables.validFraction = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().validFraction() : -99", float, doc="Inner Track Valid Fraction")
    proc.muonTable.variables.pixelHits = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().numberOfValidPixelHits() : -99", float, doc="Numbr of valid pixel hits")
    proc.muonTable.variables.muonStations = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().muonStationsWithValidHits() : -99", float, doc="No of valid hits in muon stations")
    proc.muonTable.variables.DTHits = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().numberOfValidMuonDTHits() : -99", float, doc="No of valid hits in DT")
    proc.muonTable.variables.CSCHits = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().numberOfValidMuonCSCHits() : -99", float, doc="No of valid hits in CSC")
    proc.muonTable.variables.RPCHits = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().hitPattern().numberOfValidMuonRPCHits() : -99", float, doc="No of valid hits in RPC")
    
    
    
    # Chi2 related to different tracks
    proc.muonTable.variables.trkChi2 = Var("? globalTrack().isNonnull() ? globalTrack().normalizedChi2() : ? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().normalizedChi2() : -99",float,doc="Normalized Chi Square from either globalTrack or innerTrack ")
    proc.muonTable.variables.trkChi2_outerTrack = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().normalizedChi2() : -99",float,doc="Normalized Chi Square from outerTrack ")
    proc.muonTable.variables.trkChi2_innerTrack = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().normalizedChi2() : -99",float,doc="Normalized Chi Square from outerTrack ")
   

    #pt, ptErr, eta, phi, charge for different tracks
    ## ptErr in standard NanoAOD are saved from bestTrack()
    ## For Spark tool it is needed from innerTrack. For completeness outerTrack
    ## variables are also saved
    proc.muonTable.variables.innerTrack_ptErr = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().ptError()/innerTrack().pt() : -99", float, doc="InnerTrack Pt Error")
    proc.muonTable.variables.outerTrack_ptErr = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().ptError()/outerTrack().pt() : -99", float, doc="OuterTrack Pt Error")
    proc.muonTable.variables.outerTrack_pt = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().pt(): -99", float, doc="OuterTrack Pt")
    proc.muonTable.variables.outerTrack_eta = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().eta(): -99", float, doc="OuterTrack Eta")
    proc.muonTable.variables.outerTrack_phi = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().phi(): -99", float, doc="OuterTrack Phi")
    proc.muonTable.variables.outerTrack_charge = Var("? outerTrack().isNonnull() && outerTrack().isAvailable() ? outerTrack().charge(): -99", float, doc="OuterTrack charge")
    proc.muonTable.variables.innerTrack_charge = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().charge(): -99", float, doc="OuterTrack charge")


    # TuneP related variables
    proc.muonTable.variables.tuneP_pt = Var("? tunePMuonBestTrack().isNonnull() ? tunePMuonBestTrack().pt() : -99", float, doc = "pT from tunePMuonBestTrack")
    proc.muonTable.variables.tuneP_pterr = Var("? tunePMuonBestTrack().isNonnull() ? tunePMuonBestTrack().ptError() : -99", float, doc = "pTerr from tunePMuonBestTrack")
    proc.muonTable.variables.tuneP_muonHits = Var("? tunePMuonBestTrack().isNonnull() ? tunePMuonBestTrack().hitPattern().numberOfValidMuonHits() : -99", int, doc="No of valid muon hists from tunePMuonBestTrack")
   

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
    proc.muonTable.variables.jetPtRatio = Var("?userCand('jetForLepJetVar').isNonnull()?min(userFloat('ptRatio'),1.5):1.0/(1.0+(pfIsolationR04().sumChargedHadronPt + max(pfIsolationR04().sumNeutralHadronEt + pfIsolationR04().sumPhotonEt - pfIsolationR04().sumPUPt/2,0.0))/pt)", float, doc="ptRatio using the LepAware JEC approach, for muon MVA")
    proc.muonTable.variables.jetDF = Var("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probbb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:probb')+userCand('jetForLepJetVar').bDiscriminator('pfDeepFlavourJetTags:problepb'),0.0):0.0",float,doc="b-tagging discriminator of the jet matched to the lepton, for muon MVA")
    proc.muonTable.variables.jetCSVv2 = Var("?userCand('jetForLepJetVar').isNonnull()?max(userCand('jetForLepJetVar').bDiscriminator('pfCombinedSecondaryVertexV2BJetTags'),0.0):0.0",float,doc="CSVv2 b-tagging discriminator of the jet matched to the lepton, for muon MVA")
    
    #Dxy Dz variables as of Spark tool
    proc.muonTable.variables.innerTrackDxy = Var("? userInt('isGoodVertex') ? userFloat('innerTrackDxy') : -99.9",float,doc = "dxy from Primary Vertex calculated with Inner Track")
    proc.muonTable.variables.innerTrackDz = Var("? userInt('isGoodVertex') ? userFloat('innerTrackDz') : -99.9",float,doc= "dz from Primary Vertex calculated with Inner Track")

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
    
    
    return proc


def PrepMuonCustomNanoAOD(process):
    
    process = Custom_Muon_Task(process)
    process = AddPFTracks(process)
    process = AddVariablesForMuon(process)


    return process

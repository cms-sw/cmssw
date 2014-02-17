import FWCore.ParameterSet.Config as cms

# $Id: TrigResRateMon_cfi.py,v 1.11 2011/10/05 18:29:44 lwming Exp $
trRateMon = cms.EDAnalyzer("TrigResRateMon",
    dirname = cms.untracked.string("HLT/TrigResults/"),
    muonRecoCollectionName = cms.untracked.string("muons"),
    plotAll = cms.untracked.bool(False),
    dRMax = cms.untracked.double(10.0),
		NbinsDR = cms.untracked.uint32(100),
    dRMaxElectronMuon = cms.untracked.double(999.0),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
		Nbins = cms.untracked.uint32(50),
    referenceBX= cms.untracked.uint32(1),
		NLuminositySegments= cms.untracked.uint32(1000),
		LuminositySegmentSize= cms.untracked.double(23.3),
		NbinsOneOverEt = cms.untracked.uint32(10000),

		muonEtaMax = cms.untracked.double(2.1),
    muonDRMatch = cms.untracked.double(0.3),
    muonEtMin = cms.untracked.double(0.0),

    jetDRMatch = cms.untracked.double(0.3),
    jetL1DRMatch = cms.untracked.double(0.5),
    jetEtMin = cms.untracked.double(5.0),

    electronDRMatch = cms.untracked.double(0.5),
    electronL1DRMatch = cms.untracked.double(0.5),
    electronEtMin = cms.untracked.double(5.0),

    photonDRMatch = cms.untracked.double(0.5),
    photonL1DRMatch = cms.untracked.double(0.5),
    photonEtMin = cms.untracked.double(5.0),

    #tauDRMatch = cms.untracked.double(0.1),

    #bjetDRMatch = cms.untracked.double(0.1),

    #trackDRMatch = cms.untracked.double(0.1),
     SpecialPaths = cms.vstring(
            'HLT_MET45',
            'HLT_L1Tech_HCAL_HF_coincidence_PM',
            'HLT_L1_BscMinBiasOR_BptxPlusORMinus',
            'HLT_MinBiasBSC',
            'HLT_MinBiasBSC_OR',
            'HLT_MinBiasEcal', 
            'HLT_MinBiasHcal', 
            'HLT_MinBiasPixel_SingleTrack', 
            'HLT_ZeroiasPixel_SingleTrack', 
            'HLT_L1_BPTX', 
            'HLT_ZeroBias'
      ),

     # Will mask any path whose name
     # contains any of the following sub-strings
     # For example: HLT_Mu
     # will mask all paths that contain the sub-string
     # HLT_Mu                            
     MaskedPaths = cms.vstring(
            'HLT_ZeroBias_v',
            'HLT_Mu5_v',
            'HLT_Mu8_v',
            'HLT_Mu12_v',
            'HLT_Jet30_v',
            'HLT_Jet60_v',
            'HLT_DiJetAve30_v'
            
      ),

  ## Robin
     testPaths = cms.vstring(
            'HLT_IsoMu30_eta2p1',
            'HLT_Ele65_CaloIdVT_TrkIdT',
            'HLT_MET200',
            'HLT_Jet370',
            'HLT_HT600',
            'HLT_Photon26_R9Id_Photon18_R9Id',
            'HLT_IsoMu15_eta2p1_LooseIsoPFTau20',
            'HLT_PFMHT150',
            'HLT_Photon90_CaloIdVL_IsoL'

      ),                           

   # Will pick the first trigger whose name contains this substring
 #  ReferenceTrigger = cms.string('HLT_Mu17_Ele8_CaloIdL_v'),        
   ReferenceTrigger = cms.string('HLT_Mu8_Ele17_CaloIdT_CaloIsoVL_v'),
                           
    paths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_"),
              denompathname = cms.string("MinBias")  
             ),
             cms.PSet(
              pathname = cms.string("EG"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("EG"),
              denompathname = cms.string("HLT_Mu5")  
             ),
             cms.PSet(
              pathname = cms.string("Jet"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Jet"),
              denompathname = cms.string("HLT_Mu5")  
             ),
             cms.PSet(
              pathname = cms.string("Ele"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Ele"),
              denompathname = cms.string("HLT_Mu5")  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Ele"),
              denompathname = cms.string("HLT_Activity_Ecal_SC7")  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Ele"),
              denompathname = cms.string("HLT_Activity_Ecal_SC1")  
             ),
             cms.PSet(
              pathname = cms.string("Pho"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Pho"),
              denompathname = cms.string("HLT_Mu5")  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Pho"),
              denompathname = cms.string("HLT_Activity_Ecal_SC7")  
             ),
             cms.PSet(
              pathname = cms.string("HLT_Pho"),
              denompathname = cms.string("HLT_Activity_Ecal_SC1")  
             ),
             cms.PSet(
              pathname = cms.string("Tau"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("MET"),
              denompathname = cms.string("HLT_Mu")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet")  
             ),
             cms.PSet(
              pathname = cms.string("Mu"),
              denompathname = cms.string("HLT_Jet50U")  
             )
    ),
    JetIDParams  = cms.PSet(
        useRecHits      = cms.bool(True),
        hbheRecHitsColl = cms.InputTag("hbhereco"),
        hoRecHitsColl   = cms.InputTag("horeco"),
        hfRecHitsColl   = cms.InputTag("hfreco"),
        ebRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEB"),
        eeRecHitsColl   = cms.InputTag("ecalRecHit", "EcalRecHitsEE")
    ),

                          
     # this is I think MC and CRUZET4
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    processname = cms.string("HLT")

    # this is data (CRUZET I or II best guess)
    #triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","FU"),
    #triggerResultsLabel = cms.InputTag("TriggerResults","","FU"),
    #processname = cms.string("FU")

 )

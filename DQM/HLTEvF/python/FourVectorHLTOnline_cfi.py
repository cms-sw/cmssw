import FWCore.ParameterSet.Config as cms

# $Id: FourVectorHLTOnline_cfi.py,v 1.11 2010/03/25 13:24:16 rekovic Exp $
hltResultsOn = cms.EDAnalyzer("FourVectorHLTOnline",
    dirname = cms.untracked.string("HLT/FourVector/paths"),
    muonRecoCollectionName = cms.untracked.string("muons"),
    plotAll = cms.untracked.bool(False),
    ptMax = cms.untracked.double(100.0),
    ptMin = cms.untracked.double(0.0),
		Nbins = cms.untracked.uint32(50),
    referenceBX= cms.untracked.uint32(1),
		NLuminositySegments= cms.untracked.uint32(2000),
		NbinsOneOverEt = cms.untracked.uint32(10000),

		muonEtaMax = cms.untracked.double(2.1),
    muonDRMatch = cms.untracked.double(0.3),

    electronDRMatch = cms.untracked.double(0.5),
    photonDRMatch = cms.untracked.double(0.5),
    #tauDRMatch = cms.untracked.double(0.1),
    #jetDRMatch = cms.untracked.double(0.1),
    #bjetDRMatch = cms.untracked.double(0.1),
    #photonDRMatch = cms.untracked.double(0.1),
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

    paths = cms.VPSet(
             cms.PSet(
              pathname = cms.string("HLT_"),
              denompathname = cms.string("HLT_MinBiasBSC")  
             )
            # cms.PSet(
            #  pathname = cms.string("HLT_Mu11"),
            #  denompathname = cms.string(""),  
            # ),
               #cms.PSet(
               # pathname = cms.string("HLT_Ele10_SW_L1R"),
               # denompathname = cms.string(""),  
               #)
#             cms.PSet(
#              pathname = cms.string("HLT_Photon15_L1R"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_MET25"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_HT250"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_BTagMu_Jet20"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string(""),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_Mu3"),  
#             )
#             cms.PSet(
#              pathname = cms.string("HLT_IsoEle15_L1I"),
#              denompathname = cms.string("HLT_IsoEle15_LW_L1I"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele15_SW_L1R"),
#              denompathname = cms.string("HLT_Ele15_SW_L1R"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele15_LW_L1R"),
#              denompathname = cms.string("HLT_Ele10_SW_L1R"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Ele15_LW_L1R"),
#              denompathname = cms.string("HLT_Ele15_LW_L1R"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu3"),
#              denompathname = cms.string("HLT_Mu3"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_Mu11"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Mu11"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Jet15"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet30"),
#              denompathname = cms.string("HLT_L1Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet50"),
#              denompathname = cms.string("HLT_Jet50"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_Jet30"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_Jet80"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet50"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet80"),
#              denompathname = cms.string("HLT_L1Mu"),  
#             ),
#             cms.PSet(
#              pathname = cms.string("HLT_Jet110"),
#              denompathname = cms.string("HLT_Jet110"),  
#             )
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


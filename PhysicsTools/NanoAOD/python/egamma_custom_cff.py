import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *
from PhysicsTools.NanoAOD.electrons_cff import *
from PhysicsTools.NanoAOD.electrons_cff import _eleVarsExtra
from PhysicsTools.NanoAOD.photons_cff import *
from PhysicsTools.NanoAOD.photons_cff import _phoVarsExtra
from PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff import *
from PhysicsTools.NanoAOD.nanoDQM_cfi import nanoDQM
from PhysicsTools.NanoAOD.nanoDQM_cff import _Photon_extra_plots, _Electron_extra_plots
from PhysicsTools.NanoAOD.triggerObjects_cff import triggerObjectTable, mksel
from RecoEgamma.EgammaIsolationAlgos.superclusValueMapProducer_cfi import superclusValueMaps

customElectronFilterBits = cms.PSet(
    doc = cms.string("PixelMatched e/gamma"), # this may also select photons!
    id = cms.int32(11),
    sel = cms.string("type(92) && pt > 7 && (coll('hltEgammaCandidates') || coll('hltEgammaCandidatesUnseeded')) && (filter('*PixelMatchFilter') || filter('*PixelMatchUnseededFilter'))"),
    l1seed = cms.string("type(-98)"),  l1deltaR = cms.double(0.3),
    skipObjectsNotPassingQualityBits = cms.bool(True),
    qualityBits = cms.VPSet(
        #HLT_Ele30_WPTight_Gsf
        mksel("filter('hltEGL1SingleEGOrFilter')","1e WPTight L1T match"), 
        mksel("filter('hltEG30L1SingleEGOrEtFilter')","1e WPTight Et"),
        mksel("filter('hltEle30WPTightClusterShapeFilter')","1e WPTight SigmaIeIe"),
        mksel("filter('hltEle30WPTightHEFilter')","1e WPTight HoE"),
        mksel("filter('hltEle30WPTightEcalIsoFilter')","1e WPTight ECAL Iso"),
        mksel("filter('hltEle30WPTightHcalIsoFilter')","1e WPTight HCAL Iso"),
        mksel("filter('hltEle30WPTightPixelMatchFilter')","1e WPTight Pixel match"),
        mksel("filter('hltEle30WPTightPMS2Filter')","1e WPTight S2"),
        mksel("filter('hltEle30WPTightGsfOneOEMinusOneOPFilter')","1e WPTight 1/E-1/p"),
        mksel("filter('hltEle30WPTightGsfMissingHitsFilter')","1e WPTight missing hits"),
        mksel("filter('hltEle30WPTightGsfDetaFilter')","1e WPTight DEta"),
        mksel("filter('hltEle30WPTightGsfDphiFilter')","1e WPTight DPhi"),
        mksel("filter('hltEle30WPTightGsfTrackIsoFilter')","1e WPTight Track Iso"),
        #HLT_Ele23_Ele12_CaloIdL_TrackIdL_IsoVL
        mksel("filter('hltEGL1SingleAndDoubleEGOrPairFilter')","2e L1T match"), 
        mksel("filter('hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg1Filter')","2e Track Iso Leg1"),
        mksel("filter('hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg2Filter')","2e Track Iso Leg2"),
        #HLT_DoubleEle33_CaloIdL_MW
        mksel("filter('hltEGL1SingleAndDoubleEGNonIsoOrWithEG26WithJetAndTauFilter')","2e (CaloIdL_MW) L1T match"), 
        mksel("filter('hltEle33CaloIdLPixelMatchFilter')","2e (CaloIdL_MW) Pixel match Leg1"),
        mksel("filter('hltEle33CaloIdLMWPMS2Filter')","2e (CaloIdL_MW) S2"),
        mksel("filter('hltDiEle33CaloIdLMWPMS2UnseededFilter')","2e (CaloIdL_MW unseeded) S2"),
        #HLT_Photon200
        mksel("filter('hltEG200HEFilter')","1e Photon200"),
        #HLT_Photon50EB
        mksel("filter('hltEG50EBEtFilter')","1e Photon50EB"),
        #HLT_Ele16_Ele12_Ele8_CaloIdL_TrackIdL
        mksel("filter('hltEle16Ele12Ele8CaloIdLTrackIdLDphiLeg3Filter')","3e Leg3"),
        #HLT_Mu12_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL (wildcarded as the name is likely going to change soon)
        mksel("filter('hltMu*TrkIsoVVLEle23CaloIdLTrackIdLIsoVLElectronlegTrackIsoFilter')","1mu-1e eLeg"),
        #HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJet30_*_eta2p3_CrossL1 OR HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1
        mksel("filter('hltEle24erWPTightGsfTrackIsoFilterForTau')","1e-1tau eLeg"),
        #HLT_Ele115_CaloIdVT_GsfTrkIdT
        mksel("filter('hltEle115CaloIdVTGsfTrkIdTGsfDphiFilter')","1e high pT noIso"),
    )
)

customPhotonFilterBits = cms.PSet(
    id = cms.int32(22),
    sel = cms.string("type(92) && pt > 15 && coll('hltEgammaCandidates')"),
    l1seed = cms.string("type(-98)"),  l1deltaR = cms.double(0.3),
    skipObjectsNotPassingQualityBits = cms.bool(True),
    qualityBits = cms.VPSet(
        #HLT_Photon50EB
        mksel("filter('hltEG50EBEtFilter')","Photon50EB"),
        #HLT_Photon***
        mksel("filter('hltEG120HEFilter')","Photon120"),
        mksel("filter('hltEG150HEFilter')","Photon150"),
        mksel("filter('hltEG175HEFilter')","Photon175"),
        mksel("filter('hltEG200HEFilter')","Photon200"),
        #HLT_ECALHT800        
        mksel("filter('hltHtEcal800')","ECAL HT800"),
        #HLT_Photon110EB_TightID_TightIso
        mksel("filter('hltEG110EBTightIDTightIsoTrackIsoFilter')","Photon110EB Tight"),
        #HLT_Mu17_Photon30_IsoCaloId
        mksel("filter('hltMu17Photon30IsoCaloIdPhotonlegTrackIsoFilter')","1mu-1photon"),
        #HLT_Diphoton30_22_R9Id_OR_IsoCaloId_AND_HE_R9Id_Mass90/95 
        mksel("filter('hltEG30LR9Id85b90eHE12R9Id50b80eR9IdLastFilter')","diPhoton 30_22 R9ID"),
        mksel("filter('hltEG30LIso60CaloId15b35eHE12R9Id50b80eEcalIsoLastFilter')","diPhoton 30_22 ECALIso"),
        mksel("filter('hltEG22R9Id85b90eHE12R9Id50b80eR9UnseededLastFilter')","diPhoton 30_22 unseeded R9ID"),
        mksel("filter('hltEG22Iso60CaloId15b35eHE12R9Id50b80eTrackIsoUnseededLastFilter')","diPhoton 30_22 unseeded TrackIso"),
    )
)

superclusterTable = cms.EDProducer("SimpleSuperclusterFlatTableProducer",
  src = cms.InputTag("reducedEgamma","reducedSuperClusters"),
  name = cms.string("Supercluster"),
  doc = cms.string("Supercluster collection"),
  variables = cms.PSet(
    energy = Var("energy()",float,doc="supercluster energy",precision=10),
    eta = Var("eta()",float,doc="supercluster eta",precision=10),
    phi = Var("phi()",float,doc="supercluster phi",precision=10),
    rawEnergy = Var("rawEnergy()",float,doc="sum of basic clusters energy",precision=10),
    preshowerEnergy = Var("preshowerEnergy()",float,doc="sum of energy in preshower",precision=10),
    etaWidth = Var("etaWidth()",float,doc="supercluster eta width",precision=10),
    phiWidth = Var("etaWidth()",float,doc="supercluster phi width",precision=10),
    seedClusEnergy = Var("seed().energy()",float,doc="seed cluster energy",precision=10),
    seedClusterEta = Var("seed().eta()",float,doc="seed cluster eta",precision=10),
    seedClusterPhi = Var("seed().phi()",float,doc="seed cluster phi",precision=10),
  ),
  externalVariables = cms.PSet(
    trkIso = ExtVar("superclusValueMaps:superclusTkIso",float,doc="supercluster track iso within 0.06 < dR < 0.4 & |dEta| > 0.03",precision=10),
  )
)

def addExtraEGammaVarsCustomize(process):
    #photon
    process.finalPhotons.cut = cms.string("pt > 1 ")
    process.photonTable.variables.setValue(_phoVarsExtra.parameters_())
    process.triggerObjectTable.selections.Photon = customPhotonFilterBits

    if hasattr(process,'nanoDQM'):
      process.nanoDQM.vplots.Photon.plots = _Photon_extra_plots

    #electron
    process.finalElectrons.cut = cms.string("pt > 1 ")
    process.electronTable.variables.setValue(_eleVarsExtra.parameters_())
    process.triggerObjectTable.selections.Electron = customElectronFilterBits

    if hasattr(process,'nanoDQM'):
      process.nanoDQM.vplots.Electron.plots = _Electron_extra_plots

    #superCluster
    process.superclusValueMaps = superclusValueMaps
    process.superclusterTable = superclusterTable

    process.superclusterTask = cms.Task(process.superclusValueMaps)
    process.superclusterTask.add(process.superclusterTable)
    process.nanoTableTaskCommon.add(process.superclusterTask)
      
    return process

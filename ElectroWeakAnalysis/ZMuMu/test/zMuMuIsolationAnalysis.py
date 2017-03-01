import FWCore.ParameterSet.Config as cms
import copy

process = cms.Process("ZMuMuIsolationAnalysis")

process.TFileService=cms.Service(
    "TFileService",
    fileName=cms.string("Prova_W_Isolamento.root")
    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
    )

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
    
 #   "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_1.root",
 #   "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_2.root",
 #   "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/zmm/zmumu_reskim_2.root",

#    "file:/scratch1/cms/data/summer08/skim/dimuons_skim_zmumu.root",
    "file:/scratch1/cms/data/summer08/skim/dimuons_skim_wmunu.root"
    
  # "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/qcd/noli/InclusiveMuPt15/InclusiveMuPt15SubSkim/d85f8e8eea12813d6b1603f1ce4b0f84/qcd_reskim_4.root",
   # "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/qcd/noli/InclusiveMuPt15/InclusiveMuPt15SubSkim/d85f8e8eea12813d6b1603f1ce4b0f84/qcd_reskim_5.root",
   # "rfio:/dpm/na.infn.it/home/cms/store/user/noli/reskim/qcd/noli/InclusiveMuPt15/InclusiveMuPt15SubSkim/d85f8e8eea12813d6b1603f1ce4b0f84/qcd_reskim_6.root"
    )
    )

process.zmumuNewIsolation = cms.EDAnalyzer(
    "ZMuMuIsolationAnalyzer",
    src = cms.InputTag("dimuonsOneTrack"),
    ptThreshold = cms.untracked.double(1),
    etEcalThreshold = cms.untracked.double(0.2),
    etHcalThreshold = cms.untracked.double(0.5),
    deltaRTrk = cms.untracked.double(0.2),
    deltaREcal = cms.untracked.double(0.25),
    deltaRHcal = cms.untracked.double(0.25),
    veto = cms.untracked.double(0.015),
    alpha = cms.untracked.double(0.75),
    beta = cms.untracked.double(-0.75),
    pt = cms.untracked.double(20),
    eta = cms.untracked.double(2),
    isoCut = cms.untracked.double(1.7)
    )

cut = [0.4,0.6,0.8,1.0,1.2]

for i in range(len(cut)):
    ptThreshold = cut[i]
    print i, ") cut = ",  ptThreshold 
    
    plotModuleLabel = "isoPlots_ptTkr_" + str(i);
    module = copy.deepcopy(process.zmumuNewIsolation)
    setattr(module, "ptThreshold", ptThreshold)
    setattr(process, plotModuleLabel, module)
    
    plotPathLabel = "isoPath_ptTkr_" + str(i);
    path = cms.Path(module);
    setattr(process, plotPathLabel, path)


ecal_cut = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]

for i in range(len(ecal_cut)):
    etEcalThreshold = ecal_cut[i]
    print i, ") cut = ",  etEcalThreshold 
    
    plotModuleLabel = "isoPlots_etEcal_" + str(i);
    module = copy.deepcopy(process.zmumuNewIsolation)
    setattr(module, "etEcalThreshold", etEcalThreshold)
    setattr(process, plotModuleLabel, module)
    
    plotPathLabel = "isoPath_etEcal_" + str(i);
    path = cms.Path(module);
    setattr(process, plotPathLabel, path)


hcal_cut = [0.5,0.6,0.7,0.8,0.9,1.]

for i in range(len(hcal_cut)):
    etHcalThreshold = hcal_cut[i]
    print i, ") cut = ",  etHcalThreshold 
    
    plotModuleLabel = "isoPlots_etHcal_" + str(i);
    module = copy.deepcopy(process.zmumuNewIsolation)
    setattr(module, "etHcalThreshold", etHcalThreshold)
    setattr(process, plotModuleLabel, module)
    
    plotPathLabel = "isoPath_etHcal_" + str(i);
    path = cms.Path(module);
    setattr(process, plotPathLabel, path)


deltaR_ = [0.05,0.10,0.15,0.18,0.20,0.25,0.30,0.35]

for i in range(len(deltaR_)):
    deltaRTrk = deltaR_[i]
    deltaREcal = deltaR_[i]
    deltaRHcal = deltaR_[i]  
    
    print i, ") deltaR = ",  deltaRTrk 

    plotModuleLabel = "isoPlots_DR_" + str(i);
    module = copy.deepcopy(process.zmumuNewIsolation)
    setattr(module, "deltaRTrk", deltaRTrk)
    setattr(module, "deltaREcal", deltaREcal)
    setattr(module, "deltaRHcal", deltaRHcal)
    setattr(process, plotModuleLabel, module)

    plotPathLabel = "isoPath_DR_" + str(i);
    path = cms.Path(module);
    setattr(process, plotPathLabel, path)


alpha_array = [0.0,0.25,0.5,0.75,1.0]
beta_array = [-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0]

for i in range(len(alpha_array)):
    alpha = alpha_array[i]
    print i, ") alpha = ",  alpha 

    for j in range(len(beta_array)):
        beta = beta_array[j]
        print i,".", j, ") beta = ",  beta
        
        plotModuleLabel = "isoPlots_LinearComb_" + str(i)+"_" + str(j);
        module = copy.deepcopy(process.zmumuNewIsolation) 
        setattr(module, "alpha", alpha)
        setattr(module, "beta", beta)
        setattr(process, plotModuleLabel, module)

        plotPathLabel = "isoPath_LineareComb_" + str(i) +"_"+ str(j);
        path = cms.Path(module);
        setattr(process, plotPathLabel, path)

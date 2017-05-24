import FWCore.ParameterSet.Config as cms

#urrgh, horrible hack coming in
def makeEGEffHistDef(baseName,filterName):   
   
    return ["{0}_{1}_vsEt_eff '{1} Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_vsEt_pass {0}_{1}_vsEt_tot".format(baseName,filterName),
            "{0}_{1}_EBvsEt_eff '{1} Barrel Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_EBvsEt_pass {0}_{1}_EBvsEt_tot".format(baseName,filterName),
            "{0}_{1}_EEvsEt_eff '{1} Endcap Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_EEvsEt_pass {0}_{1}_EEvsEt_tot".format(baseName,filterName),
            "{0}_{1}_vsSCEta_eff '{1} Efficiency vs SC #eta;#eta_{{SC}};Efficiency' {0}_{1}_vsSCEta_pass {0}_{1}_vsSCEta_tot".format(baseName,filterName),
            "{0}_{1}_EBvsPhi_eff '{1} Barrel Efficiency vs #phi [rad];#phi [rad];Efficiency' {0}_{1}_EBvsPhi_pass {0}_{1}_EBvsPhi_tot".format(baseName,filterName),
            "{0}_{1}_EEvsPhi_eff '{1} Endcap Efficiency vs #phi;#phi [rad];Efficiency' {0}_{1}_EEvsPhi_pass {0}_{1}_EEvsPhi_tot".format(baseName,filterName),            
            "{0}_{1}_vsSCEtaPhi_eff '{1} Efficiency vs SC #eta/#phi;#eta_{{SC}};#phi [rad];' {0}_{1}_vsSCEtaPhi_pass {0}_{1}_vsSCEtaPhi_tot".format(baseName,filterName)]
            
def makeAllEGEffHistDefs():
    baseNames=["ele27Tag","ele27Tag_HEM17","ele27Tag_HEP17"]
    filterNames=["hltEle33CaloIdLMWPMS2Filter","hltDiEle33CaloIdLMWPMS2UnseededFilter","hltEG300erFilter","hltEG70HEFilter","hltDiEG70HEUnseededFilter","hltEG85HEFilter","hltDiEG85HEUnseededFilter","hltEG30EIso15HE30EcalIsoLastFilter","hltEG18EIso15HE30EcalIsoUnseededFilter","hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg1Filter","hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg2Filter","hltEle27WPTightGsfTrackIsoFilter","hltEle32noerWPTightGsfTrackIsoFilter","hltEle38noerWPTightGsfTrackIsoFilter","hltEle40noerWPTightGsfTrackIsoFilter","hltEG33L1EG26HEFilter","hltEG50HEFilter","hltEG75HEFilter","hltEG90HEFilter","hltEG120HEFilter","hltEG150HEFilter","hltEG175HEFilter","hltEG200HEFilter","hltSingleCaloJet500","hltSingleCaloJet550"]

    histDefs=[]
    for baseName in baseNames:
        for filterName in filterNames:
            histDefs.extend(makeEGEffHistDef(baseName,filterName))
    return histDefs


egTPEffClient = cms.EDAnalyzer("DQMGenericClient",
                                subDirs        = cms.untracked.vstring("HLT/EGTagAndProbeEffs/*"),
                                verbose        = cms.untracked.uint32(2), # Set to 2 for all messages
                                resolution     = cms.vstring(),
                                efficiency     = cms.vstring(),
                                efficiencyProfile = cms.untracked.vstring()
                                
                                )
egTPEffClient.efficiency.extend(makeAllEGEffHistDefs())

egammaClient = cms.Sequence(
    egTPEffClient
)

import FWCore.ParameterSet.Config as cms

#urrgh, horrible hack coming in
def makeEGEffHistDef(baseName,filterName):   
   
    return ["{0}_{1}_vsEt_eff '{1} Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_vsEt_pass {0}_{1}_vsEt_tot".format(baseName,filterName),
            "{0}_{1}_EBvsEt_eff '{1} Barrel Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_EBvsEt_pass {0}_{1}_EBvsEt_tot".format(baseName,filterName),
            "{0}_{1}_EEvsEt_eff '{1} Endcap Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_EEvsEt_pass {0}_{1}_EEvsEt_tot".format(baseName,filterName),
            "{0}_{1}_vsEt_eff '{1} Efficiency vs E_{{T}};E_{{T}} [GeV];Efficiency' {0}_{1}_vsEt_pass {0}_{1}_vsEt_tot".format(baseName,filterName),
            "{0}_{1}_vsSCEta_eff '{1} Efficiency vs SC #eta;#eta_{{SC}};Efficiency' {0}_{1}_vsSCEta_pass {0}_{1}_vsSCEta_tot".format(baseName,filterName),
            "{0}_{1}_EBvsPhi_eff '{1} Barrel Efficiency vs #phi [rad];#phi [rad];Efficiency' {0}_{1}_EBvsPhi_pass {0}_{1}_EBvsPhi_tot".format(baseName,filterName),
            "{0}_{1}_EEvsPhi_eff '{1} Endcap Efficiency vs #phi;#phi [rad];Efficiency' {0}_{1}_EEvsPhi_pass {0}_{1}_EEvsPhi_tot".format(baseName,filterName), 
            "{0}_{1}_vsPhi_eff '{1} Efficiency vs #phi;#phi [rad];Efficiency' {0}_{1}_vsPhi_pass {0}_{1}_vsPhi_tot".format(baseName,filterName),            
            "{0}_{1}_vsSCEtaPhi_eff '{1} Efficiency vs SC #eta/#phi;#eta_{{SC}};#phi [rad];' {0}_{1}_vsSCEtaPhi_pass {0}_{1}_vsSCEtaPhi_tot".format(baseName,filterName)]
            
def makeAllEGEffHistDefs():
    baseNames=["eleWPTightTag","eleWPTightTag-HEP17","eleWPTightTag-HEM17"]
    filterNames=["hltEle33CaloIdLMWPMS2Filter","hltDiEle33CaloIdLMWPMS2UnseededFilter","hltEG300erFilter","hltEG70HEFilter","hltDiEG70HEUnseededFilter","hltEG85HEFilter","hltDiEG85HEUnseededFilter","hltEG30EIso15HE30EcalIsoLastFilter","hltEG18EIso15HE30EcalIsoUnseededFilter","hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg1Filter","hltEle23Ele12CaloIdLTrackIdLIsoVLTrackIsoLeg2Filter","hltEle27WPTightGsfTrackIsoFilter","hltEle32WPTightGsfTrackIsoFilter","hltEle35noerWPTightGsfTrackIsoFilter","hltEle38noerWPTightGsfTrackIsoFilter","hltEle27L1DoubleEGWPTightGsfTrackIsoFilter","hltEle32L1DoubleEGWPTightGsfTrackIsoFilter","hltEG25L1EG18HEFilter","hltEG33L1EG26HEFilter","hltEG50HEFilter","hltEG75HEFilter","hltEG90HEFilter","hltEG120HEFilter","hltEG150HEFilter","hltEG175HEFilter","hltEG200HEFilter","hltSingleCaloJet500","hltSingleCaloJet550","hltEle28HighEtaSC20TrackIsoFilter","hltEle50CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle115CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle135CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle145CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle200CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle250CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle300CaloIdVTGsfTrkIdTGsfDphiFilter","hltEle20WPLoose1GsfTrackIsoFilter","hltEle20erWPLoose1GsfTrackIsoFilter","hltEle20WPTightGsfTrackIsoFilter","hltEle27L1DoubleEGWPTightEcalIsoFilter","hltDiEle27L1DoubleEGWPTightEcalIsoFilter","hltEle27CaloIdLMWPMS2Filter","hltDiEle27CaloIdLMWPMS2UnseededFilter","hltEle25CaloIdLMWPMS2Filter","hltDiEle25CaloIdLMWPMS2UnseededFilter","hltEle27CaloIdLMWPMS2Filter","hltDiEle27CaloIdLMWPMS2UnseededFilter","hltEle37CaloIdLMWPMS2UnseededFilter","hltSingleEle35WPTightGsfL1EGMTTrackIsoFilter"
                 ]
    
    

    histDefs=[]
    for baseName in baseNames:
        for filterName in filterNames:
            histDefs.extend(makeEGEffHistDef(baseName,filterName))

    baseNames=["eleWPTightTagPhoHighEtaProbe","eleWPTightTagPhoHighEtaProbe-HEM17","eleWPTightTagPhoHighEtaProbe-HEP17"]
    filterNames=["hltEle28HighEtaSC20Mass55Filter","hltEle28HighEtaSC20HcalIsoFilterUnseeded"]

    for baseName in baseNames:
        for filterName in filterNames:
            histDefs.extend(makeEGEffHistDef(baseName,filterName))


    baseNames=["eleWPTightTagPhoProbe","eleWPTightTagPhoProbe-HEM17","eleWPTightTagPhoProbe-HEP17"]
    filterNames=["hltEG20CaloIdLV2ClusterShapeL1TripleEGFilter","hltTriEG20CaloIdLV2ClusterShapeUnseededFilter","hltEG20CaloIdLV2R9IdVLR9IdL1TripleEGFilter","hltTriEG20CaloIdLV2R9IdVLR9IdUnseededFilter","hltEG30CaloIdLV2ClusterShapeL1TripleEGFilter","hltEG10CaloIdLV2ClusterShapeUnseededFilter","hltDiEG30CaloIdLV2EtUnseededFilter","hltEG30CaloIdLV2R9IdVLR9IdL1TripleEGFilter","hltEG10CaloIdLV2R9IdVLR9IdUnseededFilter","hltDiEG30CaloIdLV2R9IdVLEtUnseededFilter","hltEG35CaloIdLV2R9IdVLR9IdL1TripleEGFilter","hltEG5CaloIdLV2R9IdVLR9IdUnseededFilter","hltDiEG35CaloIdLV2R9IdVLEtUnseededFilter"]

    for baseName in baseNames:
        for filterName in filterNames:
            histDefs.extend(makeEGEffHistDef(baseName,filterName))
    
    baseNames=["muonIsoMuTagPhoProbe","muonIsoMuTagPhoProbe-HEM17","muonIsoMuTagPhoProbe-HEP17"]
    filterNames=["hltMu12DiEG20HEUnseededFilter"]

    for baseName in baseNames:
        for filterName in filterNames:
            histDefs.extend(makeEGEffHistDef(baseName,filterName))

    baseNames=["muonIsoMuTagEleProbe","muonIsoMuTagEleProbe-HEM17","muonIsoMuTagEleProbe-HEP17"]
    filterNames=["hltMu12DiEG20HEUnseededFilter","hltEle27CaloIdLMWPMS2UnseededFilter","hltEle37CaloIdLMWPMS2UnseededFilter","hltEle33CaloIdLMWPMS2Filter","hltEle32WPTightGsfTrackIsoFilter","hltEle32L1DoubleEGWPTightGsfTrackIsoFilter"]
    for baseName in baseNames:
        for filterName in filterNames:
            histDefs.extend(makeEGEffHistDef(baseName,filterName))

#    print histDefs
    return histDefs


egTPEffClient = cms.EDProducer("DQMGenericClient",
                               subDirs        = cms.untracked.vstring("HLT/EGTagAndProbeEffs/*"),
                               verbose        = cms.untracked.uint32(0), 
                               resolution     = cms.vstring(),
                               efficiency     = cms.vstring(),
                               efficiencyProfile = cms.untracked.vstring()
                               )
egTPEffClient.efficiency.extend(makeAllEGEffHistDefs())

egammaClient = cms.Sequence(
    egTPEffClient
)

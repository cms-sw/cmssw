import FWCore.ParameterSet.Config as cms

# rootfile="relval214/qcd80-120_output.root"
# rootfile="relval220pre1/qcd80-120_output.root"
# rootfile="relval220/qcd80-120_output.root"
# rootfile="relval220_ideal/qcd80-120_output.root"
# rootfile="relval220_ideal/qcd80-120_fastsim_output.root"
# rootfile="relval214/ttbar_output.root"
# rootfile="relval220pre1/ttbar_output.root"
# rootfile="relval220/ttbar_output.root"
# rootfile="relval220_ideal/ttbar_output.root"
rootfile="relval220_ideal/ttbar_fastsim_output.root"


# calojetcoll="iterativeCone5CaloJets"
calojetcoll="hltIterativeCone5CaloJets"

probetrig50="hlt1jet50"
reftrig50  ="hlt1jet30"
hltname50="HLT_Jet50"
folderjet50="HLT/HLTJETMET/SingleJet50"

probetrig80="hlt1jet80"
reftrig80="hlt1jet30"
hltname80="HLT_Jet80"
folderjet80="HLT/HLTJETMET/SingleJet80"

probetrig110="hlt1jet110"
reftrig110="hlt1jet30"
hltname110="HLT_Jet110"
folderjet110="HLT/HLTJETMET/SingleJet110"

probetrig180="hlt1jet180"
reftrig180="hlt1jet30"
hltname180="HLT_Jet180"
folderjet180="HLT/HLTJETMET/SingleJet180"

probetrigM35="hlt1MET35"
reftrigM35="hlt1MET25"
hltnameM35="HLT_MET35"
folderMET35="HLT/HLTJETMET/SingleMET35"

probetrigM50="hlt1MET50"
reftrigM50="hlt1MET25"
hltnameM50="HLT_MET50"
folderMET50="HLT/HLTJETMET/SingleMET50"

probetrigM65="hlt1MET65"
reftrigM65="hlt1MET25"
hltnameM65="HLT_MET65"
folderMET65="HLT/HLTJETMET/SingleMET65"

probetrigM75="hlt1MET75"
reftrigM75="hlt1MET25"
hltnameM75="HLT_MET75"
folderMET75="HLT/HLTJETMET/SingleMET75"

probetrigQJ30="hltquadjet30"
reftrigQJ30="hlt1jet30"
hltnameQJ30="HLT_QuadJet30"
folderQJ30="HLT/HLTJETMET/QuadJet30"

probetrigQJ60="hltquadjet60"
reftrigQJ60="hlt1jet60"
hltnameQJ60="HLT_QuadJet60"
folderQJ60="HLT/HLTJETMET/QuadJet60"


SingleJetPathVal50 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderjet50),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrig50,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrig50,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltname50),
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleJetPathVal80 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderjet80),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrig80,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrig80,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltname80),                                  
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),                                  
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleJetPathVal110 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderjet110),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrig110,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrig110,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltname110),                                   
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleJetPathVal180 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderjet180),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrig180,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrig180,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltname180),                                   
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),                                
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleMETPathVal35 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderMET35),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrigM35,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrigM35,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltnameM35),                                  
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),                                  
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleMETPathVal50 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderMET50),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrigM50,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrigM50,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltnameM50),                                  
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),                                  
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleMETPathVal65 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderMET65),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrigM65,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrigM65,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltnameM65),                                  
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),                                  
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

SingleMETPathVal75 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderMET75),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrigM75,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrigM75,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltnameM75),                                  
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),                                  
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)

QuadJetPathVal30 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderQJ30),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrigQJ30,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrigQJ30,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltnameQJ30),
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)


QuadJetPathVal60 = cms.EDFilter("HLTJetMETValidation",
    triggerEventObject    = cms.untracked.InputTag("hltTriggerSummaryRAW","","HLT"),
#    refTauCollection      = cms.untracked.InputTag("JetMETMCProducer","Taus"),
#    refLeptonCollection   = cms.untracked.InputTag("NOTHING"),
    DQMFolder             = cms.untracked.string(folderQJ60),
#    L1SeedFilter          = cms.untracked.InputTag("hltSingleTauL1SeedFilter","","HLT"),
#    L2EcalIsolFilter      = cms.untracked.InputTag("hltFilterSingleTauEcalIsolation","","HLT"),
#    L25PixelIsolFilter    = cms.untracked.InputTag("hltFilterL25SingleTau","","HLT"),
#    L3SiliconIsolFilter   = cms.untracked.InputTag("hltFilterL3SingleTau","","HLT"),
#    MuonFilter            = cms.untracked.InputTag("DUMMY"),
#    ElectronFilter        = cms.untracked.InputTag("DUMMY"),
#    NTriggeredTaus        = cms.untracked.uint32(1),
#    NTriggeredLeptons     = cms.untracked.uint32(0),
    DoReferenceAnalysis   = cms.untracked.bool(True),
    OutputFileName        = cms.untracked.string(rootfile),
    LogFileName           = cms.untracked.string('JetMETSingleJetValidation.log'),
#    RefFilter             = cms.untracked.InputTag("hltL1s1Level1jet15","","HLT"),
    RefFilter             = cms.untracked.InputTag(reftrigQJ60,"","HLT"),
    ProbeFilter           = cms.untracked.InputTag(probetrigQJ60,"","HLT"),
    HLTPath               = cms.untracked.InputTag(hltnameQJ60),
    CaloJetAlgorithm      = cms.untracked.InputTag(calojetcoll),
    GenJetAlgorithm       = cms.untracked.InputTag("iterativeCone5GenJets"),
    CaloMETCollection     = cms.untracked.InputTag("hltMet"),
    GenMETCollection      = cms.untracked.InputTag("genMetTrue"),
    HLTriggerResults = cms.InputTag("TriggerResults::HLT"),
#    MatchDeltaRL1         = cms.untracked.double(0.5),
#    MatchDeltaRHLT        = cms.untracked.double(0.3)
)



#SingleJetValidation = cms.Sequence(SingleJetPathVal + SingleJetL2Val + SingleJetL25Val+SingleJetL3Val)
SingleJetValidation = cms.Sequence(SingleJetPathVal50 + SingleJetPathVal80 + SingleJetPathVal110 + SingleJetPathVal180 +
                                   SingleMETPathVal35 +
                                   SingleMETPathVal50 +
                                   SingleMETPathVal65 +
                                   SingleMETPathVal75 +
                                   QuadJetPathVal30 + QuadJetPathVal60
                                   )



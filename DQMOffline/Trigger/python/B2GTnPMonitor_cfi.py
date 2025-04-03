import FWCore.ParameterSet.Config as cms

#this is the config to define t&p based DQM offline monitoring for B2G (copied from  cmssw/DQMOffline/Trigger/python/HLTEGTnPMonitor_cfi.py

etBinsStd=cms.vdouble(5,10,12.5,15,17.5,20,22.5,25,30,35,40,45,50,60,80,100,150,200,250,300,350,400)
scEtaBinsStd = cms.vdouble(-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.566,-1.4442,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4442,1.566,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5)
phiBinsStd = cms.vdouble(-3.32,-2.97,-2.62,-2.27,-1.92,-1.57,-1.22,-0.87,-0.52,-0.18,0.18,0.52,0.87,1.22,1.57,1.92,2.27,2.62,2.97,3.32)

etRangeCut= cms.PSet(
    rangeVar=cms.string("et"),
    allowedRanges=cms.vstring("0:10000"),
    )
ecalBarrelEtaCut=cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-1.4442:1.4442")
    )
ecalEndcapEtaCut=cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-2.5:-1.556","1.556:2.5")
    )
ecalBarrelAndEndcapEtaCut = cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-1.4442:1.4442","-2.5:-1.556","1.556:2.5"),
    )
hcalPosEtaCut= cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("1.3:1.4442","1.556:2.5"),
    )
hcalNegEtaCut= cms.PSet(
    rangeVar=cms.string("scEta"),
    allowedRanges=cms.vstring("-2.5:-1.556","-1.4442:-1.3"),
    )
hcalPhi17Cut = cms.PSet(
    rangeVar=cms.string("phi"),
    allowedRanges=cms.vstring("-0.87:-0.52"),
    )

tagAndProbeConfigEle50CaloIdVTGsfTrkIdT = cms.PSet(
    trigEvent = cms.InputTag("hltTriggerSummaryAOD","","HLT"),

    tagVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-RunIIIWinter22-V1-tight"),
    probeVIDCuts = cms.InputTag("egmGsfElectronIDsForDQM:cutBasedElectronID-RunIIIWinter22-V1-tight"),
    sampleTrigRequirements = cms.PSet(
        hltInputTag = cms.InputTag("TriggerResults","","HLT"),
        hltPaths = cms.vstring("HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165_v*")
        ),
    #it is intended that these are the filters of the triggers listed for sampleTrigRequirements
    tagFilters = cms.vstring("hltEle50CaloIdVTGsfTrkIdTCentralPFJet165EleCleaned"),
    tagFiltersORed = cms.bool(True),
    tagRangeCuts = cms.VPSet(ecalBarrelAndEndcapEtaCut),
    probeFilters = cms.vstring(),
    probeFiltersORed = cms.bool(False),
    probeRangeCuts = cms.VPSet(ecalBarrelAndEndcapEtaCut),
    minMass = cms.double(70.0),
    maxMass = cms.double(110.0),
    requireOpSign = cms.bool(False),
    )
    
egammaStdHistConfigs = cms.VPSet(
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("et"),
        nameSuffex=cms.string("_EBvsEt"),
        rangeCuts=cms.VPSet(ecalBarrelEtaCut),
        binLowEdges=etBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("et"),
        nameSuffex=cms.string("_EEvsEt"),
        rangeCuts=cms.VPSet(ecalEndcapEtaCut),
        binLowEdges=etBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("scEta"),
        nameSuffex=cms.string("_vsSCEta"),
        rangeCuts=cms.VPSet(),
        binLowEdges=scEtaBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_EBvsPhi"),
        rangeCuts=cms.VPSet(ecalBarrelEtaCut),
        binLowEdges=phiBinsStd,
        ),
    cms.PSet(
        histType=cms.string("1D"),
        vsVar=cms.string("phi"),
        nameSuffex=cms.string("_EEvsPhi"),
        rangeCuts=cms.VPSet(ecalEndcapEtaCut),
        binLowEdges=phiBinsStd,
        ),
    cms.PSet(
        histType=cms.string("2D"),
        xVar=cms.string("scEta"),
        yVar=cms.string("phi"),
        nameSuffex=cms.string("_vsSCEtaPhi"), 
        rangeCuts=cms.VPSet(),
        xBinLowEdges=scEtaBinsStd,
        yBinLowEdges=phiBinsStd,
        ),
    
    )

egammaStdFiltersToMonitor= cms.VPSet(
    cms.PSet(
        folderName = cms.string("HLT/B2G/HLT_Ele50_CaloIdVT_GsfTrkIdT_PFJet165"),
        rangeCuts = cms.VPSet(etRangeCut.clone(allowedRanges= ["55:99999"]),),
        filterName = cms.string("hltEle50CaloIdVTGsfTrkIdTCentralPFJet165EleCleaned"),
        histTitle = cms.string(""),
        tagExtraFilter = cms.string(""),
        ),
       
   
     )
  
 

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
B2GegHLTDQMOfflineTnPSource = DQMEDAnalyzer("HLTEleTagAndProbeOfflineSource",
                                          tagAndProbeCollections = cms.VPSet(
        cms.PSet( 
            tagAndProbeConfigEle50CaloIdVTGsfTrkIdT,
            histConfigs = egammaStdHistConfigs,
            baseHistName = cms.string("eleWPTightTag_"),
            filterConfigs = egammaStdFiltersToMonitor,
        ),

           
        )
                                         )

from RecoEgamma.ElectronIdentification.egmGsfElectronIDs_cff import egmGsfElectronIDs

B2GegmGsfElectronIDsForDQM = egmGsfElectronIDs.clone(
    physicsObjectsIDs = cms.VPSet(),
    physicsObjectSrc = 'gedGsfElectrons'
)
#note: be careful here to when selecting new ids that the vid tools doesnt do extra setup for them
#for example the HEEP cuts need an extra producer which vid tools automatically handles
from PhysicsTools.SelectorUtils.tools.vid_id_tools import setupVIDSelection
my_id_modules = ['RecoEgamma.ElectronIdentification.Identification.cutBasedElectronID_Winter22_122X_V1_cff']
for id_module_name in my_id_modules: 
    idmod= __import__(id_module_name, globals(), locals(), ['idName','cutFlow'])
    for name in dir(idmod):
        item = getattr(idmod,name)
        if hasattr(item,'idName') and hasattr(item,'cutFlow'):
            setupVIDSelection(B2GegmGsfElectronIDsForDQM,item)

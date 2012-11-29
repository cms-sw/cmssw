import FWCore.ParameterSet.Config as cms

isData=True
outputRAW=False
maxNrEvents=1000
outputSummary=True
newL1Menu=False
hltProcName="HLT3"
runOpen=False #ignore all filter decisions, true for testing
runProducers=True #run the producers or not, 
if isData:
    from hlt import *
else:
    from muPathsMC import *

process.load("setup_cff")

if runProducers==False:
    hltProcName=hltProcName+"PB"

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(maxNrEvents)
)

# enable the TrigReport and TimeReport
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool( outputSummary ),
    SkipEvent = cms.untracked.vstring('ProductNotFound')
)

import sys
filePrefex="file:"
if(sys.argv[2].find("/pnfs/")==0):
    filePrefex="dcap://heplnx209.pp.rl.ac.uk:22125"

if(sys.argv[2].find("/store/")==0):
    filePrefex=""

if(sys.argv[2].find("/castor/")==0):
    filePrefex="rfio:"
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(),
                            eventsToProcess =cms.untracked.VEventRange()
                            )
for i in range(2,len(sys.argv)-1):
    print filePrefex+sys.argv[i]
    process.source.fileNames.extend([filePrefex+sys.argv[i],])




process.load('Configuration/EventContent/EventContent_cff')
process.output = cms.OutputModule("PoolOutputModule",
                                  splitLevel = cms.untracked.int32(0),
                                  outputCommands =cms.untracked.vstring("drop *",
                                                                        "keep *_TriggerResults_*_*",
                                                                        "keep *_hltTriggerSummaryAOD_*_*"),
                                  
                                  fileName = cms.untracked.string(sys.argv[len(sys.argv)-1]),
                                  dataset = cms.untracked.PSet(dataTier = cms.untracked.string('HLTDEBUG'),)
                                  )
if outputRAW:
    process.output.outputCommands=cms.untracked.vstring("drop *","keep *_rawDataCollector_*_*","keep *_addPileupInfo_*_*","keep *_TriggerResults_*_*","keep *_hltTriggerSummaryAOD_*_*")
                                                                
process.HLTOutput_sam = cms.EndPath(process.output)

isCrabJob=False
#if 1, its a crab job...
if isCrabJob:
    print "using crab specified filename"
    process.output.fileName= "OUTPUTFILE"
  
else:
    print "using user specified filename"
    process.output.fileName= sys.argv[len(sys.argv)-1]

#hlt stuff
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(500),
    limit = cms.untracked.int32(10000000)
)

# override the process name
process.setName_(hltProcName)

# En-able HF Noise filters in GRun menu
if 'hltHfreco' in process.__dict__:
    process.hltHfreco.setNoiseFlags = cms.bool( True )

# override the L1 menu from an Xml file
if newL1Menu:
    process.l1GtTriggerMenuXml = cms.ESProducer("L1GtTriggerMenuXmlProducer",
                                                TriggerMenuLuminosity = cms.string('startup'),
                                                DefXmlFile = cms.string('L1Menu_Collisions2012_v0_L1T_Scales_20101224_Imp0_0x1027.xml'),
                                                VmeXmlFile = cms.string('')
                                                )
    
    process.L1GtTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
                                                    recordName = cms.string('L1GtTriggerMenuRcd'),
                                                    iovIsRunNotTime = cms.bool(True),
                                                    firstValid = cms.vuint32(1)
                                                    )
    
    process.es_prefer_l1GtParameters = cms.ESPrefer('L1GtTriggerMenuXmlProducer','l1GtTriggerMenuXml') 


# adapt HLT modules to the correct process name
if 'hltTrigReport' in process.__dict__:
    process.hltTrigReport.HLTriggerResults                    = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreExpressCosmicsOutputSmart' in process.__dict__:
    process.hltPreExpressCosmicsOutputSmart.TriggerResultsTag = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreExpressOutputSmart' in process.__dict__:
    process.hltPreExpressOutputSmart.TriggerResultsTag        = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreDQMForHIOutputSmart' in process.__dict__:
    process.hltPreDQMForHIOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreDQMForPPOutputSmart' in process.__dict__:
    process.hltPreDQMForPPOutputSmart.TriggerResultsTag       = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreHLTDQMResultsOutputSmart' in process.__dict__:
    process.hltPreHLTDQMResultsOutputSmart.TriggerResultsTag  = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreHLTDQMOutputSmart' in process.__dict__:
    process.hltPreHLTDQMOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltPreHLTMONOutputSmart' in process.__dict__:
    process.hltPreHLTMONOutputSmart.TriggerResultsTag         = cms.InputTag( 'TriggerResults', '', hltProcName )

if 'hltDQMHLTScalers' in process.__dict__:
    process.hltDQMHLTScalers.triggerResults                   = cms.InputTag( 'TriggerResults', '', hltProcName )
    process.hltDQMHLTScalers.processname                      = hltProcName

if 'hltDQML1SeedLogicScalers' in process.__dict__:
    process.hltDQML1SeedLogicScalers.processname              = hltProcName

# remove the HLT prescales
if 'PrescaleService' in process.__dict__:
    process.PrescaleService.lvl1DefaultLabel = cms.string( '0' )
    process.PrescaleService.lvl1Labels       = cms.vstring( '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' )
    process.PrescaleService.prescaleTable    = cms.VPSet( )


# override the GlobalTag, connection string and pfnPrefix
if 'GlobalTag' in process.__dict__:
    process.GlobalTag.connect   = 'frontier://FrontierProd/CMS_COND_31X_GLOBALTAG'
    process.GlobalTag.pfnPrefix = cms.untracked.string('frontier://FrontierProd/')
    from Configuration.AlCa.autoCond import autoCond
    if isData:
       # process.GlobalTag.globaltag = autoCond['hltonline'].split(',')[0]
        process.GlobalTag.globaltag = 'GR_H_V29::All'
    else:
        process.GlobalTag.globaltag = autoCond['startup']

if 'MessageLogger' in process.__dict__:
    process.MessageLogger.categories.append('TriggerSummaryProducerAOD')
    process.MessageLogger.categories.append('L1GtTrigReport')
    process.MessageLogger.categories.append('HLTrigReport')
    process.MessageLogger.suppressInfo = cms.untracked.vstring('ElectronSeedProducer',"hltL1NonIsoStartUpElectronPixelSeeds","hltL1IsoStartUpElectronPixelSeeds","BasicTrajectoryState")


prodWhiteList=[]
prodWhiteList.append("hltBLifetimeL25TagInfosbbPhiL1FastJet")
prodWhiteList.append("hltBSoftMuonDiJet20L1FastJetL25BJetTagsByDR")
prodWhiteList.append("hltBSoftMuonGetJetsFromDiJet20L1FastJet")
prodWhiteList.append("hltBSoftMuonDiJet40L1FastJetL25TagInfos")
prodWhiteList.append("hltBSoftMuonDiJet40L1FastJetL25BJetTagsByDR")
prodWhiteList.append("hltBSoftMuonDiJet40L1FastJetMu5SelL3BJetTagsByDR")
prodWhiteList.append("hltBSoftMuonDiJet40L1FastJetMu5SelL3TagInfos")
prodWhiteList.append("hltDisplacedHT250L1FastJetL25JetTags")
prodWhiteList.append("hltDisplacedHT250L1FastJetL25TagInfos")
prodWhiteList.append("hltDisplacedHT250L1FastJetL25Associator")
prodWhiteList.append("hltDisplacedHT250L1FastJetL3JetTags")
prodWhiteList.append("hltDisplacedHT250L1FastJetL3TagInfos")
prodWhiteList.append("hltDisplacedHT250L1FastJetL3Associator")
prodWhiteList.append("hltMuTrackJpsiPixelTrackSelector")
prodWhiteList.append("hltMuTrackJpsiPixelTrackCands")
prodWhiteList.append("hltFastPVJetTracksAssociator")
prodWhiteList.append("hltBLifetimeL3BJetTagsbbPhiL1FastJetFastPV")
prodWhiteList.append("hltBLifetimeL3TagInfosbbPhiL1FastJetFastPV")
prodWhiteList.append("hltBLifetimeL3AssociatorbbPhiL1FastJetFastPV")
prodWhiteList.append("hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV")
prodWhiteList.append("hltBSoftMuonJet300L1FastJetMu5SelL3BJetTagsByDR")
prodWhiteList.append("hltBSoftMuonJet300L1FastJetMu5SelL3TagInfos")
prodWhiteList.append("hltMuPFTauLooseIsolationDiscriminator")
prodWhiteList.append("hltElePFTauLooseIsolationDiscriminator")
prodWhiteList.append("hltIsoElePFTauLooseIsolationDiscriminator")
prodWhiteList.append("hltIsoMuPFTauLooseIsolationDiscriminator")
prodWhiteList.append("hltFastPVJetTracksAssociator")
prodWhiteList.append("hltESPTrackCounting3D2nd")
prodWhiteList.append("hltFastPVPixelVertices3D")
prodWhiteList.append("hltBLifetimeL25AssociatorbbPhiL1FastJetFastPV")
prodWhiteList.append("hltBLifetimeL25TagInfosbbPhiL1FastJetFastPV")
prodWhiteList.append("hltBLifetimeL25BJetTagsbbPhiL1FastJetFastPV")
prodWhiteList.append("hltSelector4JetsL1FastJet")
prodWhiteList.append("hltSelectorJets20L1FastJet")
prodWhiteList.append("hltCombinedSecondaryVertex")
prodWhiteList.append("hltFastPVPixelVertices")
prodWhiteList.append("hltFastPixelBLifetimeL3AssociatorHbb")
prodWhiteList.append("hltFastPixelBLifetimeL3TagInfosHbb")
prodWhiteList.append("hltL3SecondaryVertexTagInfos")
prodWhiteList.append("hltL3CombinedSecondaryVertexBJetTags")
prodWhiteList.append("hltBLifetime3D1stTrkL25BJetTagsJet20HbbL1FastJet")
prodWhiteList.append("hltBLifetimeL25BJetTagsbbPhi1stTrackL1FastJetFastPV")
prodWhiteList.append("hltBLifetimeL25BJetTagsHbbVBF")
prodWhiteList.append("hltCombinedSecondaryVertexL25BJetTagsHbbVBF")
prodWhiteList.append("hltDisplacedHT300L1FastJetL25JetTags")
prodWhiteList.append("hltL2TauPixelIsoTagProducer")
prodWhiteList.append("hltBSoftMuonJet300L1FastJetL25BJetTagsByDR")
prodWhiteList.append("hltBSoftMuonJet300L1FastJetL25TagInfos")
prodWhiteList.append("hltEleGetJetsfromBPFNoPUJet30Central")
prodWhiteList.append("hltEleBLifetimeL3BPFNoPUJetTagsSingleTop")
prodWhiteList.append("hltEleBLifetimeL3PFNoPUTagInfosSingleTop")
prodWhiteList.append("hltEleBLifetimeL3PFNoPUAssociatorSingleTop")
prodWhiteList.append("hltMu17BLifetimeL3BPFNoPUJetTagsSingleTop")
prodWhiteList.append("hltMu17BLifetimeL3PFNoPUTagInfosSingleTop")
prodWhiteList.append("hltMu17BLifetimeL3PFNoPUAssociatorSingleTop")
prodWhiteList.append("hltMu17BLifetimeL3PFNoPUTagInfosSingleTopNoIso")
prodWhiteList.append("hltMu17BLifetimeL3PFNoPUAssociatorSingleTopNoIso")
prodWhiteList.append("hltMu17BLifetimeL3BPFNoPUJetTagsSingleTopNoIso")

prodTypeWhiteList=[]
prodTypeWhiteList.append("HLTPFJetCollectionsForLeptonPlusJets")

pathBlackList=[]
pathBlackList.append("HLT_BeamHalo_v10")
pathBlackList.append("HLT_IsoTrackHE_v12")
pathBlackList.append("HLT_IsoTrackHB_v11")
pathBlackList.append("DQM_FEDIntegrity_v7")

filterBlackList=[]

## Invalid Reference:

## Product Not Found

if runProducers==False:
    for pathName in process.pathNames().split():
        path = getattr(process,pathName)
        for moduleName in path.moduleNames():
            if moduleName in filterBlackList:
                notAllCopiesRemoved=True
                while notAllCopiesRemoved:
                    notAllCopiesRemoved = path.remove(getattr(process,moduleName))

for pathName in process.pathNames().split():
    if pathName in pathBlackList:
        path = getattr(process,pathName)
        for moduleName in path.moduleNames():
            notAllCopiesRemoved=True
            while notAllCopiesRemoved:
                notAllCopiesRemoved = path.remove(getattr(process,moduleName))
        

if runProducers==False:
    for path in process.pathNames().split():
       # print path
        for producer in process.producerNames().split():
            if producer not in prodWhiteList:
                if getattr(process,producer).type_() not in prodTypeWhiteList:
                    notAllCopiesRemoved=True
                    #print producer
                    while notAllCopiesRemoved:
                        notAllCopiesRemoved = getattr(process,path).remove(getattr(process,producer))


def findFiltersAlreadyIgnored(path): #there has got to be a better way...
    filtersAlreadyIgnored=[]
    pathSeq= path.dumpPython(options=cms.Options())
    for module in pathSeq.split("+"):
       # print "mod one ",module
        if module.startswith("cms.ignore"):
            module=module.lstrip("cms.ignore")
            module=module.lstrip("(")
            module=module.rstrip(")");
            module=module.lstrip("process.")
            filtersAlreadyIgnored.append(module)
        #    print module
    return filtersAlreadyIgnored

if runOpen:
    for pathName in process.pathNames().split():
        path = getattr(process,pathName)
        filtersAlreadyIgnored=[]
        filtersAlreadyIgnored=findFiltersAlreadyIgnored(path)
        for filterName in path.moduleNames():
            filt = getattr(process,filterName)
            if type(filt).__name__=="EDFilter":
                if filterName not in filtersAlreadyIgnored:
                    path.replace(filt,cms.ignore(filt))



            
    
def uniq(input):
    output = []
    for x in input:
        if x not in output:
            output.append(x)
    return output

def cleanList(input,blacklist):
    output = []
    for x in input:
        if x not in blacklist:
            output.append(x)
    return output

productsToKeep = []
for pathName in process.pathNames().split():
    path = getattr(process,pathName)
    for filterName in path.moduleNames():
        filt = getattr(process,filterName)
        #print filt.type_()
        if type(filt).__name__=="EDFilter":
            #print filterName
            for paraName in filt.parameters_():
                para = filt.getParameter(paraName)
                if type(para).__name__=="InputTag":
                    if para.getModuleLabel()!="":
                        productsToKeep.append(para.getModuleLabel())
                    #print paraName,type(para).__name__,para.getModuleLabel()
                if type(para).__name__=="VInputTag":
                    for tag in para:
                        if tag!="":
                            productsToKeep.append(tag)
                
                    

productsToKeep = uniq(productsToKeep)
productsToKeep = cleanList(productsToKeep,process.filterNames().split())

productsToKeep.append("hltAlCaEtaRecHitsFilterEBonly")
productsToKeep.append("hltBLifetimeL25TagInfosbbPhiL1FastJet")
productsToKeep.append("hltBLifetimeL25*")
productsToKeep.append("hltPixelVertices3DbbPhi")
productsToKeep.append("hltBLifetimeL25AssociatorbbPhiL1FastJet")
productsToKeep.append("hltPixelTracks")
productsToKeep.append("hltL2MuonSeeds")
productsToKeep.append("hltL2OfflineMuonSeeds")
productsToKeep.append("hltL3Muons")
productsToKeep.append("hltL3MuonsLinksCombination")
productsToKeep.append("hltL3TkTracksFromL2")
productsToKeep.append("hltL3TrajSeedOIState")
productsToKeep.append("hltL3TrackCandidateFromL2OIState")
productsToKeep.append("hltL3TkTracksFromL2OIState")
productsToKeep.append("hltL3MuonsOIState")
productsToKeep.append("hltL3TrajSeedOIHit")
productsToKeep.append("hltL3TrackCandidateFromL2OIHit")
productsToKeep.append("hltL3TkTracksFromL2OIHit")
productsToKeep.append("hltL3MuonsOIHit")
productsToKeep.append("hltL3TkFromL2OICombination")
productsToKeep.append("hltL3TrajSeedIOHit")
productsToKeep.append("hltL3TrackCandidateFromL2IOHit")
productsToKeep.append("hltL3TkTracksFromL2IOHit")
productsToKeep.append("hltL3MuonsIOHit")
productsToKeep.append("hltL3TrajectorySeed")
productsToKeep.append("hltL3TrackCandidateFromL2")
productsToKeep.append("hltDiMuonMerging")
productsToKeep.append("hltGlbTrkMuons")
productsToKeep.append("hltMuTrackJpsiPixelTrackSelector")
productsToKeep.append("hltCorrectedHybridSuperClustersL1Seeded")
productsToKeep.append("hltCorrectedMulti5x5EndcapSuperClustersWithPreshowerL1Seeded")
productsToKeep.append("hltMulti5x5BasicClustersL1Seeded")
productsToKeep.append("hltCorrectedHybridSuperClustersActivity")
productsToKeep.append("hltCorrectedMulti5x5SuperClustersWithPreshowerActivity")
productsToKeep.append("hltMulti5x5BasicClustersActivity")
productsToKeep.append("hltCorrectedHybridSuperClustersActivitySC4")
productsToKeep.append("hltCorrectedMulti5x5SuperClustersWithPreshowerActivitySC4")
productsToKeep.append("hltCtfL1SeededWithMaterialTracks")
productsToKeep.append("hltCtfL1SeededWithMaterialCleanTracks")
productsToKeep.append("hltCtf3HitL1SeededWithMaterialTracks")
productsToKeep.append("hltCtf3HitActivityWithMaterialTracks")
productsToKeep.append("hltCtfActivityWithMaterialTracks")
productsToKeep.append("hltPFTauTrackFindingDiscriminator")
productsToKeep.append("hltPFTauLooseIsolationDiscriminator")
productsToKeep.append("hltPFTauTrackPt20Discriminator")
productsToKeep.append("hltPFTauTrackPt5Discriminator")
productsToKeep.append("hltBSoftMuonDiJet20L1FastJetL25TagInfos")
productsToKeep.append("hltBSoftMuonDiJet20L1FastJetL25Jets")
productsToKeep.append("hltBSoftMuonGetJetsFromDiJet20L1FastJet")
productsToKeep.append("hltBDiJet20L1FastJetCentral")
productsToKeep.append("hltBSoftMuonDiJet40L1FastJetMu5SelL3TagInfos")
productsToKeep.append("hltCaloJetIDPassed")
productsToKeep.append("hltAntiKT5CaloJets")
productsToKeep.append("hltTowerMakerForAll")
productsToKeep.append("hltDisplacedHT250L1FastJetL25TagInfos")
productsToKeep.append("hltDisplacedHT250L1FastJetL25Associator")
productsToKeep.append("hltAntiKT5L2L3CorrCaloJetsL1FastJetPt60Eta2")
productsToKeep.append("hltDisplacedHT250L1FastJetRegionalCtfWithMaterialTracks")
productsToKeep.append("hltMuTrackJpsiEffCtfTracks")
productsToKeep.append("hltBSoftMuonDiJet70L1FastJetL25Jets")
productsToKeep.append("hltPFTauTagInfo")
productsToKeep.append("hltPFTauJetTracksAssociator")
productsToKeep.append("hltPFMuonMerging")
productsToKeep.append("hltBSoftMuonDiJet110L1FastJetL25Jets")
productsToKeep.append("hltBSoftMuonDiJet110L1FastJetL25TagInfos")
productsToKeep.append("hltBSoftMuonDiJet110L1FastJetL25BJetsTagsByDR")
productsToKeep.append("hltBSoftMuon300L1FastJetL25Jets")
productsToKeep.append("hltBSoftMuon300L1FastJetL25TagInfos")
productsToKeep.append("hltBSoftMuon300L1FastJetL25BJetTagsByDR")
productsToKeep.append("hltMuTrackJpsiCtfTracks")
productsToKeep.append("hltTrackTauPixelTrackCands")
productsToKeep.append("hltTrackTauRegionalPixelTrackSelector")
productsToKeep.append("hltCaloTowersTau1Regional")
productsToKeep.append("hltBSoftMuonDiJet40L1FastJetL25TagInfos")
productsToKeep.append("hltSecondaryVertexL25TagInfosHbbVBF")
productsToKeep.append("hltIter1ClustersRefRemoval")
productsToKeep.append("hltIter3PFJetMixedSeeds")
productsToKeep.append("hltBLifetimeRegionalCtfWithMaterialTracksHbbVBF")
productsToKeep.append("hltCtfWithMaterialTracksJpsiTk")
productsToKeep.append("hltHITCkfTrackCandidatesHE")
productsToKeep.append("hltHITCkfTrackCandidatesHB")
productsToKeep.append("hltBSoftMuonDiJet70L1FastJetL25TagInfos")
productsToKeep.append("hltCaloTowersTau2Regional")
productsToKeep.append("hltFEDSelector")
productsToKeep.append("hltParticleFlowClusterECAL")
productsToKeep.append("hltMulti5x5SuperClustersL1Seeded")
productsToKeep.append("hltIter4ClustersRefRemoval")
productsToKeep.append("hltEle32WP80CleanMergedTracks")
productsToKeep.append("hltIter4Tau3MuClustersRefRemoval")
productsToKeep.append("hltMuPFTauTrackFindingDiscriminator")
productsToKeep.append("hltIter1PFJetCkfTrackCandidates")
productsToKeep.append("hltDisplacedHT300L1FastJetRegionalPixelSeedGenerator")
productsToKeep.append("hltDisplacedHT300L1FastJetL25Associator")
productsToKeep.append("hltBLifetimeDiBTagIP3D1stTrkL3TagInfosJet20HbbL1FastJet")
productsToKeep.append("hltMuCtfTracks")
productsToKeep.append("hltTau3MuTrackSelectionHighPurity")
productsToKeep.append("hltKT6CaloJetsForMuons")
productsToKeep.append("hltEcalCalibrationRaw")
productsToKeep.append("hltSiPixelRecHits")
productsToKeep.append("hltParticleFlowRecHitPS")
productsToKeep.append("hltEle32WP80BarrelTracks")
productsToKeep.append("hltDisplacedHT300L1FastJetL3TagInfos")
productsToKeep.append("hltActivityElectronGsfTracks")
productsToKeep.append("hltEcalActivityEgammaRegionalCkfTrackCandidates")
productsToKeep.append("hltBSoftMuonDiJet110L1FastJetMu5SelL3TagInfos")
productsToKeep.append("hltIsoMuPFTauLooseIsolationDiscriminator")
productsToKeep.append("hltHcalDigis")
productsToKeep.append("hltPFPileUp")
productsToKeep.append("hltEcalRegionalEgammaFEDs")
productsToKeep.append("hltL1SeededEgammaRegionalCkfTrackCandidates")
productsToKeep.append("hltAlCaPhiSymUncalibrator")
productsToKeep.append("hltIter4Tau3MuCtfWithMaterialTracks")
productsToKeep.append("hltCscSegments")
productsToKeep.append("hltHITCtfWithMaterialTracksHB")
productsToKeep.append("hltIter3Tau3MuTrackSelectionHighPurityLoose")
productsToKeep.append("hltL1SeededGsfElectrons")
productsToKeep.append("hltIter1PFlowTrackSelectionHighPurityLoose")
productsToKeep.append("hltSiStripRawToClustersFacility")
productsToKeep.append("hltHITCtfWithMaterialTracksHE")
productsToKeep.append("hltBLifetimeRegionalPixelSeedGeneratorbbPhiL1FastJetFastPV")
productsToKeep.append("hltMuonLinks")
productsToKeep.append("hltIter2Tau3MuClustersRefRemoval")
productsToKeep.append("hltIter1Tau3MuTrackSelectionHighPurityTight")
productsToKeep.append("hltHybridSuperClustersActivity")
productsToKeep.append("hltCkfL1SeededTrackCandidates")
productsToKeep.append("hltBSoftMuonDiJet70L1FastJetMu5SelL3TagInfos")
productsToKeep.append("hltIter3PFJetCtfWithMaterialTracks")
productsToKeep.append("hltIconeCentral4Regional")
productsToKeep.append("hltRegionalTracksForL3MuonIsolation")
productsToKeep.append("hltPFMuonMergingPromptTracks")
productsToKeep.append("hltIter1Merged")
productsToKeep.append("hltPFTauMediumIsolationDiscriminator")
productsToKeep.append("hltBSoftMuonJet300L1FastJetMu5SelL3TagInfos")
productsToKeep.append("hltIter2Tau3MuCtfWithMaterialTracks")
productsToKeep.append("hltIter2PFJetCtfWithMaterialTracks")
productsToKeep.append("hltElePFTauTagInfo")
productsToKeep.append("hltEcalRawToRecHitFacility")
productsToKeep.append("hltIter3PFlowTrackSelectionHighPurityTight")
productsToKeep.append("hltAntiKT5TrackJetsIter0")
productsToKeep.append("hltPFJetCkfTrackCandidates")
productsToKeep.append("hltElePFTauLooseIsolationDiscriminator")
productsToKeep.append("hltIter1PFJetPixelSeeds")
productsToKeep.append("hltPFNoPileUp")
productsToKeep.append("hltTriggerSummaryRAW")
productsToKeep.append("hltParticleFlowClusterPS")
productsToKeep.append("hltEcalActivityEgammaRegionalAnalyticalTrackSelectorHighPurity")
productsToKeep.append("hltFastPixelBLifetimeL3TagInfosHbb")
productsToKeep.append("hltBLifetimeRegionalPixelSeedGeneratorHbbVBF")
productsToKeep.append("hltIter1Tau3MuPixelSeeds")
productsToKeep.append("hltBLifetimeL3AssociatorbbPhiL1FastJetFastPV")
productsToKeep.append("hltCaloTowersTau4Regional")
productsToKeep.append("hltIter3Tau3MuTrackSelectionHighPurityTight")
productsToKeep.append("hltAntiKT5CaloJetsRegional")
productsToKeep.append("hltCsc2DRecHits")
productsToKeep.append("hltBLifetimeRegionalCtfWithMaterialTracksbbPhiL1FastJetFastPV")
productsToKeep.append("hltBLifetimeFastL25AssociatorHbbVBF")
productsToKeep.append("hltMediumPFTauTrackPt1Discriminator")
productsToKeep.append("hltEle15CaloIdTTrkIdTCaloIsoVLTrkIsoVLPFJetCollForElePlusJetsNoPU")
productsToKeep.append("hltCaloTowersTau3Regional")
productsToKeep.append("hltBSoftMuonJet300L1FastJetL25TagInfos")
productsToKeep.append("hltESRecHitAll")
productsToKeep.append("hltIter4PFlowTrackSelectionHighPurity")
productsToKeep.append("hltEcalRawToRecHitByproductProducer")
productsToKeep.append("hltBLifetimeFastRegionalCtfWithMaterialTracksHbbVBF")
productsToKeep.append("hltBLifetimeDiBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet")
productsToKeep.append("hltCkfActivityTrackCandidates")
productsToKeep.append("hltMuCkfTrackCandidates")
productsToKeep.append("hltBLifetimeL3TagInfosHbbVBF")
productsToKeep.append("hltParticleFlowClusterHFHAD")
productsToKeep.append("hltFastPVPixelVertices3D")
productsToKeep.append("hltCaloJetIDPassedRegional")
productsToKeep.append("hltHoreco")
productsToKeep.append("hltEcalRegionalPi0EtaFEDs")
productsToKeep.append("hltSecondaryVertexL3TagInfosHbbVBF")
productsToKeep.append("hltEle27WP80CleanMergedTracks")
productsToKeep.append("hltTau3MuCkfTrackCandidates")
productsToKeep.append("hltAntiKT5TrackJetsIter3")
productsToKeep.append("hltAntiKT5TrackJetsIter2")
productsToKeep.append("hltAntiKT5TrackJetsIter1")
productsToKeep.append("hltRegionalCandidatesForL3MuonIsolation")
productsToKeep.append("hltStoppedHSCPTowerMakerForAll")
productsToKeep.append("hltBLifetimeL3AssociatorbbHbbVBF")
productsToKeep.append("hltFastPixelBLifetimeRegionalPixelSeedGeneratorHbb")
productsToKeep.append("hltAlCaEtaEEUncalibrator")
productsToKeep.append("hltTrackAndTauJetsIter3")
productsToKeep.append("hltTrackAndTauJetsIter2")
productsToKeep.append("hltTrackAndTauJetsIter1")
productsToKeep.append("hltTrackAndTauJetsIter0")
productsToKeep.append("hltIter3Tau3MuMerged")
productsToKeep.append("hltMuTrackJpsiEffCkfTrackCandidates")
productsToKeep.append("hltTrackTauRegPixelTrackSelector")
productsToKeep.append("hltIsoEleVertex")
productsToKeep.append("hltParticleFlowClusterHFEM")
productsToKeep.append("hltIter1PFlowTrackSelectionHighPurityTight")
productsToKeep.append("hltBLifetimeRegionalCkfTrackCandidatesHbbVBF")
productsToKeep.append("hltBSoftMuonDiJet20L1FastJetMu5SelL3TagInfos")
productsToKeep.append("hltBLifetime3DL25TagInfosJet20HbbL1FastJet")
productsToKeep.append("hltAlCaEtaEBUncalibrator")
productsToKeep.append("hltDisplacedHT300L1FastJetRegionalCkfTrackCandidates")
productsToKeep.append("hltL1SeededEgammaRegionalAnalyticalTrackSelectorHighPurity")
productsToKeep.append("hltSiPixelClustersReg")
productsToKeep.append("hltMuonCSCDigis")
productsToKeep.append("hltMuonRPCDigis")
productsToKeep.append("hltIter1PFlowTrackSelectionHighPurity")
productsToKeep.append("hltFastPVPixelTracksRecover")
productsToKeep.append("hltDisplacedHT300L1FastJetRegionalCtfWithMaterialTracks")
productsToKeep.append("hltIter2PFlowTrackSelectionHighPurity")
productsToKeep.append("hltEcalRegionalJetsRecHit")
productsToKeep.append("hltDTCalibrationRaw")
productsToKeep.append("hltIsoMuPFTauTrackFindingDiscriminator")
productsToKeep.append("hltIter3Merged")
productsToKeep.append("hltIconeCentral2Regional")
productsToKeep.append("hltGctDigis")
productsToKeep.append("hltL1SeededCkfTrackCandidatesForGSF")
productsToKeep.append("hltIter3ClustersRefRemoval")
productsToKeep.append("hltActivityCkfTrackCandidatesForGSF")
productsToKeep.append("hltFastPrimaryVertex")
productsToKeep.append("hltMuTrackJpsiTrackSeeds")
productsToKeep.append("hltIter3PFJetCkfTrackCandidates")
productsToKeep.append("hltGoodPixelTracksForHighMult")
productsToKeep.append("hltBLifetimeFastRegionalPixelSeedGeneratorHbbVBF")
productsToKeep.append("hltMu17BLifetimeL3PFNoPUAssociatorSingleTopNoIso")
productsToKeep.append("hltIter1Tau3MuClustersRefRemoval")
productsToKeep.append("hltSiStripExcludedFEDListProducer")
productsToKeep.append("hltDt1DRecHits")
productsToKeep.append("hltTau3MuCtfWithMaterialTracks")
productsToKeep.append("hltIter2Tau3MuMerged")
productsToKeep.append("hltIter4Tau3MuMerged")
productsToKeep.append("hltIsoMuPFTauTagInfo")
productsToKeep.append("hltCaloTowersCentral2Regional")
productsToKeep.append("hltIter3Tau3MuTrackSelectionHighPurity")
productsToKeep.append("hltIconeCentral1Regional")
productsToKeep.append("hltIter1Tau3MuCkfTrackCandidates")
productsToKeep.append("hltIsoElePFTauTrackFindingDiscriminator")
productsToKeep.append("hltFastPVPixelTracksMerger")
productsToKeep.append("hltCaloTowersCentral1Regional")
productsToKeep.append("hltTrackTauRegPixelTrackCands")
productsToKeep.append("hltBLifetimeBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet")
productsToKeep.append("hltMuTrackSeeds")
productsToKeep.append("hltRegionalPixelTracks")
productsToKeep.append("hltIconeTau4Regional")
productsToKeep.append("hltCorrectedMulti5x5SuperClustersWithPreshowerActivitySC5")
productsToKeep.append("hltIter2Tau3MuPixelSeeds")
productsToKeep.append("hltTowerMakerForPF")
productsToKeep.append("hltL1SeededEgammaRegionalPixelSeedGenerator")
productsToKeep.append("hltHITPixelTracksHE")
productsToKeep.append("hltEle5CaloIdTTrkIdTCaloIsoVLTrkIsoVLPFJetCollForElePlusJetsNoPU")
productsToKeep.append("hltEleBLifetimeL3PFNoPUAssociatorSingleTop")
productsToKeep.append("hltIter1Tau3MuMerged")
productsToKeep.append("hltEcalActivityEgammaRegionalCTFFinalFitWithMaterial")
productsToKeep.append("hltCkf3HitL1SeededTrackCandidates")
productsToKeep.append("hltIsoElePFTauTagInfo")
productsToKeep.append("hltParticleFlowRecHitECAL")
productsToKeep.append("hltIter2PFJetPixelSeeds")
productsToKeep.append("hltIter1Tau3MuCtfWithMaterialTracks")
productsToKeep.append("hltBLifetimeFastL25TagInfosHbbVBF")
productsToKeep.append("hltParticleFlowBlockPromptTracks")
productsToKeep.append("hltMulti5x5SuperClustersActivity")
productsToKeep.append("hltHFEMClusters")
productsToKeep.append("hltEle27WP80BarrelTracks")
productsToKeep.append("hltL1SeededElectronGsfTracks")
productsToKeep.append("hltTrackerCalibrationRaw")
productsToKeep.append("hltDisplacedHT300L1FastJetL25TagInfos")
productsToKeep.append("hltBLifetimeBTagIP3D1stTrkL3TagInfosJet20HbbL1FastJet")
productsToKeep.append("hltMulti5x5SuperClustersWithPreshowerActivity")
productsToKeep.append("hltEcalRegionalESRestFEDs")
productsToKeep.append("hltEcalActivityEgammaRegionalPixelSeedGenerator")
productsToKeep.append("hltIter3Tau3MuCkfTrackCandidates")
productsToKeep.append("hltIconeTau2Regional")
productsToKeep.append("hltDTDQMEvF")
productsToKeep.append("hltMuons")
productsToKeep.append("hltIter2Merged")
productsToKeep.append("hltPixelVerticesReg")
productsToKeep.append("hltL3CaloMuonCorrectedIsolations")
productsToKeep.append("hltIter2SiStripClusters")
productsToKeep.append("hltIconeTau1Regional")
productsToKeep.append("hltPixelTracksForHighMult")
productsToKeep.append("hltBLifetimeDiBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet")
productsToKeep.append("hltParticleFlowClusterHCAL")
productsToKeep.append("hltParticleFlowPromptTracks")
productsToKeep.append("hltBLifetimeFastL3AssociatorbbHbbVBF")
productsToKeep.append("hltHcalCalibrationRaw")
productsToKeep.append("hltCkfTrackCandidatesJpsiTk")
productsToKeep.append("hltIter3Tau3MuClustersRefRemoval")
productsToKeep.append("hltJpsiTkPixelSeedFromL3Candidate")
productsToKeep.append("hltMuonDTDigis")
productsToKeep.append("hltPFlowTrackSelectionHighPurity")
productsToKeep.append("hltAlCaPi0EEUncalibrator")
productsToKeep.append("hltIter1PFJetCtfWithMaterialTracks")
productsToKeep.append("hltSiPixelDigis")
productsToKeep.append("hltIter2Tau3MuTrackSelectionHighPurity")
productsToKeep.append("hltTrackRefsForJetsIter2")
productsToKeep.append("hltTrackRefsForJetsIter3")
productsToKeep.append("hltTrackRefsForJetsIter0")
productsToKeep.append("hltTowerMakerForMuons")
productsToKeep.append("hltHcalNoiseInfoProducer")
productsToKeep.append("hltTau3MuPixelSeedsFromPixelTracks")
productsToKeep.append("hltIter4SiStripClusters")
productsToKeep.append("hltIter3SiStripClusters")
productsToKeep.append("hltElePFTauTrackFindingDiscriminator")
productsToKeep.append("hltIter1Tau3MuTrackSelectionHighPurityLoose")
productsToKeep.append("hltMu17BLifetimeL3PFNoPUTagInfosSingleTopNoIso")
productsToKeep.append("hltHITPixelVerticesHE")
productsToKeep.append("hltIter1SiStripClusters")
productsToKeep.append("hltFastPVPixelTracks")
productsToKeep.append("hltIter2Tau3MuSiStripClusters")
productsToKeep.append("hltIconeTau3Regional")
productsToKeep.append("hltMuPFTauLooseIsolationDiscriminator")
productsToKeep.append("hltIter3Tau3MuMixedSeeds")
productsToKeep.append("hltKT6CaloJets")
productsToKeep.append("hltIsoElePFTauLooseIsolationDiscriminator")
productsToKeep.append("hltBLifetimeBTagIP3D1stTrkRegionalCkfTrackCandidatesJet20HbbL1FastJet")
productsToKeep.append("hltParticleFlowBlock")
productsToKeep.append("hltBLifetimeRegionalCkfTrackCandidatesbbPhiL1FastJetFastPV")
productsToKeep.append("hltBLifetimeBTagIP3D1stTrkL3AssociatorJet20HbbL1FastJet")
productsToKeep.append("hltCaloTowersCentral4Regional")
productsToKeep.append("hltDiMuonLinks")
productsToKeep.append("hltMulti5x5EndcapSuperClustersWithPreshowerL1Seeded")
productsToKeep.append("hltFastPixelBLifetimeL3AssociatorHbb")
productsToKeep.append("hltIter4PFJetCkfTrackCandidates")
productsToKeep.append("hltLightPFPromptTracks")
productsToKeep.append("hltEcalRegionalJetsFEDs")
productsToKeep.append("hltBLifetimeBTagIP3D1stTrkRegionalCtfWithMaterialTracksJet20HbbL1FastJet")
productsToKeep.append("hltIconeCentral3Regional")
productsToKeep.append("hltIter3PFlowTrackSelectionHighPurityLoose")
productsToKeep.append("hltIter4Tau3MuCkfTrackCandidates")
productsToKeep.append("hltEle60CaloIdVTTrkIdTPFJetCollForElePlusJetsNoPU")
productsToKeep.append("hltIter4PFJetCtfWithMaterialTracks")
productsToKeep.append("hltBLifetimeDiBTagIP3D1stTrkRegionalPixelSeedGeneratorJet20HbbL1FastJet")
productsToKeep.append("hltPixelTracksReg")
productsToKeep.append("hltRegionalSeedsForL3MuonIsolation")
productsToKeep.append("hltCaloTowersCentral3Regional")
productsToKeep.append("hltBLifetimeFastL3TagInfosHbbVBF")
productsToKeep.append("hltIter3PFlowTrackSelectionHighPurity")
productsToKeep.append("hltBLifetimeDiBTagIP3D1stTrkL3AssociatorJet20HbbL1FastJet")
productsToKeep.append("hltEleBLifetimeL3PFNoPUTagInfosSingleTop")
productsToKeep.append("hltCorrectedHybridSuperClustersActivitySC5")
productsToKeep.append("hltSiPixelDigisReg")
productsToKeep.append("hltTowerMakerForJets")
productsToKeep.append("hltESRawToRecHitFacility")
productsToKeep.append("hltMu17BLifetimeL3PFNoPUAssociatorSingleTop")
productsToKeep.append("hltParticleFlowRecHitHCAL")
productsToKeep.append("hltSiPixelRecHitsReg")
productsToKeep.append("hltMuPFTauTagInfo")
productsToKeep.append("hltHITPixelTripletSeedGeneratorHB")
productsToKeep.append("hltCkf3HitActivityTrackCandidates")
productsToKeep.append("hltHITPixelTracksHB")
productsToKeep.append("hltL3SecondaryVertexTagInfos")
productsToKeep.append("hltHITPixelTripletSeedGeneratorHE")
productsToKeep.append("hltEle40CaloIdVTTrkIdTPFJetCollForElePlusJetsNoPU")
productsToKeep.append("hltFastPixelBLifetimeRegionalCtfWithMaterialTracksHbb")
productsToKeep.append("hltIter2ClustersRefRemoval")
productsToKeep.append("hltLightPFTracks")
productsToKeep.append("hltBLifetimeFastRegionalCkfTrackCandidatesHbbVBF")
productsToKeep.append("hltFastPixelBLifetimeRegionalCkfTrackCandidatesHbb")
productsToKeep.append("hltIter2PFJetCkfTrackCandidates")
productsToKeep.append("hltIter3Tau3MuSiStripClusters")
productsToKeep.append("hltMuTrackJpsiCkfTrackCandidates")
productsToKeep.append("hltMediumPFTauTrackFindingDiscriminator")
productsToKeep.append("hltPixelTracksForMinBias")
productsToKeep.append("hltDt4DSegments")
productsToKeep.append("hltPFJetCtfWithMaterialTracks")
productsToKeep.append("hltAlCaPi0EBUncalibrator")
productsToKeep.append("hltTrackRefsForJetsIter1")
productsToKeep.append("hltIter4PFJetPixelLessSeeds")
productsToKeep.append("hltESRegionalEgammaRecHit")
productsToKeep.append("hltDisplacedHT300L1FastJetL3Associator")
productsToKeep.append("hltCaloJetL1MatchedRegional")
productsToKeep.append("hltIter1Tau3MuTrackSelectionHighPurity")
productsToKeep.append("hltEcalRegionalMuonsFEDs")
productsToKeep.append("hltEleVertex")
productsToKeep.append("hltHcalTowerNoiseCleaner")
productsToKeep.append("hltHITPixelVerticesHB")
productsToKeep.append("hltHybridSuperClustersL1Seeded")
productsToKeep.append("hltMuonVertex")
productsToKeep.append("hltEcalRegionalRestFEDs")
productsToKeep.append("hltIter3Tau3MuCtfWithMaterialTracks")
productsToKeep.append("hltFEDSelectorLumiPixels")
productsToKeep.append("hltEcalRegionalMuonsRecHit")
productsToKeep.append("hltActivityGsfElectrons")
productsToKeep.append("hltIter4Tau3MuSiStripClusters")
productsToKeep.append("hltPFJetPixelSeedsFromPixelTracks")
productsToKeep.append("hltIter2Tau3MuCkfTrackCandidates")
productsToKeep.append("hltIter4Tau3MuPixelLessSeeds")
productsToKeep.append("hltIter1Tau3MuSiStripClusters")
productsToKeep.append("hltBLifetimeL3TagInfosbbPhiL1FastJetFastPV")
productsToKeep.append("hltIsoMuonVertex")
productsToKeep.append("hltMu17BLifetimeL3PFNoPUTagInfosSingleTop")
productsToKeep.append("hltIter4Tau3MuTrackSelectionHighPurity")

process.output.outputCommands=cms.untracked.vstring("drop *","keep *_TriggerResults_*_*",
                                                    "keep *_hltTriggerSummaryAOD_*_*")

for product in productsToKeep:
    process.output.outputCommands.append("keep *_"+product+"_*_*")

# version specific customizations
import os
cmsswVersion = os.environ['CMSSW_VERSION']

# ---- dump ----
dump = open('dump.py', 'w')
dump.write( process.dumpPython() )
dump.close()

# This is an example PAT configuration showing the usage of PAT on minbias data

# Starting with a skeleton process which gets imported with the following line
from PhysicsTools.PatAlgos.patTemplate_cfg import *

from PhysicsTools.PatAlgos.tools.coreTools import *
removeMCMatching(process, ['All'])

## global tag for data
process.GlobalTag.globaltag = 'GR_R_42_V14::All'


# Triggers for the /Jet PD in the used example run 165121 are from:
# /cdaq/physics/Run2011/1e33/v1.3/HLT/V2
# The list is here, let's take all of them :
## Jet = cms.vstring( 'HLT_DiJetAve110_v3',
##   'HLT_DiJetAve150_v3',
##   'HLT_DiJetAve190_v3',
##   'HLT_DiJetAve240_v3',
##   'HLT_DiJetAve300_v3',
##   'HLT_DiJetAve30_v3',
##   'HLT_DiJetAve370_v3',
##   'HLT_DiJetAve60_v3',
##   'HLT_DiJetAve80_v3',
##   'HLT_Jet110_v3',
##   'HLT_Jet150_v3',
##   'HLT_Jet190_v3',
##   'HLT_Jet240_v3',
##   'HLT_Jet300_v2',
##   'HLT_Jet30_v3',
##   'HLT_Jet370_NoJetID_v3',
##   'HLT_Jet370_v3',
##   'HLT_Jet60_v3',
##   'HLT_Jet80_v3' ),
mytrigs = ['*']

# Jet energy corrections to use:
#inputJetCorrLabel = ('AK5PF', ['L1Offset', 'L2Relative', 'L3Absolute', 'L2L3Residual'])
inputJetCorrLabel = ('AK5PF', ['L1Offset', 'L2Relative', 'L3Absolute'])

# add pf met
from PhysicsTools.PatAlgos.tools.metTools import *
addPfMET(process, 'PF')

# Add PF jets
from PhysicsTools.PatAlgos.tools.jetTools import *
switchJetCollection(process,cms.InputTag('ak5PFJets'),
                 doJTA        = True,
                 doBTagging   = True,
                 jetCorrLabel = inputJetCorrLabel,
                 doType1MET   = True,
                 genJetCollection=cms.InputTag("ak5GenJets"),
                 doJetID      = True
                 )
process.patJets.addTagInfos = True
process.patJets.tagInfoSources  = cms.VInputTag(
    cms.InputTag("secondaryVertexTagInfosAOD"),
    )

# Apply loose PF jet ID
from PhysicsTools.SelectorUtils.pfJetIDSelector_cfi import pfJetIDSelector
process.goodPatJets = cms.EDFilter("PFJetIDSelectionFunctorFilter",
                                   filterParams = pfJetIDSelector.clone(),
                                   src = cms.InputTag("selectedPatJets"),
                                   filter = cms.bool(True)
                                   )


# Taus are currently broken in 4.1.x
removeSpecificPATObjects( process, ['Taus'] )
process.patDefaultSequence.remove( process.patTaus )

# require physics declared
process.load('HLTrigger.special.hltPhysicsDeclared_cfi')
process.hltPhysicsDeclared.L1GtReadoutRecordTag = 'gtDigis'

# require scraping filter
process.scrapingVeto = cms.EDFilter("FilterOutScraping",
                                    applyfilter = cms.untracked.bool(True),
                                    debugOn = cms.untracked.bool(False),
                                    numtrack = cms.untracked.uint32(10),
                                    thresh = cms.untracked.double(0.2)
                                    )
# HB + HE noise filtering
process.load('CommonTools.RecoAlgos.HBHENoiseFilter_cfi')
process.HBHENoiseFilter.minIsolatedNoiseSumE        = 999999.
process.HBHENoiseFilter.minNumIsolatedNoiseChannels = 999999
process.HBHENoiseFilter.minIsolatedNoiseSumEt       = 999999.


from HLTrigger.HLTfilters.hltHighLevel_cfi import *
if mytrigs is not None :
    process.hltSelection = hltHighLevel.clone(TriggerResultsTag = 'TriggerResults::HLT', HLTPaths = mytrigs)
    process.hltSelection.throw = False

from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
goodVertexSelection = pvSelector.clone( maxZ = 24. )
process.primaryVertexFilter = cms.EDFilter("PrimaryVertexFilter",
                                           goodVertexSelection
                                           )


# Select jets
process.selectedPatJets.cut = cms.string('pt > 25')

# Add the files
readFiles = cms.untracked.vstring()
secFiles = cms.untracked.vstring()


readFiles.extend( [
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/E4D2CB53-9881-E011-8D99-003048F024DC.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/DC453887-7481-E011-89B9-001617E30D0A.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/DABFB9E8-9B81-E011-9FF7-0030487CD812.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/D4E6F338-9B81-E011-A8DE-003048F110BE.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/CEC22BF3-9681-E011-AA91-003048CFB40C.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/BC737791-9581-E011-90E2-0030487CD6DA.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/B6E2792E-8881-E011-9C34-000423D9A212.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/A47B3EF5-9D81-E011-AC11-003048F024FA.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/84253FEF-9B81-E011-9DB7-001617DBD230.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/80749EFD-AB81-E011-A6E3-0030487CD76A.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/749CCE90-9581-E011-862A-000423D9A212.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/6CDB388A-9681-E011-B324-0030487CD178.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/464C7F95-9581-E011-BAEC-0030487CAEAC.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/42B01B46-9D81-E011-AF75-003048F1C58C.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/36C0F1A9-1382-E011-A5B5-003048F11942.root',
        '/store/data/Run2011A/Jet/AOD/PromptReco-v4/000/165/121/10691D45-9D81-E011-9A9A-003048F118C2.root',
 ] )

process.source.fileNames = readFiles

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

# let it run

process.p = cms.Path(
    process.hltSelection*
    process.scrapingVeto*
    process.primaryVertexFilter*
    process.HBHENoiseFilter*
    process.patDefaultSequence*
    process.goodPatJets
    )

# rename output file
process.out.fileName = cms.untracked.string('jet2011A_aod.root')

# reduce verbosity
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

# process all the events
process.maxEvents.input = 1000
process.options.wantSummary = True

from PhysicsTools.PatAlgos.patEventContent_cff import patEventContentNoCleaning
from PhysicsTools.PatAlgos.patEventContent_cff import patExtraAodEventContent
process.out.outputCommands = patEventContentNoCleaning
process.out.outputCommands += patExtraAodEventContent
process.out.outputCommands += [
    'drop patJets_selectedPatJets_*_*',
    'keep patJets_goodPatJets_*_*',
    'keep recoPFCandidates_selectedPatJets*_*_*'
    ]

# switch on PAT trigger
from PhysicsTools.PatAlgos.tools.trigTools import switchOnTrigger
switchOnTrigger( process )
process.patTrigger.addL1Algos = cms.bool( True )
switchOnTrigger( process ) # to fix event content

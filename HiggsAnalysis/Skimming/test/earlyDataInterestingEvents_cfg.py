from FWCore.ParameterSet import Config as cms

process = cms.Process("Skim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

### Load cff file
process.load("HiggsAnalysis.Skimming.earlyDataInterestingEvents_cff")

### Define source and max events 
## -------------------------------
process.source = cms.Source("PoolSource",  
    fileNames = cms.untracked.vstring(
        # Files also available under /castor/cern.ch/user/g/gpetrucc/7TeV/DATA/, but xroot is faster and it's in any case accessible from everywhere at cern
        #'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1324xx.root', ## CMSSW_3_5_4; can merge only if you add in inputCommands 'drop recoPFRecHits_*_*_*', 'drop *_trackerDrivenElectronSeeds_*_*'
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1325xx.root',
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1326xx.root',
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1327xx.root',
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1329xx.root',
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1330xx.root',
        'root://pcmssd12.cern.ch//data/gpetrucc/7TeV/Cocktail_runs1331xx.root',
    ),
    inputCommands = cms.untracked.vstring(
        'drop *', 'keep *_*_*_HLT', 'keep *_*_*_EXPRESS', 'keep *_*_*_RECO',  # drop things from previous skims, keep only "official" data
        'drop *_MEtoEDMConverter_*_*', 'drop *_l1GtTriggerMenuLite_*_*',      # drop per-run or per-lumi stuff that takes a lot of space even if the run/lumi doesn't contain events
        #'drop recoPFRecHits_*_*_*', 'drop *_trackerDrivenElectronSeeds_*_*'  # drop incompatible stuff from past releases
   ),
   dropDescendantsOfDroppedBranches = cms.untracked.bool(False)               # this is necessary when dropping incompatible stuff without dropping everything
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

### Define output module and endpath
## -------------------------------
process.out = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string("interesting.root"),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring())
)
process.end = cms.EndPath(process.out)

### Define skim paths
## -------------------------------
## make a function to avoid all the boring typing over and over
def mkSkimPath(label, saveEvents=True):
    filter = cms.EDFilter("CandViewCountFilter", src = cms.InputTag(label), minNumber = cms.uint32(1));
    setattr(process,label+"Filter", filter)
    if saveEvents:
        setattr(process,"Skim_"+label, cms.Path(process.earlyDataInterestingEvents + filter))
        process.out.SelectEvents.SelectEvents += [ 'Skim_' + label ]
    else:
        setattr(process,"Count_"+label, cms.Path(process.earlyDataInterestingEvents + filter))
## Pre-define a path that just count the events passing the prefilter 
process.Count_Events = cms.Path(process.earlyDataPreFilter)
## Then add all the other paths 
mkSkimPath('highEnergyMuons')
mkSkimPath('highEnergyElectrons')
mkSkimPath('diMuons')
mkSkimPath('diMuonsJPsi')
mkSkimPath('diMuonsZ')
mkSkimPath('diElectrons')
mkSkimPath('diElectronsJPsi') 
mkSkimPath('diElectronsZ')
mkSkimPath('crossLeptons')
mkSkimPath('recoWMNfromPf')
mkSkimPath('recoWMNfromTc')
mkSkimPath('recoWENfromPf')
mkSkimPath('recoWENfromTc')
for X in ("MuMuMu", "MuMuEl", "MuElEl", "ElElEl"): mkSkimPath('triLeptons'+X)
for X in ("4Mu", "2Mu2El", "4El"): mkSkimPath('quadLeptons'+X)

## To test just a single path
# process.out.SelectEvents.SelectEvents = [ 'Skim_diElectronsJPsi' ]

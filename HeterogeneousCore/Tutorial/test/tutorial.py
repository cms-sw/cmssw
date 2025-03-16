import FWCore.ParameterSet.Config as cms
  
process = cms.Process("TUTORIAL")

# enable multithreading
process.options.numberOfThreads = 8
process.options.numberOfStreams = 0

# enable alpaka and GPU support
process.load("Configuration.StandardSequences.Accelerators_cff")

# run over recent RelVal samples
from IOPool.Input.modules import PoolSource
process.source = PoolSource(
    fileNames = [
        # dasgoclient --query 'file dataset=/RelValTTbar_14TeV/CMSSW_15_0_0-PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/GEN-SIM-DIGI-RAW' | head -n 10
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/749ab261-6527-4e7f-b57a-08b7118954a8.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/1c2caeef-e246-4b6d-bebc-4fb6df4f9bbd.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/b000989f-2100-4550-a776-1e10f9ecfbad.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/92857a1a-c082-4092-895a-01557f3194a2.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/a383e5fc-3bd7-4a73-95ef-240761cc60b2.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/d3e3e977-a027-4afa-a026-fc4ba8f9ac37.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/8c24243d-e4a9-43d8-96fb-8d61b6eb6ab1.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/204ee874-198f-4ba6-ab18-32164a3e0e59.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/c699d6a4-3651-4c51-bc2c-258d4f86a2cd.root",
        "/store/relval/CMSSW_15_0_0/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_142X_mcRun3_2025_realistic_v7_STD_2025_PU-v3/2580000/994fc44d-b234-44f7-a98b-bc9c83be2a84.root",
    ])

# process only 1000 events
process.maxEvents.input = 100

# print a message every event
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# do not print the time and trigger reports at the end of the job
process.options.wantSummary = False

# configure the global tag
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(None, globaltag = '142X_mcRun3_2025_realistic_v7')

# import the definition of the Tutorial modules
from HeterogeneousCore.Tutorial.modules import *

# convert PFJets to SoA format
process.pfJetsSoA = tutorial_PFJetsSoAProducer_alpaka(
    jets = "hltAK4PFJetsCorrected"
)

# or, use the underlying syntax
# process.pfJetsSoA = cms.EDProducer('tutorial::PFJetsSoAProducer@alpaka',
#     jets = cms.InputTag("hltAK4PFJetsCorrected")
# )

# produce the corrections in the EventSetup
from FWCore.Modules.modules import EmptyESSource
process.SoACorrectorRecord = EmptyESSource(
    recordName = "tutorial::SoACorrectorRecord",
    firstValid = 1
)

process.SoACorrectorESProducer = tutorial_SoACorrectorESProducer_alpaka()

# apply the corrections
process.pfJetsSoACorrected = tutorial_PFJetsSoACorrector_alpaka(
    jets = "pfJetsSoA"
)

# select pairs and triplets of corrected jets with an invariant mass within the given range
process.invariantMassSelector = tutorial_InvariantMassSelector_alpaka(
    jets = "pfJetsSoACorrected",
    pT_min = 20.,    # GeV
    pT_max = 300.,   # GeV
    eta_min = 0.,
    eta_max = 3.,
    mass_min = 75.,  # GeV
    mass_max = 105., # GeV
)
process.MessageLogger.cerr.InvariantMassSelector = cms.untracked.PSet()

# dump the PF jets and SoA jets
process.pfJetsSoAAnalyzer = tutorial_PFJetsSoAAnalyzer(
    jets = "hltAK4PFJetsCorrected",
    soa = "pfJetsSoACorrected",
    ntuplets = "invariantMassSelector"
)
process.MessageLogger.cerr.PFJetsSoAAnalyzer = cms.untracked.PSet()

# schedule the modules
process.path = cms.Path(
    process.pfJetsSoA +
    process.pfJetsSoACorrected +
    process.invariantMassSelector +
    process.pfJetsSoAAnalyzer
)

process.schedule = cms.Schedule(
    process.path
)

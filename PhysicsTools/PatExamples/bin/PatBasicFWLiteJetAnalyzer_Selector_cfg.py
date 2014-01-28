import FWCore.ParameterSet.Config as cms

process = cms.Process("FWLitePlots")

#input stuff for Run/Lumi selection with the "JSON"-formatted files from the PVT group
import FWCore.PythonUtilities.LumiList as LumiList


# setup process
process = cms.Process("FWLitePlots")

# get JSON file correctly parced
JSONfile = 'DCSTRONLY_132440-140388'
myList = LumiList.LumiList (filename = JSONfile).getCMSSWString().split(',')


# Set up the parameters for the calo jet analyzer
process.jetStudies = cms.PSet(
    # input parameter sets
    jetSrc = cms.InputTag('selectedPatJets'),
    pfJetSrc = cms.InputTag('selectedPatJetsAK5PF'),
    metSrc = cms.InputTag('patMETs'),
    pfMetSrc = cms.InputTag('patMETsPF'),
    useCalo = cms.bool(True)
)

# Set up the parameters for the PF jet analyzer
process.pfJetStudies = process.jetStudies.clone( useCalo = cms.bool(False) )


process.load('PhysicsTools.SelectorUtils.pfJetIDSelector_cfi')
process.load('PhysicsTools.SelectorUtils.jetIDSelector_cfi')

process.plotParameters = cms.PSet (
    doTracks = cms.bool(False),
    useMC = cms.bool(False)
)


process.inputs = cms.PSet (
    fileNames = cms.vstring(
        'reco_7TeV_380_pat.root'
        ),
    lumisToProcess = cms.untracked.VLuminosityBlockRange( myList )

)

process.outputs = cms.PSet (
    outputName = cms.string('jetPlots.root')
)

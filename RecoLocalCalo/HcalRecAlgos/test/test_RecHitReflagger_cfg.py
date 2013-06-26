import FWCore.ParameterSet.Config as cms

maxevents=10
isMC=False
#isMC=True


process = cms.Process('TEST')

#process.load('JetMETAnalysis.PromptAnalysis.ntuple_cff')

process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration/StandardSequences/GeometryExtended_cff')
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.load("Configuration/StandardSequences/FrontierConditions_GlobalTag_cff")

# GLOBAL TAGS:  REPLACE WITH WHATEVER IS APPROPRIATE FOR YOUR WORK!

#MC (Summer09-V16D_900GeV-v1)
if (isMC):
    process.GlobalTag.globaltag ='START3X_V16D::All'

#DATA (Feb9ReReco)  
else:
    process.GlobalTag.globaltag ='GR09_R_34X_V5::All'

#process.TFileService = cms.Service("TFileService",
    #fileName = cms.string( THISROOTFILE ),
    #closeFileFast = cms.untracked.bool(True)
#)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(maxevents) )
process.source = cms.Source (
    "PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/data/BeamCommissioning09/ZeroBias/RECO/Feb9ReReco_v2/0027/F08E9178-7016-DF11-82B4-00163E0101D2.root',
    ),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    secondaryFileNames = cms.untracked.vstring()
)

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.MessageLogger.cerr.default.limit = 100

# summary
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.load("hcalrechitreflagger_cfi")
#process.hcalrecoReflagged = cms.EDProducer("RecHitReflagger")
#process.hcalrechitReflagger.debug=4
process.towerMakerPET = process.towerMaker.clone()
process.towerMakerPET.hfInput = cms.InputTag("hcalrechitReflagger")
process.metPET = process.met.clone()
process.metPET.src = cms.InputTag("towerMakerPET")
process.ak5CaloJetsPET = process.ak5CaloJets.clone()
process.ak5CaloJetsPET.src = cms.InputTag("towerMakerPET")

# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    #outputCommands = process.RECOSIMEventContent.outputCommands,
    fileName = cms.untracked.string('output_file.root')
)


#New SeverityLevelComputer that forces RecHits with UserDefinedBit0 set to be excluded from new rechit collection

#process.hcalRecAlgos.SeverityLevels.append(cms.PSet(Level = cms.int32(2),
#                                                    RecHitFlags = cms.vstring('HFPET','HFS9S1'),
#                                                    ChannelStatus = cms.vstring('')))

print "STARTING SL:"
for i in process.hcalRecAlgos.SeverityLevels:
        print i
        


severitylevels=[]  # Store all severity levels
AddedFlag=False
NewSevLevel=10
for i in range(len(process.hcalRecAlgos.SeverityLevels)):  # loop over each severity level
    severitylevels.append(process.hcalRecAlgos.SeverityLevels[i].Level.value())  # store severity value
    flagvec=process.hcalRecAlgos.SeverityLevels[i].RecHitFlags.value()  # Get vector of rechit flags for this severity level
    flaglevel=process.hcalRecAlgos.SeverityLevels[i].Level.value()
    if "UserDefinedBit0" in flagvec and flaglevel<>10:  # remove HFLongShort from its default position
        flagvec.remove("UserDefinedBit0")
        process.hcalRecAlgos.SeverityLevels[i].RecHitFlags=flagvec
        print "Removed 'UserDefinedBit0' from severity level %i"%(process.hcalRecAlgos.SeverityLevels[i].Level.value())
    if (flaglevel==NewSevLevel):  # Set UserDefinedBit0 severity to 10, which will exclude such rechits from CaloTower
        print "FOUND LEVEL %i!"%NewSevLevel
        if "UserDefinedBit0" not in flagvec:
            if (flagvec<>['']):
                flagvec.append("UserDefinedBit0")
            else:
                flagvec=["UserDefinedBit0"]
            process.hcalRecAlgos.SeverityLevels[i].RecHitFlags=flagvec
            AddedFlag=True
if (AddedFlag==False):
    print "Found no Severity Level = %i; Adding it now"%NewSevLevel
    process.hcalRecAlgos.SeverityLevels.append(cms.PSet(Level=cms.int32(NewSevLevel),
                                                        RecHitFlags=cms.vstring("UserDefinedBit0"),
                                                        ChannelStatus=cms.vstring("")))

print "New Severity Levels:"
for i in process.hcalRecAlgos.SeverityLevels:
    print i

#print process.hbhereco.firstSample, "  FIRST"

process.reflagging_step = cms.Path(process.hcalrechitReflagger)
process.reconstruction_step = cms.Path(process.towerMakerPET*(process.metPET+process.ak5CaloJetsPET))
process.out_step = cms.EndPath(process.output)

process.schedule = cms.Schedule(process.reflagging_step,process.reconstruction_step,process.out_step)


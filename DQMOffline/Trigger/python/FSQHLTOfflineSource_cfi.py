import FWCore.ParameterSet.Config as cms


def getPTAveVPSet():
    ret=cms.VPSet(
        cms.PSet(
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string("HLT_DiPFJetAve30_HFJEC_"),
            partialFilterName  = cms.string("ForHFJECBase"),
            dqmhistolabel  = cms.string("hltcalo"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            drawables =  cms.VPSet(
                cms.PSet (
                    name = cms.string("pt"),
                    expression = cms.string("at(0).pt"),
                    bins = cms.int32(100),
                    min = cms.double(0),
                    max = cms.double(100)
                )
            )
        ),




        cms.PSet(
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string("HLT_DiPFJetAve60_HFJEC_"),
            partialFilterName  = cms.string("ForHFJECBase"),
            dqmhistolabel  = cms.string("hltcalo"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7"),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            drawables =  cms.VPSet(
                cms.PSet (
                    name = cms.string("pt"),
                    expression = cms.string("at(0).pt"),
                    bins = cms.int32(100),
                    min = cms.double(0),
                    max = cms.double(100)
                )
            )
        ),
        # vector<reco::GenJet>                "ak5GenJets"                ""          "SIM"          recoGenJets_ak5GenJets__SIM
        cms.PSet(
            handlerType = cms.string("FromRecoCandidate"),
            inputCol = cms.InputTag("ak4GenJets"),
            partialPathName = cms.string("HLT_DiPFJetAve30_HFJEC_"),
            partialFilterName  = cms.string("dummy"),
            dqmhistolabel  = cms.string("ak4gen"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("abs(eta)< 5.2 && pt > 20"),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            drawables =  cms.VPSet(
                cms.PSet (
                    name = cms.string("pt"),
                    expression = cms.string("at(0).pt"),
                    bins = cms.int32(100),
                    min = cms.double(0),
                    max = cms.double(100)
                )
            )
        ),
    )
    return ret
fsqdirname = "HLT/FSQ/"
ttt = cms.EDAnalyzer("FSQDiJetAve",
    dirname = cms.untracked.string("HLT/FSQ/DiJETAve/"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","TTT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","TTT"),
    useGenWeight = cms.bool(False),
    #useGenWeight = cms.bool(True),
    todo = cms.VPSet(getPTAveVPSet())
)


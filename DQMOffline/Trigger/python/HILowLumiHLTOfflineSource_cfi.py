import FWCore.ParameterSet.Config as cms

def getHILowLumiTriggers():
    ret=cms.VPSet()
    partialPathName = "HLT_AK4CaloJet30_v"
    hltHICaloJet30 =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltSingleAK4CaloJet"),
        dqmhistolabel  = cms.string("hltHICaloJet30"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(58), min = cms.double(10), max = cms.double(300)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5), max = cms.double(5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltHICaloJet30)

    hltHICaloJet40 = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet40_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet40_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet40")
                                  )
    ret.append(hltHICaloJet40)

    hltHICaloJet50 = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet50_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet50_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet50")
    )
    ret.append(hltHICaloJet50)

    hltHICaloJet80 = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet80_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet80_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet80")
    )
    ret.append(hltHICaloJet80)

    hltHICaloJet100 = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet100_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet100_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet100")
    )
    ret.append(hltHICaloJet100)

    hltHICaloJet30ForEndOfFill = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet30ForEndOfFill_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet30ForEndOfFill_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet30ForEndOfFill")
    )
    ret.append(hltHICaloJet30ForEndOfFill)

    hltHICaloJet40ForEndOfFill = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet40ForEndOfFill_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet40ForEndOfFill_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet40ForEndOfFill")
    )
    ret.append(hltHICaloJet40ForEndOfFill)

    hltHICaloJet50ForEndOfFill = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4CaloJet50ForEndOfFill_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet50ForEndOfFill_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet50ForEndOfFill")
    )
    ret.append(hltHICaloJet50ForEndOfFill)


    hltHIPFJet30 = hltHICaloJet30.clone(partialPathName = cms.string("HLT_AK4PFJet30_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet30_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet30"),
                                  partialFilterName  = cms.string("hltSingleAK4PFJet")
                                  )
    ret.append(hltHIPFJet30)

    hltHIPFJet50 = hltHIPFJet30.clone(partialPathName = cms.string("HLT_AK4PFJet50_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet50_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet50")
    )
    ret.append(hltHIPFJet50)

    hltHIPFJet80 = hltHIPFJet30.clone(partialPathName = cms.string("HLT_AK4PFJet80_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet80_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet80")
    )
    ret.append(hltHIPFJet80)

    hltHIPFJet100 = hltHIPFJet30.clone(partialPathName = cms.string("HLT_AK4PFJet100_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet100_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet100")
    )
    ret.append(hltHIPFJet100)

    hltHIPFJet30ForEndOfFill = hltHIPFJet30.clone(partialPathName = cms.string("HLT_AK4PFJet30ForEndOfFill_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet30ForEndOfFill_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet30ForEndOfFill")
    )
    ret.append(hltHIPFJet30ForEndOfFill)

    hltHIPFJet50ForEndOfFill = hltHIPFJet30.clone(partialPathName = cms.string("HLT_AK4PFJet50ForEndOfFill_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet50ForEndOfFill_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet50ForEndOfFill")
    )
    ret.append(hltHIPFJet50ForEndOfFill)

    hltHISinglePhoton10 = hltHICaloJet30.clone(partialPathName = cms.string("HLT_HISinglePhoton10_v"),
                                               triggerSelection = cms.string("HLT_HISinglePhoton10_v*"),
                                               dqmhistolabel = cms.string("hltHISinglePhoton10"),
                                               partialFilterName  = cms.string("hltHIPhoton")
    )
    ret.append(hltHISinglePhoton10)

    hltHISinglePhoton15 = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton15_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton15_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton15")
    )
    ret.append(hltHISinglePhoton15)


    hltHISinglePhoton20 = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton20_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton20_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton20")
    )
    ret.append(hltHISinglePhoton20)

    hltHISinglePhoton40 = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton40_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton40_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton40")
    )
    ret.append(hltHISinglePhoton40)

    hltHISinglePhoton60 = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton60_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton60_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton60")
    )
    ret.append(hltHISinglePhoton60)

    hltHISinglePhoton10ForEndOfFill = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton10ForEndOfFill_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton10ForEndOfFill_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton10ForEndOfFill")
    )
    ret.append(hltHISinglePhoton10ForEndOfFill)

    hltHISinglePhoton15ForEndOfFill = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton15ForEndOfFill_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton15ForEndOfFill_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton15ForEndOfFill")
    )
    ret.append(hltHISinglePhoton15ForEndOfFill)

    hltHISinglePhoton20ForEndOfFill = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton20ForEndOfFill_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton20ForEndOfFill_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton20ForEndOfFill")
    )
    ret.append(hltHISinglePhoton20ForEndOfFill)

    return ret

def getFullTrackVPSet():
    ret=cms.VPSet()
    thresholds = [12, 20, 30, 50]
    for t in thresholds:
        partialPathName = "HLT_FullTrack"+str(t)+"_v"
        hltFullTrack =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHighPtFullTrack"),
            dqmhistolabel  = cms.string("hltHighPtFullTrack"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(0), max = cms.double(100)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltFullTrack)

    thresholds2 = [12]
    for t in thresholds2:
        partialPathName = "HLT_FullTrack"+str(t)+"ForEndOfFill_v"
        hltFullTrack =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHighPtFullTrack"),
            dqmhistolabel  = cms.string("hltHighPtFullTrack"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(0), max = cms.double(100)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltFullTrack)
    
    thresholds3 = [80,100,130,150]
    for t in thresholds3:
        partialPathName = "HLT_FullTracks_Multiplicity"+str(t)+"_v"
#        partialPathName = "HLT_FullTracks_Multiplicity80_v"
        hltFullTrackMult =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltPAFullTrackHighMult"),
            dqmhistolabel  = cms.string("hltPAFullTrackHighMult"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(0), max = cms.double(100)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
	ret.append(hltFullTrackMult)

    return ret

def getHILowLumi():
    ret = cms.VPSet()
    ret.extend(getHILowLumiTriggers())
    ret.extend(getFullTrackVPSet())
    return ret

dirname = "HLT/HI/"

processName = "HLT"

HILowLumiHLTOfflineSource = cms.EDAnalyzer("FSQDiJetAve",
    triggerConfiguration =  cms.PSet(
      hltResults = cms.InputTag('TriggerResults','',processName),
      l1tResults = cms.InputTag(''),
      #l1tResults = cms.InputTag('gtDigis'),
      daqPartitions = cms.uint32(1),
      l1tIgnoreMask = cms.bool( False ),
      l1techIgnorePrescales = cms.bool( False ),
      throw  = cms.bool( False )
    ),

#                                           hltProcessName = cms.string("HLT"),
    # HLT paths passing any one of these regular expressions will be included    

#    hltPathsToCheck = cms.vstring(
#      "HLT_HISinglePhoton10_v1",
#    ),

#    requiredTriggers = cms.untracked.vstring(
#      "HLT_HISinglePhoton10_v1",
#    ),

                                           
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","", processName),
    triggerResultsLabel = cms.InputTag("TriggerResults","", processName),
    useGenWeight = cms.bool(False),
    #useGenWeight = cms.bool(True),
    todo = cms.VPSet(getHILowLumi())
)

#from JetMETCorrections.Configuration.CorrectedJetProducers_cff import *
HILowLumiHLTOfflineSourceSequence = cms.Sequence(HILowLumiHLTOfflineSource)

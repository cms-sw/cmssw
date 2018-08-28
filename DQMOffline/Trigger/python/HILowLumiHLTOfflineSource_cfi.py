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

    thresholds3 = [85,105,135,155]
    for t in thresholds3:
        partialPathName = "HLT_FullTracks_Multiplicity"+str(t)+"_v"
        hltFullTrackMult =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltFullTrackHighMult"),
            dqmhistolabel  = cms.string("hltFullTrackHighMult"),
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

    thresholds4 = [20, 30, 40]
    for t in thresholds4:
        partialPathName = "HLT_PAFullTracks_HighPt"+str(t)+"_v"
        hltPAFullTrack =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltPAFullTrack"),
            dqmhistolabel  = cms.string("hltPAFullTrack"),
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
        ret.append(hltPAFullTrack)

    return ret

def getPAHighMultVPSet():
    ret=cms.VPSet()
    thresholds = [120, 150, 185, 220, 250, 280]
    for t in thresholds:
        partialPathName = "HLT_PAFullTracks_Multiplicity"+str(t)+"_v"

        hltPAFullTracks =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltPAFullTrackHighMult"),
            dqmhistolabel  = cms.string("hltPAFullTracks"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltPAFullTracks)

    return ret

def getPAHighMultHighPtVPSet():
    ret=cms.VPSet()
    thresholds = [8, 16]
    for t in thresholds:
        partialPathName = "HLT_PAFullTracks_Multiplicity110_HighPt"+str(t)+"_v"

        hltPAFullTracks =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltPAFullTrackHighPt"),
            dqmhistolabel  = cms.string("hltPAFullTracks"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltPAFullTracks)

    thresholds = [8, 16]
    for t in thresholds:
        partialPathName = "HLT_PAFullTracks_HFSumEt005_HighPt"+str(t)+"_v"

        hltPAFullTracks =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltPAFullTrackHighPt"),
            dqmhistolabel  = cms.string("hltPAFullTracks"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltPAFullTracks)

    return ret

def getPAHighPtVPSet():
    ret=cms.VPSet()
    jetTypes = ["Calo", "PF"]
    jetThresholds = [40, 60, 80, 100, 120]
    jetThresholdsFor1 = [40, 60]
    jetThresholdsFor2 = [40]
    jetThresholdsForMB = [40]
    bjetThresholds = [40, 60, 80]
    dijetAveThresholds = [40, 60, 80]
    gammaThresholds = [10, 15, 20, 30, 40]
    gammaMBThresholds = [15, 20]
    gammaThresholdsEGJet = [30, 40]
    isogammaThresholds = [20]
    eleThresholds = [20]

    jetThresholdsMu = [30, 40, 60]
    gammaThresholdsMu = [10, 15, 20]
    muThresholds = [3, 5]

    for jType in jetTypes:
        for t in jetThresholds:
            if jType == "Calo" and t == 120:
                continue
            partialPathName = "HLT_PAAK4" + jType + "Jet" + str(t) + "_Eta5p1_v"
            hltSingleJet =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltSinglePAAK4" + jType + "Jet"),
                dqmhistolabel  = cms.string("hltSingleAK4" + jType + "Jet" + str(t)),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                  singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltSingleJet)

        for t in jetThresholdsForMB:
            partialPathName = "HLT_PAAK4" + jType + "Jet" + str(t) + "_Eta5p1_SeededWithMB_v"
            hltSingleJet =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltSinglePAAK4" + jType + "Jet"),
                dqmhistolabel  = cms.string("hltSingleAK4" + jType + "Jet" + str(t)),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                  singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltSingleJet)

        for t in jetThresholdsFor1:
            partialPathName = "HLT_PAAK4" + jType + "Jet" + str(t) + "_Eta1p9toEta5p1_v"
            hltSingleJet =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltSinglePAAK4" + jType + "Jet" + str(t) + "MinEta1p9"),
                dqmhistolabel  = cms.string("hltSingleAK4" + jType + "Jet" + str(t) + "MinEta1p9"),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                  )
            ret.append(hltSingleJet)

        for t in jetThresholdsFor2:
            partialPathName = "HLT_PAAK4" + jType + "Jet" + str(t) + "_Eta2p9toEta5p1_v"
            hltSingleJet =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltSinglePAAK4" + jType + "Jet" + str(t) + "MinEta2p9"),
                dqmhistolabel  = cms.string("hltSingleAK4" + jType + "Jet" + str(t) + "MinEta2p9"),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltSingleJet)

        for t in dijetAveThresholds:
            partialPathName = "HLT_PADiAK4" + jType + "JetAve" + str(t) + "_Eta5p1_v"
            hltSingleJet =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiAk4" + jType + "JetAve"),
                dqmhistolabel  = cms.string("hltDiAk4" + jType + "JetAve" + str(t)),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltSingleJet)

        for t in bjetThresholds:
            partialPathName = "HLT_PAAK4" + jType + "BJetCSV" + str(t) + "_Eta2p1_v"
            hltSingleJet =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltSinglePAAK4" + jType + "Jet"),
                dqmhistolabel  = cms.string("hltSinglePAAK4" + jType + "BJetCSV" + str(t)),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltSingleJet)

    for t in bjetThresholds:
        partialPathName = "HLT_PAAK4" + "PF" + "BJetCSV" + str(t) + "_CommonTracking_Eta2p1_v"
        hltSingleJet =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltSinglePAAK4" + jType + "Jet"),
            dqmhistolabel  = cms.string("hltSinglePAAK4" + jType + "BJetCSV" + str(t) + "CommonTracking"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleJet)

    for t in gammaThresholds:
        partialPathName = "HLT_PASinglePhoton" + str(t) + "_Eta3p1_v"
        hltSingleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHIPhoton"),
            dqmhistolabel  = cms.string("hltHIPhoton" + str(t)),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleGamma)

    for t in gammaMBThresholds:
        partialPathName = "HLT_PASinglePhoton" + str(t) + "_Eta3p1_SeededWithMB_v"
        hltSingleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHIPhoton"),
            dqmhistolabel  = cms.string("hltHIPhoton" + str(t)),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleGamma)

    for t in gammaThresholdsEGJet:
        partialPathName = "HLT_PASinglePhoton" + str(t) + "_L1EGJet_Eta3p1_v"
        hltSingleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHIPhoton"),
            dqmhistolabel  = cms.string("hltHIPhoton" + str(t)),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleGamma)

    for t in gammaThresholds:
        partialPathName = "HLT_PAPhoton" + str(t) + "_Eta3p1_PPStyle_v"
        hltSingleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltEGBptxAND" + str(t) + "EtFilter"),
            dqmhistolabel  = cms.string("hltHIPhoton" + str(t) + "PPStyle"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleGamma)

    for t in isogammaThresholds:
        partialPathName = "HLT_PASingleIsoPhoton" + str(t) + "_Eta3p1_v"
        hltSingleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHIPhoton"),
            dqmhistolabel  = cms.string("hltHIIsoPhoton" + str(t)),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleGamma)

    for t in isogammaThresholds:
        partialPathName = "HLT_PAIsoPhoton" + str(t) + "_Eta3p1_PPStyle_v"
        hltSingleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltIsoEGBptxAND" + str(t) + "EtFilter"),
            dqmhistolabel  = cms.string("hltHIIsoPhoton" + str(t) + "PPStyle"),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleGamma)

    for t in eleThresholds:
        partialPathName = "HLT_PAEle" + str(t) + "_WPLoose_Gsf_v"
        hltSingleElectron =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltPAEle" + str(t) + "WPLooseGsfTrackIsoFilter"),
            dqmhistolabel  = cms.string("hltHIPAElectron" + str(t)),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )
        ret.append(hltSingleElectron)

    for t in jetThresholdsMu:
        for tMu in muThresholds:
            partialPathName = "HLT_PAAK4CaloJet" + str(t) + "_Eta5p1_PAL3Mu" + str(tMu) + "_v"
            hltJetMu =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltSinglePAAK4CaloJet"),
                dqmhistolabel  = cms.string("hltSingleAK4CaloJet" + str(t)),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(420)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5.0), max = cms.double(5.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltJetMu)

    for t in gammaThresholdsMu:
        for tMu in muThresholds:
            partialPathName = "HLT_PASinglePhoton" + str(t) + "_Eta3p1_PAL3Mu" + str(tMu) + "_v"
            hltGammaMu =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltHIPhoton"),
                dqmhistolabel  = cms.string("hltHIPhoton" + str(t)),
                mainDQMDirname = cms.untracked.string(dirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                    cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                    ),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet()
                )
            ret.append(hltGammaMu)

    partialPathName = "HLT_PADoublePhoton15_Eta3p1_Mass50_1000_v"
    hltDoubleGamma =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltHIDoublePhotonCut"),
        dqmhistolabel  = cms.string("hltHIDoublePhotonCut15"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
        )

    ret.append(hltDoubleGamma)

    return ret




def getPAMBVPSet():
    ret=cms.VPSet()
    partialPathName = "HLT_PAL1MinimumBiasHF_OR_SinglePixelTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hltPAMBPixelTracks"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PAL1MinimumBiasHF_AND_SinglePixelTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hltPAMBPixelTracks"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PAZeroBias_SinglePixelTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hltPAMBPixelTracks"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PAZeroBias_DoublePixelTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hltPAMBPixelTracks"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PASingleEG5_HFTwoTowerVeto_SingleTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hlt_PASingleEG5_HFTTV_ST"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PASingleEG5_HFOneTowerVeto_SingleTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hlt_PASingleEG5_HFOTV_ST"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PADoubleEG2_HFTwoTowerVeto_SingleTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hlt_PADoubleEG2_HFTTV_ST"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    partialPathName = "HLT_PADoubleEG2_HFOneTowerVeto_SingleTrack_v"
    hltPAMBPixelTracks =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltPAPixelFilter"),
        dqmhistolabel  = cms.string("hlt_PADoubleEG2_HFOTV_ST"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(100)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPAMBPixelTracks)

    return ret

#Note that jet triggers up to Jet100 are pre-covered; need to handle Jets120 and photon
def getHILowPU2017Triggers():
    ret=cms.VPSet()
    partialPathName = "HLT_AK4CaloJet120_v"
    hltHICaloJet120 =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltSingleAK4CaloJet"),
        dqmhistolabel  = cms.string("hltHICaloJet120"),
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
    ret.append(hltHICaloJet120)

    hltHIPFJet120 = hltHICaloJet120.clone(partialPathName = cms.string("HLT_AK4PFJet120_v"),
                                          triggerSelection = cms.string("HLT_AK4PFJet120_v*"),
                                          dqmhistolabel = cms.string("hltHIPFJet120"),
                                          partialFilterName = cms.string("hltSingleAK4PFJet"),
                                          )
    ret.append(hltHIPFJet120)

    photonThresh = ['10', '15', '20', '30', '40', '50', '60']
    photonThresh2 = ['20', '30', '40', '50', '60']

    for thresh in photonThresh:
        inString = 'HLT_HISinglePhoton' + thresh + '_Eta3p1ForPPRef_v'
        inStringAsterisk = 'HLT_HISinglePhoton' + thresh + '_Eta3p1ForPPRef_v*'
        inStringHistoLabel = 'hltHIPhoton' + thresh

        temp = hltHICaloJet120.clone(partialPathName = cms.string(inString),
                                     triggerSelection = cms.string(inStringAsterisk),
                                     dqmhistolabel = cms.string(inStringHistoLabel),
                                     partialFilterName  = cms.string("hltHIPhoton"),
                                     singleObjectDrawables =  cms.VPSet(cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(58), min = cms.double(10), max = cms.double(300)),
                                                                        cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3), max = cms.double(3)),
                                                                        cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                                                                        )
                                     )
        ret.append(temp)

    for thresh in photonThresh2:
        inString = 'HLT_Photon' + thresh + '_HoverELoose_v'
        inStringAsterisk = 'HLT_Photon' + thresh + '_HoverELoose_v*'
        inStringHistoLabel = 'hltPhoton' + thresh
        filterName = "hltEG" + thresh + "EtFilterLoose"

        temp = hltHICaloJet120.clone(partialPathName = cms.string(inString),
                                     triggerSelection = cms.string(inStringAsterisk),
                                     dqmhistolabel = cms.string(inStringHistoLabel),
                                     partialFilterName  = cms.string(filterName),
                                     singleObjectDrawables =  cms.VPSet(cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(58), min = cms.double(10), max = cms.double(300)),
                                                                        cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3), max = cms.double(3)),
                                                                        cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                                                                        )
                                     )
        ret.append(temp)

    return ret

def getPPRefHighPtVPSet():
    ret=cms.VPSet()

    # Calo Jets: 60
    partialPathName = "HLT_AK4CaloJet60_v"
    hltHICaloJet60 =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltSingleAK4CaloJet"),
        dqmhistolabel  = cms.string("hltHICaloJet60"),
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
    ret.append(hltHICaloJet60)

    # PF Jets: 40, 60
    hltHIPFJet40 = hltHICaloJet60.clone(partialPathName = cms.string("HLT_AK4PFJet40_v"),
        triggerSelection = cms.string("HLT_AK4PFJet40_v*"),
        dqmhistolabel = cms.string("hltHIPFJet40"),
        partialFilterName  = cms.string("hltSingleAK4PFJet")
    )
    ret.append(hltHIPFJet40)

    hltHIPFJet60 = hltHICaloJet60.clone(partialPathName = cms.string("HLT_AK4PFJet60_v"),
        triggerSelection = cms.string("HLT_AK4PFJet60_v*"),
        dqmhistolabel = cms.string("hltHIPFJet60"),
        partialFilterName  = cms.string("hltSingleAK4PFJet")
    )
    ret.append(hltHIPFJet60)

    # BJets: 30, 40, 60, 80
    BJetThresholds = ['30', '40', '60', '80']
    for thresh in BJetThresholds:
        hltHIBJet = hltHICaloJet60.clone(partialPathName = cms.string("HLT_AK4PFJet" + thresh + "_bTag_v"),
            triggerSelection = cms.string("HLT_AK4PFJet" + thresh + "_bTag_v*"),
            dqmhistolabel = cms.string("hltHIPFBJet" + thresh + ""),
            partialFilterName  = cms.string("hltSingleAK4PFJet" + thresh)
            )
        ret.append(hltHIBJet)

    # Calo FWD: 30, 40, 60, 80
    CaloFWDThresholds = ['30', '40', '60', '80']
    for thresh in CaloFWDThresholds:
        hltHICaloJetFWD = hltHICaloJet60.clone(partialPathName = cms.string("HLT_AK4CaloJet" + thresh + "FWD_v"),
            triggerSelection = cms.string("HLT_AK4CaloJet" + thresh + "FWD_v*"),
            dqmhistolabel = cms.string("hltHICaloJet" + thresh + "FWD"),
            partialFilterName  = cms.string("hltSingleAK4CaloJet" + thresh + "FWD")
            )
        ret.append(hltHICaloJetFWD)

    # PF Jet FWD: 30, 40, 60, 80
    PFFWDThresholds = ['30', '40', '60', '80']
    for thresh in PFFWDThresholds:
        hltHIPFJetFWD = hltHIPFJet60.clone(partialPathName = cms.string("HLT_AK4PFJet" + thresh + "FWD_v"),
            triggerSelection = cms.string("HLT_AK4PFJet" + thresh + "FWD_v*"),
            dqmhistolabel = cms.string("hltHIPFJet" + thresh + "FWD"),
            partialFilterName  = cms.string("hltSingleAK4PFJet" + thresh + "FWD")
            )
        ret.append(hltHIPFJetFWD)

    # Single Photon Eta 1p5
    photonThresh = ['10', '15', '20', '30', '40', '50', '60']
    for thresh in photonThresh:
        inString = 'HLT_HISinglePhoton' + thresh + '_Eta1p5ForPPRef_v'
        inStringAsterisk = inString + '*'
        inStringHistoLabel = 'hltHIPhoton' + thresh + '1p5'

        temp = hltHICaloJet60.clone(partialPathName = cms.string(inString),
            triggerSelection = cms.string(inStringAsterisk),
            dqmhistolabel = cms.string(inStringHistoLabel),
            partialFilterName  = cms.string("hltHIPhoton" + thresh + "Eta1p5"),
            singleObjectDrawables =  cms.VPSet(cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(58), min = cms.double(10), max = cms.double(300)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3), max = cms.double(3)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                )
            )
        ret.append(temp)

    # Electrons: 10, 15, 20, 30, 40
    ElectronThreshold = ['10', '15', '20', '30', '40']

    for thresh in ElectronThreshold:
        inString = 'HLT_Ele' + thresh + '_WPLoose_Gsf_v'
        inStringAsterisk = inString + '*'
        inStringHistoLabel = 'hltHIEle' + thresh

        temp = hltHICaloJet60.clone(partialPathName = cms.string(inString),
            triggerSelection = cms.string(inStringAsterisk),
            dqmhistolabel = cms.string(inStringHistoLabel),
            partialFilterName  = cms.string("hltEle" + thresh + 'WPLoose1GsfTrackIsoFilter'),
            singleObjectDrawables =  cms.VPSet(cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(58), min = cms.double(10), max = cms.double(300)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3), max = cms.double(3)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                )
            )
        ret.append(temp)

    # EG+Jet
    hltHIEGJet = hltHIPFJet60.clone(partialPathName = cms.string("HLT_Ele20_eta2p1_WPTight_Gsf_CentralPFJet15_EleCleaned_v"),
        triggerSelection = cms.string("HLT_Ele20_eta2p1_WPTight_Gsf_CentralPFJet15_EleCleaned_v*"),
        dqmhistolabel = cms.string("hltHIEle20WPLooseAK4PFJet15"),
        partialFilterName  = cms.string("hltEle20PFJet15EleCleaned")
        )
    ret.append(hltHIEGJet)

    # Ele+Ele
    hltHIEGEG = hltHIPFJet60.clone(partialPathName = cms.string("HLT_Ele20_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v"),
        triggerSelection = cms.string("HLT_Ele20_Ele12_CaloIdL_TrackIdL_IsoVL_DZ_v*"),
        dqmhistolabel = cms.string("hltHIEle20Ele12CaloIdLTrackIdLIsoVLDZ"),
        partialFilterName  = cms.string("hltEle20Ele12CaloIdLTrackIdLIsoVLDZFilter")
        )
    ret.append(hltHIEGEG)

    # Double photon with mass
    DoublePhotonThreshold = ['15', '20And15', '20']
    for threshold in DoublePhotonThreshold:
        partialPathName = "HLT_HIDoublePhoton" + threshold + "_Eta3p1ForPPRef_Mass50to1000_v"
        hltDoubleGamma =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltHIDoublePhoton" + threshold[-2:] + "Eta3p1"),
            dqmhistolabel  = cms.string("hltHIDoublePhoton" + threshold + '_Eta3p1ForPPRef'),
            mainDQMDirname = cms.untracked.string(dirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(100), min = cms.double(20), max = cms.double(220)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-3.0), max = cms.double(3.0)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
                ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
            )

        ret.append(hltDoubleGamma)

    return ret

def getHILowLumi():
    ret = cms.VPSet()
    ret.extend(getHILowLumiTriggers())
    ret.extend(getFullTrackVPSet())
    ret.extend(getPAHighMultVPSet())
    ret.extend(getPAHighMultHighPtVPSet())
    ret.extend(getPAMBVPSet())
    ret.extend(getPAHighPtVPSet())
    ret.extend(getHILowPU2017Triggers())
    ret.extend(getPPRefHighPtVPSet())
    return ret

dirname = "HLT/HI/"

processName = "HLT"

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
HILowLumiHLTOfflineSource = DQMEDAnalyzer('FSQDiJetAve',
    triggerConfiguration =  cms.PSet(
      hltResults = cms.InputTag('TriggerResults','',processName),
      l1tResults = cms.InputTag(''),
      l1tIgnoreMaskAndPrescale = cms.bool( False ),
      throw = cms.bool( False )
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

import FWCore.ParameterSet.Config as cms
import math

def getHILowLumiTriggers():
    ret=cms.VPSet()
    partialPathName = "HLT_AK4CaloJet15_v"
    hltHICaloJet15 =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltSingleAK4CaloJet"),
        dqmhistolabel  = cms.string("hltHICaloJet15"),
        mainDQMDirname = cms.untracked.string(dirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(50), min = cms.double(10), max = cms.double(300)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-5), max = cms.double(5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltHICaloJet15)

    hltHICaloJet20 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4CaloJet20_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet20_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet20")
                                  )
    ret.append(hltHICaloJet20)

    hltHICaloJet30 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4CaloJet30_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet30_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet30")
                                  )
    ret.append(hltHICaloJet30)

    hltHICaloJet40 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4CaloJet40_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet40_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet40")
                                  )
    ret.append(hltHICaloJet40)

    hltHICaloJet50 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4CaloJet50_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet50_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet50")
    )
    ret.append(hltHICaloJet50)

    hltHICaloJet80 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4CaloJet80_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet80_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet80")
    )
    ret.append(hltHICaloJet80)

    hltHICaloJet100 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4CaloJet100_v"),
                                  triggerSelection = cms.string("HLT_AK4CaloJet100_v*"),
                                  dqmhistolabel = cms.string("hltHICaloJet100")
    )
    ret.append(hltHICaloJet100)

    hltHIPFJet15 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_AK4PFJet15_v"),
                                        triggerSelection = cms.string("HLT_AK4PFJet15_v*"),
                                        dqmhistolabel = cms.string("hltHIPFJet15"),
                                        partialFilterName = cms.string("hltSingleAK4PFJet")
    )
    ret.append(hltHIPFJet15)


    hltHIPFJet20 = hltHIPFJet15.clone(partialPathName = cms.string("HLT_AK4PFJet20_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet20_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet20")
                                  )
    ret.append(hltHIPFJet20)

    hltHIPFJet30 = hltHIPFJet15.clone(partialPathName = cms.string("HLT_AK4PFJet30_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet30_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet30")
                                  )
    ret.append(hltHIPFJet30)

    hltHIPFJet40 = hltHIPFJet15.clone(partialPathName = cms.string("HLT_AK4PFJet40_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet40_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet40")
                                  )
    ret.append(hltHIPFJet40)

    hltHIPFJet50 = hltHIPFJet15.clone(partialPathName = cms.string("HLT_AK4PFJet50_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet50_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet50")
    )
    ret.append(hltHIPFJet50)

    hltHIPFJet80 = hltHIPFJet15.clone(partialPathName = cms.string("HLT_AK4PFJet80_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet80_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet80")
    )
    ret.append(hltHIPFJet80)

    hltHIPFJet100 = hltHIPFJet15.clone(partialPathName = cms.string("HLT_AK4PFJet100_v"),
                                  triggerSelection = cms.string("HLT_AK4PFJet100_v*"),
                                  dqmhistolabel = cms.string("hltHIPFJet100")
    )
    ret.append(hltHIPFJet100)

    hltHISinglePhoton10 = hltHICaloJet15.clone(partialPathName = cms.string("HLT_HISinglePhoton10_v"),
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

    hltHISinglePhoton30 = hltHISinglePhoton10.clone(partialPathName = cms.string("HLT_HISinglePhoton30_v"),
                                                    triggerSelection = cms.string("HLT_HISinglePhoton30_v*"),
                                                    dqmhistolabel = cms.string("hltHISinglePhoton30")
    )
    ret.append(hltHISinglePhoton30)

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

    return ret

def getHILowLumi():
    ret = cms.VPSet()
    ret.extend(getHILowLumiTriggers())
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

    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","", processName),
    triggerResultsLabel = cms.InputTag("TriggerResults","", processName),
    useGenWeight = cms.bool(False),
    #useGenWeight = cms.bool(True),
    todo = cms.VPSet(getHILowLumi())
)

#from JetMETCorrections.Configuration.CorrectedJetProducers_cff import *
HILowLumiHLTOfflineSourceSequence = cms.Sequence(HILowLumiHLTOfflineSource)

import FWCore.ParameterSet.Config as cms
import math
#
# see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePhysicsCutParser
#
#   trigger condition - check if trigger matching this pattern string was fired
#                       empty string - dont check anything
#
#
#   handler of type "fromHLT" fetches products of filter with name matching 
#    the  partialFilterName string,  that was run inside path with name matching 
#    the  partialPathName 
#  
#   other handlers read data from collection pointed by inputCol parameter 
#       (partialFilterName, partialPathName params are ignored)
#
#   note: be extra carefull when using singleObject and combinedObject drawables in same
#          handler definition. Histo names may be the same, currently there is no protection against it
#
def getHighMultVPSet():
    ret=cms.VPSet()
    thresholds = [60, 85, 110, 135, 160]
    for t in thresholds:
        partialPathName = "HLT_PixelTracks_Multiplicity"+str(t)+"_v"

        tracksL = 0
        tracksH = 200
        tracksBins = (tracksH-tracksL)/5
        tracksCount  =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("RecoTrackCounter"),
                #inputCol = cms.InputTag("hltPixelTracksForHighMult"),
                inputCol = cms.InputTag("generalTracks"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1sETT"),
                #dqmhistolabel  = cms.string("hltPixelTracks"),
                dqmhistolabel  = cms.string("recoTracks"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("1==1"), # add reco::Tracks selection criteria
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string('size()'),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("count_nominator"), expression = cms.string('at(0)'), 
                             bins = cms.int32(tracksBins), min = cms.double(tracksL), max = cms.double(tracksH))
                )
        )
        ret.append(tracksCount)				

        tracksCountDenom = tracksCount.clone()
        alwaysTrue = partialPathName+"*" + " OR NOT " + partialPathName+"*"
        tracksCountDenom.triggerSelection = cms.string(alwaysTrue)
        tracksCountDenom.combinedObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("count_denominator"), expression = cms.string("at(0)"),
                             bins = cms.int32(tracksBins), min = cms.double(tracksL), max = cms.double(tracksH))
        )
        ret.append(tracksCountDenom)


        hltPixelTracks =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("FromHLT"),
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hlt1HighMult"),
            dqmhistolabel  = cms.string("hltPixelTracks"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("1==1"),
            singleObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(20), min = cms.double(0), max = cms.double(10)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltPixelTracks)

        # FIXME: what variables it makes sense to plot in case of ETT seeds?
        l1 =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1sETT"),
                dqmhistolabel  = cms.string("l1"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(256/4), min = cms.double(0), max = cms.double(256)),
                )
        )
        ret.append(l1) 



    return ret


def getPTAveVPSet():
    ret=cms.VPSet()
    # note: always give integer values (!)
    thresholds = [30, 60, 80, 100, 160, 220, 300]
    #thresholds = [30,]
    for t in thresholds:
            partialPathName = "HLT_DiPFJetAve"+ str(t) +"_HFJEC_"
            ptBinLow  = t/2
            ptBinHigh = max(100, t*2)
            ptBins = min(100, ptBinHigh-ptBinLow)

        
            hltCalo =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("ForHFJECBase"), # note: this matches to hltSingleCaloJetXXXForHFJECBase
                dqmhistolabel  = cms.string("hltCaloJets"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(hltCalo)

            l1 =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1"),
                dqmhistolabel  = cms.string("l1"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("1==1"),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(256/4), min = cms.double(0), max = cms.double(256)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104/4), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(l1)

            '''
            hltPFSingle  =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("hltpfsingle"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(hltPFSingle)
            '''


            hltPFtopology  =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("hltPFJetsTopology"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("abs(at(0).eta())< 1.4 && abs(at(1).eta()) > 2.7 && abs(deltaPhi(at(0).phi, at(1).phi)) > 2.5"),
                combinedObjectSortCriteria = cms.string("(at(0).pt+at(1).pt)/2"),
                combinedObjectDimension = cms.int32(2),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("deltaEta"), expression = cms.string("abs(at(0).eta-at(1).eta)"), 
                             bins = cms.int32(70), min = cms.double(0), max = cms.double(7)),
                    cms.PSet (name = cms.string("deltaPhi"), expression = cms.string("abs(deltaPhi(at(0).phi, at(1).phi))"), 
                             bins = cms.int32(100), min = cms.double(0), max = cms.double(3.2)),
                    cms.PSet (name = cms.string("ptAve"), expression = cms.string("(at(0).pt+at(1).pt)/2"), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("ptTag"), expression = cms.string("? abs(at(0).eta) < 2 ? at(0).pt : at(1).pt "), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                    cms.PSet (name = cms.string("ptProbe"), expression = cms.string("? abs(at(0).eta) > 2 ? at(0).pt : at(1).pt "), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  )

                )
            )
            ret.append(hltPFtopology)




            '''
            # FromJet
            recoThr = t
            recoPF  =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromRecoCandidate"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoJet"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > + "+str(recoThr) +" && (abs(eta)<1.3 || abs(eta) > 2.8) "),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(52), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(recoPF) 
            '''
            recoThr = t/2
            recoPFtopology  =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("FromRecoCandidate"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoPFJetsTopology"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > + "+str(recoThr) +" && abs(eta)<1.4 || abs(eta) > 2.7 "),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("abs(at(0).eta())< 1.3 && abs(at(1).eta()) > 2.8 && abs(deltaPhi(at(0).phi, at(1).phi)) > 2.5"),
                combinedObjectSortCriteria = cms.string("(at(0).pt+at(1).pt)/2"),
                combinedObjectDimension = cms.int32(2),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("deltaEta"), expression = cms.string("abs(at(0).eta-at(1).eta)"), 
                             bins = cms.int32(70), min = cms.double(0), max = cms.double(7)),
                    cms.PSet (name = cms.string("deltaPhi"), expression = cms.string("abs(deltaPhi(at(0).phi, at(1).phi))"), 
                             bins = cms.int32(100), min = cms.double(0), max = cms.double(3.2)),
                    cms.PSet (name = cms.string("ptAve"), expression = cms.string("(at(0).pt+at(1).pt)/2"), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("ptTag"), expression = cms.string("? abs(at(0).eta) < 2 ? at(0).pt : at(1).pt "), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                    cms.PSet (name = cms.string("ptProbe"), expression = cms.string("? abs(at(0).eta) > 2 ? at(0).pt : at(1).pt "), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                    cms.PSet (name = cms.string("ptAve_nominator"), expression = cms.string("(at(0).pt+at(1).pt)/2"),
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                )
            )
            ret.append(recoPFtopology)
            recoPFtopologyDenom = recoPFtopology.clone()
            #recoPFtopologyDenom.triggerSelection = cms.string("HLTriggerFirstPath*")
            alwaysTrue = partialPathName+"*" + " OR NOT " + partialPathName+"*"
            #recoPFtopologyDenom.triggerSelection = cms.string(partialPathName+"*")
            recoPFtopologyDenom.triggerSelection = cms.string(alwaysTrue)
            recoPFtopologyDenom.combinedObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("ptAve_denominator"), expression = cms.string("(at(0).pt+at(1).pt)/2"),
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  )
            )
            ret.append(recoPFtopologyDenom)

            # RecoCandidateCounter
            ''' example on how to count objects
            recoThr = t/2
            recoPFJetCnt  =  cms.PSet(
                triggerSelection = cms.string(partialPathName+"*"),
                handlerType = cms.string("RecoCandidateCounter"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoPFJetsCnt"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > + "+str(recoThr) +" && abs(eta)<1.4 || abs(eta) > 2.7 "),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string('size()'),
                combinedObjectDimension = cms.int32(1),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("count"), expression = cms.string('at(0)'), 
                             bins = cms.int32(30), min = cms.double(0), max = cms.double(30))
                )
            )
            ret.append(recoPFJetCnt)
            '''

    return ret


def getFSQAll():
    ret = cms.VPSet()
    ret.extend(getHighMultVPSet())
    ret.extend(getPTAveVPSet())
    return ret


fsqdirname = "HLT/FSQ/"

#processName = "TTT"
#processName = "HLT"
processName = "TEST"

fsqHLTOfflineSource = cms.EDAnalyzer("FSQDiJetAve",
    triggerConfiguration =  cms.PSet(
      hltResults = cms.InputTag('TriggerResults','',processName),
      l1tResults = cms.InputTag(''),
      #l1tResults = cms.InputTag('gtDigis'),
      daqPartitions = cms.uint32(1),
      l1tIgnoreMask = cms.bool( False ),
      l1techIgnorePrescales = cms.bool( False ),
      throw  = cms.bool( False )
    ),

    #dirname = cms.untracked.string("HLT/FSQ/DiJETAve/"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","", processName),
    triggerResultsLabel = cms.InputTag("TriggerResults","", processName),
    useGenWeight = cms.bool(False),
    #useGenWeight = cms.bool(True),
    todo = cms.VPSet(getFSQAll())
)


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
def getZeroBias_SinglePixelTrackVPSet():
    ret=cms.VPSet()
    partialPathName = "HLT_ZeroBias_SinglePixelTrack_v"
    hltPixelTracksZB =  cms.PSet(
        triggerSelection = cms.string(partialPathName+"*"),
        handlerType = cms.string("FromHLT"),
        partialPathName = cms.string(partialPathName),
        partialFilterName  = cms.string("hltMinBiasPixelFilt"),
        dqmhistolabel  = cms.string("hltPixelTracks"),
        mainDQMDirname = cms.untracked.string(fsqdirname),
        singleObjectsPreselection = cms.string("1==1"),
        singleObjectDrawables =  cms.VPSet(
            cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(50), min = cms.double(0.4), max = cms.double(10)),
            cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
            cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
        ),
        combinedObjectSelection =  cms.string("1==1"),
        combinedObjectSortCriteria = cms.string("at(0).pt"),
        combinedObjectDimension = cms.int32(1),
        combinedObjectDrawables =  cms.VPSet()
    )
    ret.append(hltPixelTracksZB)

    # note: for global efficiency (ie not efficiency as a funtion of something)
    # calculation we use RecoTrack handler in a bit twisted way.
    #     RecoTrack handler assumes, that the efficiency calculation is done
    #   only for events with at least one offline track (from generalTracks collection)
    #   passing the selection criteria from singleObjectsPreselection variable
    #     Such events are used to fill 1-bin-large-histogram with a range -0.5...0.5
    #     Note, that the histogram is always filled with the 0 value ("0*" part in 
    #   expression string). The "at(0).pt()" part is needed to make expression parses
    #   happy.
    tracksCountZB  =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string("RecoTrack"),
            inputCol = cms.InputTag("generalTracks"),
            # l parameters
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltL1sETT"),
            dqmhistolabel  = cms.string("zb"),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("pt > 0.4 && abs(eta) < 2.4"), 
            singleObjectDrawables =  cms.VPSet(),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string('at(0).pt()'), # doesnt matter
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("Eff_nominator"), expression = cms.string('0*at(0).pt()'), 
                         bins = cms.int32(1), min = cms.double(-0.5), max = cms.double(0.5))
            )
    )
    ret.append(tracksCountZB)
    tracksCountDenomZB = tracksCountZB.clone()
    tracksCountDenomZB.triggerSelection = cms.string("HLT_ZeroBias_v*")
    tracksCountDenomZB.combinedObjectDrawables =  cms.VPSet(
        cms.PSet (name = cms.string("Eff_denominator"), expression = cms.string("0*at(0).pt()"),
                         bins = cms.int32(1), min = cms.double(-0.5), max = cms.double(0.5))
    )
    ret.append(tracksCountDenomZB)

    return ret

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
                handlerType = cms.string("RecoTrackCounterWithVertexConstraint"),
                inputCol = cms.InputTag("generalTracks"),
                # l parameters
                vtxCollection = cms.InputTag("offlinePrimaryVertices"),
                minNDOF = cms.int32(7),
                maxZ = cms.double(15),
                maxDZ = cms.double(0.12),
                maxDZ2dzsigma = cms.double(3),
                maxDXY = cms.double(0.12),
                maxDXY2dxysigma = cms.double(3),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1sETT"),
                #dqmhistolabel  = cms.string("hltPixelTracks"),
                dqmhistolabel  = cms.string("recoTracks"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > 0.4 && abs(eta) < 2.4"), 
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
        tracksCountDenom.triggerSelection = cms.string("TRUE")
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
                cms.PSet (name = cms.string("pt"), expression = cms.string("pt"), bins = cms.int32(200), min = cms.double(0.0), max = cms.double(10)),
                cms.PSet (name = cms.string("eta"), expression = cms.string("eta"), bins = cms.int32(100), min = cms.double(-2.5), max = cms.double(2.5)),
                cms.PSet (name = cms.string("phi"), expression = cms.string("phi"), bins = cms.int32(100), min = cms.double(-3.15), max = cms.double(3.15))
            ),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet()
        )
        ret.append(hltPixelTracks)

        hltPixelTracksEta16to18 = hltPixelTracks.clone()
        hltPixelTracksEta16to18.singleObjectsPreselection='abs(eta) > 1.6 && abs(eta) < 1.8'
        hltPixelTracksEta16to18.dqmhistolabel  = cms.string("hltPixelTracksEta16to18")
        for i in hltPixelTracksEta16to18.singleObjectDrawables:
            if i.name == "eta":
                hltPixelTracksEta16to18.singleObjectDrawables.remove(i)

        ret.append(hltPixelTracksEta16to18)

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


# note: always give integer values (!)
def getPTAveVPSet(thresholds = [30, 60, 80, 100, 160, 220, 300], flavour="HFJEC", disableCalo = False):
    # HLT_DiPFJetAve35_HFJEC_v1
    # HLT_DiPFJetAve15_Central_v1
    if flavour == "HFJEC":
        probeEtaSelection = "abs(eta) > 2.7"
        probeEtaSelectionCombined = "abs(at(1).eta) > 2.7"
    elif flavour == "Central":
        probeEtaSelection = "abs(eta) < 2.7"
        probeEtaSelectionCombined = "abs(at(1).eta) < 2.7"
    else:
        raise Exception("Flavour not known "+ flavour)
    ret=cms.VPSet()
    for t in thresholds:
            #partialPathName = "HLT_DiPFJetAve"+ str(t) +"_HFJEC_"
            partialPathName = "HLT_DiPFJetAve"+ str(t)+"_" + flavour + "_v"

            ptBinLow  = t/2
            ptBinHigh = max(100, t*2)
            ptBins = min(100, ptBinHigh-ptBinLow)

        
            if not disableCalo:
                hltCalo =  cms.PSet(
                    triggerSelection = cms.string(partialPathName+"*"),
                    handlerType = cms.string("FromHLT"),
                    partialPathName = cms.string(partialPathName),
                    partialFilterName  = cms.string("ForHFJECBase"), # note: this matches to hltSingleCaloJetXXXForHFJECBase
                    dqmhistolabel  = cms.string("hltCaloJets"),
                    mainDQMDirname = cms.untracked.string(fsqdirname),
                    singleObjectsPreselection = cms.string("abs(eta)<1.4 || " + probeEtaSelection),
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
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || " + probeEtaSelection),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("abs(at(0).eta())< 1.4 && "+ probeEtaSelectionCombined +
                                                      " && abs(deltaPhi(at(0).phi, at(1).phi)) > 2.5"),
                combinedObjectSortCriteria = cms.string("(at(0).pt+at(1).pt)/2"),
                combinedObjectDimension = cms.int32(2),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("deltaEta"), expression = cms.string("abs(at(0).eta-at(1).eta)"), 
                             bins = cms.int32(70), min = cms.double(0), max = cms.double(7)),
                    cms.PSet (name = cms.string("deltaPhi"), expression = cms.string("abs(deltaPhi(at(0).phi, at(1).phi))"), 
                             bins = cms.int32(100), min = cms.double(0), max = cms.double(3.2)),
                    cms.PSet (name = cms.string("ptAve"), expression = cms.string("(at(0).pt+at(1).pt)/2"), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("ptTag"), expression = 
                             cms.string("? abs(at(0).eta) < abs(at(1).eta) ? at(0).pt : at(1).pt "), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                    cms.PSet (name = cms.string("ptProbe"), expression = 
                             cms.string("? abs(at(0).eta) < abs(at(1).eta) ? at(1).pt : at(0).pt "), 
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
                handlerType = cms.string("RecoPFJetWithJEC"),
                PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3Corrector"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoPFJetsTopology"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > "+str(recoThr) +" && (abs(eta)<1.4 ||"+probeEtaSelection + ")" ),
                singleObjectDrawables =  cms.VPSet(),
                combinedObjectSelection =  cms.string("abs(at(0).eta())< 1.3 && " + probeEtaSelectionCombined + 
                                                      " && abs(deltaPhi(at(0).phi, at(1).phi)) > 2.5"),
                combinedObjectSortCriteria = cms.string("(at(0).pt+at(1).pt)/2"),
                combinedObjectDimension = cms.int32(2),
                combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("deltaEta"), expression = cms.string("abs(at(0).eta-at(1).eta)"), 
                             bins = cms.int32(70), min = cms.double(0), max = cms.double(7)),
                    cms.PSet (name = cms.string("deltaPhi"), expression = cms.string("abs(deltaPhi(at(0).phi, at(1).phi))"), 
                             bins = cms.int32(100), min = cms.double(0), max = cms.double(3.2)),
                    cms.PSet (name = cms.string("ptAve"), expression = cms.string("(at(0).pt+at(1).pt)/2"), 
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("ptTag"), expression = 
                            cms.string("?  abs(at(0).eta) < abs(at(1).eta) ? at(0).pt : at(1).pt "), 
                            bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                    cms.PSet (name = cms.string("ptProbe"), expression = 
                            cms.string("?  abs(at(0).eta) < abs(at(1).eta) ? at(1).pt : at(0).pt "), 
                            bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                    cms.PSet (name = cms.string("ptAve_nominator"), expression = cms.string("(at(0).pt+at(1).pt)/2"),
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  ),
                )
            )
            ret.append(recoPFtopology)
            recoPFtopologyDenom = recoPFtopology.clone()
            #recoPFtopologyDenom.triggerSelection = cms.string("HLTriggerFirstPath*")
            #recoPFtopologyDenom.triggerSelection = cms.string(partialPathName+"*")
            recoPFtopologyDenom.triggerSelection = cms.string("TRUE")
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
                singleObjectsPreselection = cms.string("pt >  "+str(recoThr) +" && abs(eta)<1.4 || abs(eta) > 2.7 "),
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

# note: thresholds should be a list with integer only values
def getSinglePFJet(thresholds, flavour=None, etaMin=-1, srcType="genJets", partialPathName = "HLT_PFJet", disableEff = False):
    if srcType == "genJets":
        inputCol = cms.InputTag("ak4GenJets")
        handlerType = "FromRecoCandidate"
        label = srcType
    elif srcType == "ak4PFJetsCHS":
        inputCol = cms.InputTag("ak4PFJetsCHS")
        handlerType = "RecoPFJetWithJEC"
        label = srcType
    elif srcType == "hlt":
        inputCol = cms.InputTag("S")
        handlerType = "FromHLT"
        label = srcType
    else:
        raise Exception("Whooops!")

    if etaMin == None:
        etaMin = -1

    ret=cms.VPSet()
    for t in thresholds:
        partialPathNameLoc = partialPathName
        partialPathNameLoc += str(t)+"_"
        if flavour != None:
            partialPathNameLoc += flavour+"_"
        partialPathNameLoc += "v"

        marginLow = max(t-t/2, 15)
        ptBinLow  = max(t-marginLow,0)
        marginHigh =  min(max(t/2, 20), 50)
        ptBinHigh = t+marginHigh
        ptBins = min(100, ptBinHigh-ptBinLow)
        fromJets =  cms.PSet(
            triggerSelection = cms.string(partialPathNameLoc+"*"),
            handlerType = cms.string(handlerType),
            PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3Corrector"),
            inputCol = inputCol,
            #    inputCol = cms.InputTag("ak4PFJetsCHS"),
            partialPathName = cms.string(partialPathNameLoc),
            partialFilterName  = cms.string("hltSinglePFJet"),
            dqmhistolabel  = cms.string(label),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("abs(eta) < 5.5 && abs(eta) > " + str(etaMin) ),
            singleObjectDrawables =  cms.VPSet(),
            combinedObjectSelection =  cms.string("1==1"),
            combinedObjectSortCriteria = cms.string("at(0).pt"),
            combinedObjectDimension = cms.int32(1),
            combinedObjectDrawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), 
                              bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), 
                              bins = cms.int32(52), min = cms.double(-5.2), max = cms.double(5.2)),
                    cms.PSet (name = cms.string("pt_nominator"), expression = cms.string("at(0).pt"),
                             bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  )
            )
        )
        if disableEff:
            for p in fromJets.combinedObjectDrawables:
                if p.name == cms.string("pt_nominator"):
                    fromJets.combinedObjectDrawables.remove(p)
                    break
        else:
            fromJetsDenom  = fromJets.clone()
            fromJetsDenom.triggerSelection = cms.string("HLT_ZeroBias_v*")
            fromJetsDenom.singleObjectDrawables =  cms.VPSet()
            fromJetsDenom.combinedObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("pt_denominator"), expression = cms.string("at(0).pt"),
                          bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  )
            )
            ret.append(fromJetsDenom)
        ret.append(fromJets)
    return ret

# note: thresholds should be a list with integer only values
# fixme: most of the code repeated from single jet case above
def getDoublePFJet(thresholds, flavour=None, etaMin=-1, srcType="genJets" ):
    if srcType == "genJets":
        inputCol = cms.InputTag("ak4GenJets")
        handlerType = "FromRecoCandidate"
        label = srcType
    elif srcType == "ak4PFJetsCHS":
        inputCol = cms.InputTag("ak4PFJetsCHS")
        handlerType = "RecoPFJetWithJEC"
        label = srcType
    elif srcType == "hlt":
        inputCol = cms.InputTag("S")
        handlerType = "FromHLT"
        label = srcType
    else:
        raise Exception("Whooops!")

    combinedObjectSortCriteria = "at(0).pt + at(1).pt"
    combinedObjectSelection = "1 == 1"
    if flavour != None and "FB" in flavour :
        combinedObjectSortCriteria = "("+combinedObjectSortCriteria+")*(  ? at(0).eta*at(1).eta < 0 ? 1 : 0 )"
        combinedObjectSelection = "at(0).eta*at(1).eta < 0"
        
    if etaMin == None:
        etaMin = -1

    ret=cms.VPSet()
    for t in thresholds:
        partialPathName = "HLT_DiPFJet"+ str(t)+"_"
        if flavour != None:
            partialPathName += flavour+"_"
        partialPathName += "v"

        marginLow = max(t-t/2, 15)
        ptBinLow  = max(t-marginLow,0)
        marginHigh =  min(max(t/3, 15), 50)
        ptBinHigh = t+marginHigh
        ptBins = min(100, ptBinHigh-ptBinLow)
        fromJets =  cms.PSet(
            triggerSelection = cms.string(partialPathName+"*"),
            handlerType = cms.string(handlerType),
            PFJetCorLabel        = cms.InputTag("ak4PFL1FastL2L3Corrector"),
            inputCol = inputCol,
            partialPathName = cms.string(partialPathName),
            partialFilterName  = cms.string("hltDoublePFJet"),
            dqmhistolabel  = cms.string(label),
            mainDQMDirname = cms.untracked.string(fsqdirname),
            singleObjectsPreselection = cms.string("abs(eta) < 5.5 && abs(eta) > " + str(etaMin) ),
            #singleObjectsPreselection = cms.string("pt > 15 && abs(eta) < 5.5 && abs(eta) > " + str(etaMin) ),
            singleObjectDrawables =  cms.VPSet(),
            combinedObjectSelection =  cms.string(combinedObjectSelection),
            combinedObjectSortCriteria = cms.string(combinedObjectSortCriteria),
            combinedObjectDimension = cms.int32(2),
            combinedObjectDrawables =  cms.VPSet(
                    #cms.PSet (name = cms.string("ptLead"), expression = cms.string("max(at(0).pt, at(1).pt())"), 
                    #          bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    #cms.PSet (name = cms.string("ptSublead"), expression = cms.string("min(at(0).pt, at(1).pt())"), 
                    #          bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("ptMostFwd"), 
                              expression = cms.string("? at(0).eta > at(1).eta ? at(0).pt : at(1).pt"), 
                              bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("ptMostBkw"), 
                              expression = cms.string("? at(0).eta > at(1).eta ? at(1).pt : at(0).pt"), 
                              bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),

                    cms.PSet (name = cms.string("etaMostFwd"), 
                              expression = cms.string("? at(0).eta > at(1).eta ? at(0).eta : at(1).eta"), 
                              bins = cms.int32(52), min = cms.double(-5.2), max = cms.double(5.2)),
                    cms.PSet (name = cms.string("etaMostBkw"), 
                              expression = cms.string("? at(0).eta > at(1).eta ? at(1).eta : at(0).eta"), 
                              bins = cms.int32(52), min = cms.double(-5.2), max = cms.double(5.2)),
                    #cms.PSet (name = cms.string("pt_nominator"), expression = cms.string("min(at(0).pt, at(1).pt)"),
                    #         bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  )
            )
        )
        #fromJets.triggerSelection = cms.string("HLT_ZeroBias_v*")
        ret.append(fromJets)
        if srcType != "hlt":
            fromJets.combinedObjectDrawables.append( cms.PSet(name = cms.string("ptm_nominator"), 
                                                              expression = cms.string("min(at(0).pt, at(1).pt)"),
                                                              bins = cms.int32(ptBins), 
                                                              min = cms.double(ptBinLow), 
                                                              max = cms.double(ptBinHigh)  ))

            fromJetsDenom  = fromJets.clone()
            fromJetsDenom.triggerSelection = cms.string("HLT_ZeroBias_v*")
            fromJetsDenom.singleObjectDrawables =  cms.VPSet()
            fromJetsDenom.combinedObjectDrawables =  cms.VPSet(
                cms.PSet (name = cms.string("ptm_denominator"), expression = cms.string("min(at(0).pt, at(1).pt)"),
                          bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)  )
            )
            ret.append(fromJetsDenom)
            

    return ret

def getFSQAll():
    ret = cms.VPSet()
    ret.extend(getHighMultVPSet())
    ret.extend(getZeroBias_SinglePixelTrackVPSet())
    ret.extend( getPTAveVPSet())

    ret.extend( getPTAveVPSet(thresholds = [15,25,35], disableCalo= True, flavour="HFJEC" ))
    ret.extend( getPTAveVPSet(thresholds = [15,25,35], disableCalo= True, flavour="Central" ))
    #todo = ["genJets", "ak4PFJetsCHS", "hlt"]
    todo = ["ak4PFJetsCHS", "hlt"]
    for t in todo:
        ret.extend(getSinglePFJet([20], flavour="NoCaloMatched", etaMin=None, srcType=t))
        ret.extend(getSinglePFJet([15, 25,40], flavour="NoCaloMatched", etaMin=None, srcType=t))
        ret.extend(getSinglePFJet([15, 25,40], flavour="FwdEta2_NoCaloMatched", etaMin=2, srcType=t))
        ret.extend(getSinglePFJet([15, 25,40], flavour="FwdEta3_NoCaloMatched", etaMin=3, srcType=t))

        ret.extend(getDoublePFJet([15], flavour="NoCaloMatched", etaMin=None, srcType=t))
        ret.extend(getDoublePFJet([15], flavour="FBEta2_NoCaloMatched", etaMin=2, srcType=t))
        ret.extend(getDoublePFJet([15], flavour="FBEta3_NoCaloMatched", etaMin=3, srcType=t))

        ret.extend(getSinglePFJet([15], partialPathName="HLT_L1Tech62_CASTORJet_SinglePFJet", srcType=t,  disableEff=True))


    return ret

def getFSQHI():
    ret = cms.VPSet()
    ret.extend(getZeroBias_SinglePixelTrackVPSet())
    #ret.extend(getHighMultVPSet())
    return ret

fsqdirname = "HLT/FSQ/"

processName = "HLT"
#processName = "TEST"

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

from JetMETCorrections.Configuration.CorrectedJetProducers_cff import *
fsqHLTOfflineSourceSequence = cms.Sequence(ak4PFL1FastL2L3CorrectorChain + fsqHLTOfflineSource)

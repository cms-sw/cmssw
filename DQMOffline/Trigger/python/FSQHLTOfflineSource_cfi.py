import FWCore.ParameterSet.Config as cms
import math
#
# see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuidePhysicsCutParser
#
def getHighMultVPSet():
    ret=cms.VPSet()
    thresholds = [60, 85, 110, 135, 160]
    for t in thresholds:
        partialPathName = "HLT_PixelTracks_Multiplicity"+str(t)+"_v"
        tracksCount  =  cms.PSet(
                handlerType = cms.string("RecoTrackCounter"),
                inputCol = cms.InputTag("hltPixelTracksForHighMult"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1sETT"),
                dqmhistolabel  = cms.string("hltPixelTracks"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("1==1"),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string('size()'),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("count"), expression = cms.string('at(0)'), 
                             bins = cms.int32(30), min = cms.double(t), max = cms.double(t+t/2))
                )
        )
        ret.append(tracksCount)				

        ptEtaHardest  =  cms.PSet(
                handlerType = cms.string("RecoTrack"),
                inputCol = cms.InputTag("hltPixelTracksForHighMult"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1sETT"),
                dqmhistolabel  = cms.string("hltPixelTracksPtEtaHardest"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string(""),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), 
                            bins = cms.int32(50), min = cms.double(0), max = cms.double(50)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), 
                            bins = cms.int32(50), min = cms.double(-2.5), max = cms.double(2.5))
                )
        )
        ret.append(ptEtaHardest) 

        # FIXME: what variables it makes sense to plot in case of ETT seeds?
        l1 =  cms.PSet(
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1sETT"),
                dqmhistolabel  = cms.string("l1"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("1==1"),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(256/4), min = cms.double(0), max = cms.double(256)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104/4), min = cms.double(-5.2), max = cms.double(5.2))
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
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("ForHFJECBase"), # note: this matches to hltSingleCaloJetXXXForHFJECBase
                dqmhistolabel  = cms.string("hltCaloJets"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(hltCalo)

            l1 =  cms.PSet(
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltL1"),
                dqmhistolabel  = cms.string("l1"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("1==1"),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(256/4), min = cms.double(0), max = cms.double(256)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104/4), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(l1)

            '''
            hltPFSingle  =  cms.PSet(
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("hltpfsingle"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(104), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(hltPFSingle)
            '''


            hltPFtopology  =  cms.PSet(
                handlerType = cms.string("FromHLT"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("hltPFJetsTopology"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("abs(eta)<1.4 || abs(eta) > 2.7 "),
                combinedObjectSelection =  cms.string("abs(at(0).eta())< 1.4 && abs(at(1).eta()) > 2.7 && abs(deltaPhi(at(0).phi, at(1).phi)) > 2.5"),
                combinedObjectSortCriteria = cms.string("(at(0).pt+at(1).pt)/2"),
                combinedObjectDimension = cms.int32(2),
                drawables =  cms.VPSet(
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
                handlerType = cms.string("FromRecoCandidate"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoJet"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > + "+str(recoThr) +" && (abs(eta)<1.3 || abs(eta) > 2.8) "),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string("at(0).pt"),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
                    cms.PSet (name = cms.string("pt"), expression = cms.string("at(0).pt"), bins = cms.int32(ptBins), min = cms.double(ptBinLow), max = cms.double(ptBinHigh)),
                    cms.PSet (name = cms.string("eta"), expression = cms.string("at(0).eta"), bins = cms.int32(52), min = cms.double(-5.2), max = cms.double(5.2))
                )
            )
            ret.append(recoPF) 
            '''
            recoThr = t/2
            recoPFtopology  =  cms.PSet(
                handlerType = cms.string("FromRecoCandidate"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoPFJetsTopology"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > + "+str(recoThr) +" && abs(eta)<1.4 || abs(eta) > 2.7 "),
                combinedObjectSelection =  cms.string("abs(at(0).eta())< 1.3 && abs(at(1).eta()) > 2.8 && abs(deltaPhi(at(0).phi, at(1).phi)) > 2.5"),
                combinedObjectSortCriteria = cms.string("(at(0).pt+at(1).pt)/2"),
                combinedObjectDimension = cms.int32(2),
                drawables =  cms.VPSet(
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
            ret.append(recoPFtopology)
            # RecoCandidateCounter
            ''' example on how to count objects
            recoThr = t/2
            recoPFJetCnt  =  cms.PSet(
                handlerType = cms.string("RecoCandidateCounter"),
                inputCol = cms.InputTag("ak4PFJetsCHS"),
                partialPathName = cms.string(partialPathName),
                partialFilterName  = cms.string("hltDiPFJetAve"),
                dqmhistolabel  = cms.string("recoPFJetsCnt"),
                mainDQMDirname = cms.untracked.string(fsqdirname),
                singleObjectsPreselection = cms.string("pt > + "+str(recoThr) +" && abs(eta)<1.4 || abs(eta) > 2.7 "),
                combinedObjectSelection =  cms.string("1==1"),
                combinedObjectSortCriteria = cms.string('size()'),
                combinedObjectDimension = cms.int32(1),
                drawables =  cms.VPSet(
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
processName = "HLT"
#processName = "TEST"
fsqHLTOfflineSource = cms.EDAnalyzer("FSQDiJetAve",
    #dirname = cms.untracked.string("HLT/FSQ/DiJETAve/"),
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","", processName),
    triggerResultsLabel = cms.InputTag("TriggerResults","", processName),
    useGenWeight = cms.bool(False),
    #useGenWeight = cms.bool(True),
    todo = cms.VPSet(getFSQAll())
)


import FWCore.ParameterSet.Config as cms

HLTPath = "HLT_Ele*"
HLTProcessName = "HLT"

### cut on electron tag
ELECTRON_ET_CUT_MIN_TIGHT = 20.0
ELECTRON_ET_CUT_MIN_LOOSE = 10.0

MASS_CUT_MIN = 40.



##    _____ _           _                     ___    _ 
##   | ____| | ___  ___| |_ _ __ ___  _ __   |_ _|__| |
##   |  _| | |/ _ \/ __| __| '__/ _ \| '_ \   | |/ _` |
##   | |___| |  __/ (__| |_| | | (_) | | | |  | | (_| |
##   |_____|_|\___|\___|\__|_|  \___/|_| |_| |___\__,_|
##   
# Electron ID  ######
from DPGAnalysis.Skims.WElectronSkim_cff import *

PassingVeryLooseId = goodElectrons.clone(
    cut = cms.string(
        goodElectrons.cut.value() +
        #    " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=1 && !(-0.02<convDist<0.02 && -0.02<convDcot<0.02))" #wrt std WP90 allowing 1 numberOfMissingExpectedHits
            " && (gsfTrack.trackerExpectedHitsInner.numberOfHits<=1 )" #wrt std WP90 allowing 1 numberOfMissingExpectedHits 
            " && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN_LOOSE) + ")"
            " && ((isEB"
            " && ( dr03TkSumPt/p4.Pt <0.2 && dr03EcalRecHitSumEt/p4.Pt < 0.3 && dr03HcalTowerSumEt/p4.Pt  < 0.3 )"
            " && (sigmaIetaIeta<0.012)"
            " && ( -0.8<deltaPhiSuperClusterTrackAtVtx<0.8 )"
            " && ( -0.01<deltaEtaSuperClusterTrackAtVtx<0.01 )"
            " && (hadronicOverEm<0.15)"
            ")"
            " || (isEE"
            " && ( dr03TkSumPt/p4.Pt <0.2 && dr03EcalRecHitSumEt/p4.Pt < 0.3 && dr03HcalTowerSumEt/p4.Pt  < 0.3 )"
            " && (sigmaIetaIeta<0.033)"
            " && ( -0.7<deltaPhiSuperClusterTrackAtVtx<0.7 )" 
            " && ( -0.01<deltaEtaSuperClusterTrackAtVtx<0.01 )"
            " && (hadronicOverEm<0.15) "
            "))"
        )
    )

PassingTightId = PassingVeryLooseId.clone(
    cut = cms.string(
        PassingVeryLooseId.cut.value() +
        " && (ecalEnergy*sin(superClusterPosition.theta)>" + str(ELECTRON_ET_CUT_MIN_TIGHT) + ")"
        )
    )

Zele_sequence = cms.Sequence(
    PassingVeryLooseId
    *PassingTightId 
    )
                         


##    _____ ___   ____    ____       _          
##   |_   _( _ ) |  _ \  |  _ \ __ _(_)_ __ ___ 
##     | | / _ \/\ |_) | | |_) / _` | | '__/ __|
##     | || (_>  <  __/  |  __/ (_| | | |  \__ \
##     |_| \___/\/_|     |_|   \__,_|_|_|  |___/
##                                              
##   
import copy
from HLTrigger.HLTfilters.hltHighLevel_cfi import *
ZEEHltFilter = copy.deepcopy(hltHighLevel)
ZEEHltFilter.throw = cms.bool(False)
ZEEHltFilter.HLTPaths = [HLTPath]

tagGsf = cms.EDProducer("CandViewShallowCloneCombiner",
    #    decay = cms.string("PassingWP90 goodElectrons"),
    # decay = cms.string("PassingVeryLooseId PassingVeryLooseId"),
    decay = cms.string("PassingTightId PassingVeryLooseId"),
    checkCharge = cms.bool(False), 
    cut   = cms.string("mass > " + str(MASS_CUT_MIN))
    )
tagGsfCounter = cms.EDFilter("CandViewCountFilter",
    src = cms.InputTag("tagGsf"),
    minNumber = cms.uint32(1)
    )
tagGsfFilter = cms.Sequence(tagGsf * tagGsfCounter)
tagGsfSeq = cms.Sequence( ZEEHltFilter * Zele_sequence * tagGsfFilter )  
#tagGsfSeq = cms.Sequence( ZEEHltFilter * Zele_sequence )  
#tagGsfSeq = cms.Sequence( ZEEHltFilter )  

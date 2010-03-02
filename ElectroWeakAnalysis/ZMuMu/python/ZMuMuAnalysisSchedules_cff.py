import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
from ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff import *


### controll by hand that all the path are in ... :-(

dimuonsSeq = cms.Sequence(
    dimuonsHLTFilter *
    goodMuonRecoForDimuon *
    dimuons *
    dimuonsGlobal *
    dimuonsOneStandAloneMuon *
    dimuonsFilter    
)

dimuonsOneTrackSeq= cms.Sequence(dimuonsHLTFilter+
                               goodMuonRecoForDimuon+
                               dimuonsOneTrack+
                               dimuonsOneTrackFilter
)






goodZToMuMuPathLoose.replace(goodZToMuMuLoose, dimuonsSeq *  goodZToMuMuLoose)

goodZToMuMu2HLTPathLoose.replace(goodZToMuMuLoose, dimuonsSeq *  goodZToMuMuLoose)

goodZToMuMu1HLTPathLoose.replace(goodZToMuMuLoose, dimuonsSeq *  goodZToMuMuLoose)

goodZToMuMuSameChargePathLoose.replace(dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)


notGoodZToMuMuSeq = cms.Sequence(
    dimuonsSeq *
    ~goodZToMuMu *
    zToMuMuOneStandAloneMuonLoose
    )

notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

goodZToMuMuOneStandAloneMuonPathLoose.remove(goodZToMuMu)

goodZToMuMuOneStandAloneMuonPathLoose.replace(zToMuMuOneStandAloneMuonLoose, notGoodZToMuMuSeq)

notGoodZToMuMuSeq = cms.Sequence(
    dimuonsSeq +
    dimuonsOneTrackSeq+
    ~goodZToMuMu +
    ~zToMuMuOneStandAloneMuon +
    zToMuGlobalMuOneTrack 
    )



notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

goodZToMuMuOneTrackPathLoose.remove( goodZToMuMu)
goodZToMuMuOneTrackPathLoose.remove(zToMuMuOneStandAloneMuon )
    
goodZToMuMuOneTrackPathLoose.replace(zToMuGlobalMuOneTrack, notGoodZToMuMuSeq *  zToMuGlobalMuOneTrack )

goodZToMuMuOneTrackPathLoose.remove(dimuonsFilter)


initialGoodZToMuMuPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)

goodZToMuMuPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)


goodZToMuMu2HLTPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)

goodZToMuMu1HLTPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)

goodZToMuMuSameChargePath.replace( dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)

goodZToMuMuSameCharge2HLTPath.replace( dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)

goodZToMuMuSameCharge1HLTPath.replace( dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)

nonIsolatedZToMuMuPath.replace(nonIsolatedZToMuMu, dimuonsSeq * nonIsolatedZToMuMu)


oneNonIsolatedZToMuMuPath.replace(nonIsolatedZToMuMu, dimuonsSeq * nonIsolatedZToMuMu)

twoNonIsolatedZToMuMuPath.replace(nonIsolatedZToMuMu, dimuonsSeq * nonIsolatedZToMuMu)

notGoodZToMuMuSeq = cms.Sequence(
    dimuonsSeq *
    ~goodZToMuMu *
    zToMuMuOneStandAloneMuon
    )

notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

goodZToMuMuOneStandAloneMuonPath.remove(goodZToMuMu)
goodZToMuMuOneStandAloneMuonPath.replace(zToMuMuOneStandAloneMuon, notGoodZToMuMuSeq)


notGoodZToMuMuSeq = cms.Sequence(
    dimuonsSeq +
    ~goodZToMuMu +
    dimuonsOneTrackSeq+
    ~zToMuMuOneStandAloneMuon +
    zToMuGlobalMuOneTrack 
    )



notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

goodZToMuMuOneTrackPath.remove( goodZToMuMu)
goodZToMuMuOneTrackPath.remove(zToMuMuOneStandAloneMuon )
    
goodZToMuMuOneTrackPath.replace(zToMuGlobalMuOneTrack, notGoodZToMuMuSeq *  zToMuGlobalMuOneTrack )
goodZToMuMuOneTrackPath.remove(dimuonsFilter)


## goodZToMuMuPathLoose = cms.Path(
    
##     goodZToMuMuLoose +
##     goodZToMuMuAtLeast1HLTLoose
##     )

## goodZToMuMu2HLTPathLoose = cms.Path(
##     goodZToMuMuLoose +
##     goodZToMuMu2HLTLoose
##     )

## goodZToMuMu1HLTPathLoose = cms.Path(
##     goodZToMuMuLoose +
##     goodZToMuMu1HLTLoose
##     )


## goodZToMuMuSameChargePathLoose = cms.Path(
##     dimuonsGlobalSameCharge+
##     goodZToMuMuSameChargeLoose +
##     goodZToMuMuSameChargeAtLeast1HLTLoose
##     )


## ## goodZToMuMuSameCharge2HLTPathLoose = cms.Path(
## ##     dimuonsGlobalSameCharge+
## ##     goodZToMuMuSameChargeLoose +
## ##     goodZToMuMuSameCharge2HLTLoose
## ##     )


## ## goodZToMuMuSameCharge1HLTPathLoose = cms.Path(
## ##     dimuonsGlobalSameCharge+
## ##     goodZToMuMuSameChargeLoose +
## ##     goodZToMuMuSameCharge1HLTLoose
## ##     )



## goodZToMuMuOneStandAloneMuonPathLoose = cms.Path(
## ### I should deby the tight zmumu, otherwise I cut to much.... 
##     ~goodZToMuMu + 
##     zToMuMuOneStandAloneMuonLoose + 
##     goodZToMuMuOneStandAloneMuonLoose +
##     goodZToMuMuOneStandAloneMuonFirstHLTLoose 
##     )


## goodZToMuMuOneTrackPathLoose=cms.Path(
##     ### I should deby the tight zmumu, otherwise I cut to much.... 
##     ~goodZToMuMu +
##     ~zToMuMuOneStandAloneMuon +
##     zToMuGlobalMuOneTrack +
##     zToMuMuOneTrackLoose +
##     goodZToMuMuOneTrackLoose +
##     goodZToMuMuOneTrackFirstHLTLoose 
##     )





## ### sequences and path for tight cuts...

## initialGoodZToMuMuPath = cms.Path( 
##     goodZToMuMu +
##     zmumuSaMassHistogram     
## )


## goodZToMuMuPath = cms.Path(
##     goodZToMuMu +
##     goodZToMuMuAtLeast1HLT
##     )



## goodZToMuMu2HLTPath = cms.Path(
##     goodZToMuMu +
##     goodZToMuMu2HLT
##     )


## goodZToMuMu1HLTPath = cms.Path(
##     goodZToMuMu +
##     goodZToMuMu1HLT
##     )



## goodZToMuMuSameChargePath = cms.Path(
##     dimuonsGlobalSameCharge+
##     goodZToMuMuSameCharge +
##     goodZToMuMuSameChargeAtLeast1HLT
##     )


## goodZToMuMuSameCharge2HLTPath = cms.Path(
##     dimuonsGlobalSameCharge+
##     goodZToMuMuSameCharge +
##     goodZToMuMuSameCharge2HLT
##     )



## goodZToMuMuSameCharge1HLTPath = cms.Path(
##     dimuonsGlobalSameCharge+
##     goodZToMuMuSameCharge +
##     goodZToMuMuSameCharge1HLT
##     )



## nonIsolatedZToMuMuPath = cms.Path (
##     nonIsolatedZToMuMu +
##     nonIsolatedZToMuMuAtLeast1HLT 
## )


## oneNonIsolatedZToMuMuPath  = cms.Path(
##     nonIsolatedZToMuMu  +
##     oneNonIsolatedZToMuMu  +
##     oneNonIsolatedZToMuMuAtLeast1HLT  
## )


## twoNonIsolatedZToMuMuPath  = cms.Path(
##     nonIsolatedZToMuMu  +
##     twoNonIsolatedZToMuMu  +
##     twoNonIsolatedZToMuMuAtLeast1HLT  
## )


## goodZToMuMuOneStandAloneMuonPath = cms.Path(
##     ~goodZToMuMu +
##     zToMuMuOneStandAloneMuon + 
##     goodZToMuMuOneStandAloneMuon +
##     goodZToMuMuOneStandAloneMuonFirstHLT 
##     )




## goodZToMuMuOneTrackPath=cms.Path(
##     ~goodZToMuMu +
##     ~zToMuMuOneStandAloneMuon +
##     zToMuGlobalMuOneTrack +
##     zToMuMuOneTrack +
##     goodZToMuMuOneTrack +
##     goodZToMuMuOneTrackFirstHLT 
##     )



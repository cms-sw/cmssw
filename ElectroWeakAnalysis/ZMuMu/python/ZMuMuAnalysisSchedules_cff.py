import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZMuMu.ZMuMuCategoriesSequences_cff import *
from ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff import *
#from ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff import *


### controll by hand that all the path are in ... :-( :-( :-(

## dimuonsSeq = cms.Sequence(
##     dimuonsHLTFilter *
##     goodMuonRecoForDimuon *
##     dimuons *
##     dimuonsGlobal *
##     dimuonsOneStandAloneMuon *
##     dimuonsFilter    
## )

## dimuonsOneTrackSeq= cms.Sequence(dimuonsHLTFilter+
##                                goodMuonRecoForDimuon+
##                                dimuonsOneTrack+
##                                dimuonsOneTrackFilter
## )






#goodZToMuMuPathLoose.replace(goodZToMuMuLoose, dimuonsSeq *  goodZToMuMuLoose)
goodZToMuMuPathLoose._seq = dimuonsPath._seq + goodZToMuMuPathLoose._seq

goodZToMuMu2HLTPathLoose._seq = dimuonsPath._seq + goodZToMuMu2HLTPathLoose._seq


#goodZToMuMu2HLTPathLoose.replace(goodZToMuMuLoose, dimuonsSeq *  goodZToMuMuLoose)

#goodZToMuMu1HLTPathLoose.replace(goodZToMuMuLoose, dimuonsSeq *  goodZToMuMuLoose)

goodZToMuMu1HLTPathLoose._seq= dimuonsPath._seq + goodZToMuMu1HLTPathLoose._seq 


goodZToMuMuAB1HLTPathLoose._seq = dimuonsPath._seq + goodZToMuMuAB1HLTPathLoose._seq

goodZToMuMuBB2HLTPathLoose._seq = dimuonsPath._seq + goodZToMuMuBB2HLTPathLoose._seq


#goodZToMuMuSameChargePathLoose.replace(dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)
goodZToMuMuSameChargePathLoose._seq = dimuonsPath._seq + goodZToMuMuSameChargePathLoose._seq


## notGoodZToMuMuSeq = cms.Sequence(
##     dimuonsSeq *
##     ~goodZToMuMu *
##     zToMuMuOneStandAloneMuonLoose
##     )

## notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

## goodZToMuMuOneStandAloneMuonPathLoose.remove(goodZToMuMu)

#goodZToMuMuOneStandAloneMuonPathLoose.replace(zToMuMuOneStandAloneMuonLoose, notGoodZToMuMuSeq)

goodZToMuMuOneStandAloneMuonPathLoose._seq = dimuonsPath._seq  + goodZToMuMuOneStandAloneMuonPathLoose._seq 

## notGoodZToMuMuSeq = cms.Sequence(
##     dimuonsSeq +
##     dimuonsOneTrackSeq+
##     ~goodZToMuMu +
##     ~zToMuMuOneStandAloneMuon +
##     zToMuGlobalMuOneTrack 
##     )

goodZToMuMuOneTrackerMuonPathLoose._seq = dimuonsPath._seq + goodZToMuMuOneTrackerMuonPathLoose._seq


## notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

## goodZToMuMuOneTrackPathLoose.remove( goodZToMuMu)
## goodZToMuMuOneTrackPathLoose.remove(zToMuMuOneStandAloneMuon )
    
#goodZToMuMuOneTrackPathLoose.replace(zToMuGlobalMuOneTrack, notGoodZToMuMuSeq *  zToMuGlobalMuOneTrack )

goodZToMuMuOneTrackPathLoose._seq = dimuonsPath._seq  + dimuonsOneTrackPath._seq + goodZToMuMuOneTrackPathLoose._seq 

goodZToMuMuOneTrackPathLoose.remove(dimuonsFilter)

#initialGoodZToMuMuPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)
initialGoodZToMuMuPath._seq = dimuonsPath._seq + initialGoodZToMuMuPath._seq


#goodZToMuMuPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)
goodZToMuMuPath._seq = dimuonsPath._seq + goodZToMuMuPath._seq


#goodZToMuMu2HLTPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)
goodZToMuMu1HLTPath._seq = dimuonsPath._seq + goodZToMuMu1HLTPath._seq

#goodZToMuMu1HLTPath.replace(goodZToMuMu, dimuonsSeq *  goodZToMuMu)
goodZToMuMu2HLTPath._seq = dimuonsPath._seq + goodZToMuMu2HLTPath._seq

goodZToMuMuAB1HLTPath._seq = dimuonsPath._seq + goodZToMuMuAB1HLTPath._seq

goodZToMuMuBB2HLTPath._seq = dimuonsPath._seq + goodZToMuMuBB2HLTPath._seq


#goodZToMuMuSameChargePath.replace( dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)
goodZToMuMuSameChargePath._seq = dimuonsPath._seq + goodZToMuMuSameChargePath._seq 

#goodZToMuMuSameCharge2HLTPath.replace( dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)

goodZToMuMuSameCharge2HLTPath._seq = dimuonsPath._seq + goodZToMuMuSameCharge2HLTPath._seq 

#goodZToMuMuSameCharge1HLTPath.replace( dimuonsGlobalSameCharge, dimuonsSeq * dimuonsGlobalSameCharge)
goodZToMuMuSameCharge1HLTPath._seq = dimuonsPath._seq + goodZToMuMuSameCharge1HLTPath._seq


#nonIsolatedZToMuMuPath.replace(nonIsolatedZToMuMu, dimuonsSeq * nonIsolatedZToMuMu)
nonIsolatedZToMuMuPath._seq = dimuonsPath._seq + nonIsolatedZToMuMuPath._seq 

#oneNonIsolatedZToMuMuPath.replace(nonIsolatedZToMuMu, dimuonsSeq * nonIsolatedZToMuMu)
oneNonIsolatedZToMuMuPath._seq = dimuonsPath._seq + oneNonIsolatedZToMuMuPath._seq 

#twoNonIsolatedZToMuMuPath.replace(nonIsolatedZToMuMu, dimuonsSeq * nonIsolatedZToMuMu)
twoNonIsolatedZToMuMuPath._seq = dimuonsPath._seq + twoNonIsolatedZToMuMuPath._seq 


## notGoodZToMuMuSeq = cms.Sequence(
##     dimuonsSeq *
##     ~goodZToMuMu *
##     zToMuMuOneStandAloneMuon
##     )

## notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

## goodZToMuMuOneStandAloneMuonPath.remove(goodZToMuMu)
## goodZToMuMuOneStandAloneMuonPath.replace(zToMuMuOneStandAloneMuon, notGoodZToMuMuSeq)
goodZToMuMuOneStandAloneMuonPath._seq = dimuonsPath._seq + goodZToMuMuOneStandAloneMuonPath._seq


## notGoodZToMuMuSeq = cms.Sequence(
##     dimuonsSeq +
##     ~goodZToMuMu +
##     dimuonsOneTrackSeq+
##     ~zToMuMuOneStandAloneMuon +
##     zToMuGlobalMuOneTrack 
##     )



## notGoodZToMuMuSeq.setLabel("notGoodZToMuMuSeq")

## goodZToMuMuOneTrackPath.remove( goodZToMuMu)
## goodZToMuMuOneTrackPath.remove(zToMuMuOneStandAloneMuon )
    
## goodZToMuMuOneTrackPath.replace(zToMuGlobalMuOneTrack, notGoodZToMuMuSeq *  zToMuGlobalMuOneTrack )


goodZToMuMuOneTrackerMuonPath._seq = dimuonsPath._seq + goodZToMuMuOneTrackerMuonPath._seq


goodZToMuMuOneTrackPath._seq = dimuonsPath._seq + dimuonsOneTrackPath._seq + goodZToMuMuOneTrackPath._seq
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



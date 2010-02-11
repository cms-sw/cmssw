import FWCore.ParameterSet.Config as cms

# reorganization of Z->mumu categories sequence, to run after the ZMuMu(Sub)Skim (i.d. supposing dimuons, dimuonsGlobal, dimuonsOneTrack and dimuonsOneStndAloneMuon categories has been built)


### parameter set to be overloaded in the configuration file 


from ElectroWeakAnalysis.ZMuMu.goodZToMuMu_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuSameCharge_cfi import *
from ElectroWeakAnalysis.ZMuMu.nonIsolatedZToMuMu_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuOneTrack_cfi import *
from ElectroWeakAnalysis.ZMuMu.goodZToMuMuOneStandAloneMuon_cfi import *



#ZMuMuCategoriesSequences= []
#ZMuMuCategoriesLabels= []

goodZToMuMuSequence = cms.Sequence(
    goodZToMuMu +
    goodZToMuMuAtLeast1HLT
    )

#ZMuMuCategoriesLabels.append("goodZToMuMuAtLeast1HLT")
#ZMuMuCategoriesSequences.append(goodZToMuMuSequence)

goodZToMuMu2HLTSequence = cms.Sequence(
    goodZToMuMu +
    goodZToMuMu2HLT
    )

#ZMuMuCategoriesLabels.append("goodZToMuMu2HLT")
#ZMuMuCategoriesSequences.append(goodZToMuMu2HLTSequence)

goodZToMuMu1HLTSequence = cms.Sequence(
    goodZToMuMu +
    goodZToMuMu1HLT
    )

#ZMuMuCategoriesLabels.append("goodZToMuMu1HLT")
#ZMuMuCategoriesSequences.append(goodZToMuMu1HLTSequence)


goodZToMuMuSameChargeSequence = cms.Sequence(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameCharge +
    goodZToMuMuSameChargeAtLeast1HLT
    )

#ZMuMuCategoriesLabels.append("goodZToMuMuSameChargeAtLeast1HLT")
#ZMuMuCategoriesSequences.append(goodZToMuMuSameChargeSequence)

goodZToMuMuSameCharge2HLTSequence = cms.Sequence(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameCharge +
    goodZToMuMuSameCharge2HLT
    )

#ZMuMuCategoriesLabels.append("goodZToMuMuSameCharge2HLT")
#ZMuMuCategoriesSequences.append(goodZToMuMuSameCharge2HLTSequence)

goodZToMuMuSameCharge1HLTSequence = cms.Sequence(
    dimuonsGlobalSameCharge+
    goodZToMuMuSameCharge +
    goodZToMuMuSameCharge1HLT
    )

#ZMuMuCategoriesLabels.append("goodZToMuMuSameCharge1HLT")
#ZMuMuCategoriesSequences.append(goodZToMuMuSameCharge1HLTSequence)


nonIsolatedZToMuMuSequence = cms.Sequence (
    nonIsolatedZToMuMu +
    nonIsolatedZToMuMuAtLeast1HLT 
)

#ZMuMuCategoriesLabels.append("nonIsolatedZToMuMuAtLeast1HLT")
#ZMuMuCategoriesSequences.append(nonIsolatedZToMuMuSequence)

oneNonIsolatedZToMuMuSequence = cms.Sequence(
    nonIsolatedZToMuMu +
    oneNonIsolatedZToMuMu +
    oneNonIsolatedZToMuMuAtLeast1HLT 
)

#ZMuMuCategoriesLabels.append("oneNonIsolatedZToMuMuAtLeast1HLT")
#ZMuMuCategoriesSequences.append(oneNonIsolatedZToMuMuSequence)

twoNonIsolatedZToMuMuSequence = cms.Sequence(
    nonIsolatedZToMuMu +
    twoNonIsolatedZToMuMu +
    twoNonIsolatedZToMuMuAtLeast1HLT 
)

#ZMuMuCategoriesLabels.append("twoNonIsolatedZToMuMuAtLeast1HLT")
#ZMuMuCategoriesSequences.append(twoNonIsolatedZToMuMuSequence)

goodZToMuMuOneStandAloneMuonSequence = cms.Sequence(
    ~goodZToMuMu + 
    zToMuMuOneStandAloneMuon + 
    goodZToMuMuOneStandAloneMuon +
    goodZToMuMuOneStandAloneMuonFirstHLT 
    )

#ZMuMuCategoriesLabels.append("goodZToMuMuOneStandAloneMuonFirstHLT")
#ZMuMuCategoriesSequences.append(goodZToMuMuOneStandAloneMuonSequence)


goodZToMuMuOneTrackSequence=cms.Sequence(
    ~goodZToMuMu +
    ~zToMuMuOneStandAloneMuon +
    zToMuGlobalMuOneTrack +
    zToMuMuOneTrack +
    goodZToMuMuOneTrack +
    goodZToMuMuOneTrackFirstHLT 
    )


#ZMuMuCategoriesLabels.append("goodZToMuMuOneTrackFirstHLT")
#ZMuMuCategoriesSequences.append(goodZToMuMuOneTrackSequence)





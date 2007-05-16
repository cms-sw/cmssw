#ifndef MuonReco_MuonEnergy_h
#define MuonReco_MuonEnergy_h


namespace reco {
    struct MuonEnergy {
       float em;        // energy deposited in ECAL
       float had;       // energy deposited in HCAL
       float ho;        // energy deposited in HO
       float emS9;     // energy deposited in ECAL in 3x3 towers
       float hadS9;    // energy deposited in HCAL in 3x3 crystals
       float hoS9;     // energy deposited in HO in 3x3 towers
       MuonEnergy():
       em(0),had(0),ho(0),emS9(0),hadS9(0),hoS9(0){ }
       
    };
}
#endif

#ifndef MuonReco_MuonEnergy_h
#define MuonReco_MuonEnergy_h


namespace reco {
    struct MuonEnergy {
       /// CaloTower based energy
       /// total energy
       float tower; 
       /// total energy in 3x3 tower shape
       float towerS9; 
       
       /// RecHit based energy (eta-phi size don't match!)
       /// energy deposited in crossed ECAL crystals
       float em; 
       /// energy deposited in 3x3 ECAL crystal shape around 
       /// crossed crystal
       float emS9;
       /// energy deposited in crossed HCAL tower (RecHits)
       float had;
       /// energy deposited in 3x3 HCAL tower shape around 
       /// crossed tower (RecHits)
       float hadS9;
       /// energy deposited in crossed HO tower (RecHits)
       float ho;
       /// energy deposited in 3x3 HO tower shape around 
       /// crossed tower (RecHits)
       float hoS9;
       
       MuonEnergy():
       tower(0), towerS9(0),
       em(0), emS9(0),
       had(0), hadS9(0),
       ho(0), hoS9(0){ }
       
    };
}
#endif

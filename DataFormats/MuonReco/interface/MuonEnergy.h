#ifndef MuonReco_MuonEnergy_h
#define MuonReco_MuonEnergy_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include <vector>
namespace reco {
    struct HcalMuonRecHit{
      float energy;
      float chi2;
      float time;
      HcalDetId detId;
      HcalMuonRecHit():
      energy(0),chi2(0),time(0){}
    };

    struct MuonEnergy {
       /// CaloTower based energy
       /// total energy
       float tower; 
       /// total energy in 3x3 tower shape
       float towerS9; 
       
       /// ECAL crystal based energy (RecHits)
       /// energy deposited in crossed ECAL crystals
       float em; 
       /// energy deposited in 3x3 ECAL crystal shape around central crystal
       float emS9;
       /// energy deposited in 5x5 ECAL crystal shape around central crystal
       float emS25;
       /// maximal energy of ECAL crystal in the 5x5 shape
       float emMax;
       
       /// HCAL tower based energy (RecHits)
       /// energy deposited in crossed HCAL towers
       float had;
       /// energy deposited in 3x3 HCAL tower shape around central tower
       float hadS9;
       /// maximal energy of HCAL tower in the 3x3 shape
       float hadMax;
       /// energy deposited in crossed HO towers
       float ho;
       /// energy deposited in 3x3 HO tower shape around central tower
       float hoS9;
       
       /// Calorimeter timing
       float ecal_time;
       float ecal_timeError;
       float hcal_time;
       float hcal_timeError;
       
       /// Trajectory position at the calorimeter
       math::XYZPointF ecal_position;
       math::XYZPointF hcal_position;
       
       /// DetId of the central ECAL crystal
       DetId ecal_id;
       
       /// DetId of the central HCAL tower with smallest depth
       DetId hcal_id;
       
       ///
       /// RecHits
       ///
       /// crossed HCAL rechits
       std::vector<HcalMuonRecHit> crossedHadRecHits;

       MuonEnergy():
       tower(0), towerS9(0),
       em(0), emS9(0), emS25(0), emMax(0),
       had(0), hadS9(0), hadMax(0),
	 ho(0), hoS9(0), ecal_time(0), ecal_timeError(0), hcal_time(0), hcal_timeError(0)
	 { }

    };
}
#endif

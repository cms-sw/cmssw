// CaloJet.cc
// $Id$
// Initial Version From Fernando Varela Rodriguez
// Revisions: R. Harris, 19-Oct-2005, modified to work with real 
//            CaloTowers from Jeremy Mans.  Commented out energy
//            fractions until we can figure out how to determine 
//            composition of total energy, and the underlying HB, HE, 
//            HF, HO and Ecal.

//Own header file
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"

#include <numeric>
#include <algorithm>

using namespace std;
using namespace reco;

CaloJet::CaloJet() {
}

/// Constructor from values
CaloJet::CaloJet( double px, double py, double pz, double e,
		  const CaloTowerCollection & caloTowerColl, 
		  const vector<CaloTowerDetId> & indices ) :
  p4_( px, py, pz, e ), towerIndices_( indices ) {
  // 1.- Loop over the tower Ids, 
  // 2.- Get the corresponding CaloTower
  // 3.- Calculate the different CaloJet specific quantities
  vector<double> eECal_i;
  vector<double> eHCal_i;

  //double eInHO = 0.;
  //double eInHB = 0.;
  //double eInHF = 0.;
  //double eInHE = 0.;
 
  n90_ = 1;
  double e90 = energy()*0.9;
  for( vector<CaloTowerDetId>::const_iterator i = towerIndices_.begin(); 
       i != towerIndices_.end(); ++i ) {
    //const CaloTower &aTower =  getTowerFromIndex(caloTowerColl, *i);
    // modifiedt tow work with EDM Sorted collection instead of STL vector
    const CaloTower aTower =  *caloTowerColl.find(*i);

    //Array of energy in EM Towers:
    eECal_i.push_back(aTower.e_em()); 

    //Array of energy in HCAL Towers:
    eHCal_i.push_back(aTower.e_had()); 
  }
 
  //Sort the arrays
  sort(eECal_i.begin(), eECal_i.end());
  sort(eHCal_i.begin(), eHCal_i.end());

  //Highest value in the array is the last element of the array
  maxEInEmTowers_ = eECal_i.back(); 
  maxEInHadTowers_ = eHCal_i.back();
  
  //Calculate the totals
  double totalEECal = accumulate( eECal_i.begin(), eECal_i.end(), 0. );
  double totalEHCal = accumulate( eHCal_i.begin(), eHCal_i.end(), 0. ); 
  
  //Energy fractions:
  energyFractionInHCAL_ = totalEHCal / ( totalEHCal + totalEECal );
  energyFractionInECAL_ = totalEECal / ( totalEHCal + totalEECal );

  //n90 using the sorted list
  for( int i = eECal_i.size() - 1; i <= 0; --i ) {
    if( eECal_i[i] + eHCal_i[i] >= e90 )
      break;
    else
      ++ n90_;
  }
 
} //end of constructor

//Destructor
CaloJet::~CaloJet() {
}

void CaloJet::setTowerIndices(const vector<CaloTowerDetId> & towerIndices) {
  towerIndices_ = towerIndices;
}

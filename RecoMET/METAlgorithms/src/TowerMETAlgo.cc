// File: TowerMETAlgo.cc
// Description:  see TowerMETAlgo.h
// Author: Michael Schmitt, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------

#include "RecoMET/METAlgorithms/interface/TowerMETAlgo.h"

using namespace std;

const int MAX_TOWERS = 100;

TowerMETAlgo::TowerMETAlgo(bool Type1) {

  doType1 = Type1;

}

TowerMETAlgo::~TowerMETAlgo() {}

void TowerMETAlgo::run(const CaloTowerCollection *towers,
                       const CaloJetCollection *rawjets, 
                       const CaloJetCollection *corrjets, 
                       TowerMETCollection &metvec) {

  // Clean up the EDProduct, it should be empty
  metvec.clear();

  double sum_et = 0.;
  double sum_ex = 0.;
  double sum_ey = 0.;
  double sum_ez = 0.;

  // Loop over CaloTowers
  CaloTowerCollection::const_iterator ct_iter;
  for (ct_iter = towers->begin(); ct_iter != towers->end(); ct_iter++) {

    // Get the relevant CaloTower data
    double e     = ct_iter->getE();
    double theta = ct_iter->getTheta();
    double phi   = ct_iter->getPhi();

    // Sum over the transverse energy
    sum_ez += e * cos(theta);
    sum_et += e * sin(theta);
    sum_ex += sum_et * cos(phi);
    sum_ey += sum_et * sin(phi);

  }

  // Calculate the Resultant MET Angle
  double met_phi = calcMETPhi(-sum_ex,-sum_ey);

  // Create a holder for the MET object
  TowerMET cmet;
  cmet.clearMET();

  // Set new TowerMET values
  cmet.setLabel("raw");
  cmet.setMET(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));
  cmet.setMETx(-sum_ex);
  cmet.setMETy(-sum_ey);
  cmet.setMETz(-sum_ez);
  cmet.setPhi(met_phi);
  cmet.setSumEt(sum_et);

  // Raw MET is pushed into the zero position
  metvec.push_back(cmet);

  // ** CORRECTION PHASE **

  // doType1  correct clustered towers
  // doType2  correct all towers
  // doH1     correct using H1 calibration technique

  // Type-1 correction: correct clustered towers
  if (doType1) {

    // Stores (CaloTower Index, Correction)
    double tower_corrections[MAX_TOWERS]; // MAX_TOWERS = 84? 88?
    memset(tower_corrections,0,8 * MAX_TOWERS);

    // Verify that the jets are valid
    if (rawjets == NULL || corrjets == NULL)
      throw "Type-1 MET Correction failed.\n\
             Re: Invalid CaloJetCollection supplied.";

    // Loop over CaloJets; for comparisons, easiest to use operator[]
    for (size_t i = 0; i < rawjets->size(); i++) {

      // Get tower information from jets
      int num_towers = (*rawjets)[i].getNConstituents();
      vector<int> tower_ids = (*rawjets)[i].getTowerIndices();
      
      // Look at the second jet collection
      // vector<int> tmp_ids = (*corrjets)[i].getTowerIndices();
      int tmp_towers = (*corrjets)[i].getNConstituents();

      // Loose check: verify that jets are comparable
      if (tmp_towers != num_towers) {
        throw "Type-1 MET Correction failed.\n\
               Re: Raw and corrected jets don't match.";
      } 

      /* More rigorous checking, optional
      else {

        // Loop over tower indices and verify that they match
        for (int j = 0; j < num_towers; j++) {
          
          // This is overly strict index checking!
          if (tmp_ids[j] != tower_ids[j]) {
            throw "Type-1 MET Correction failed.\n\
                   Re: Towers in raw and corrected jets don't match.";
          }

          // Instead, we could also do looser checking at cost of cycles (ironic?)
          if (find(tmp_towers->begin(),tmp_towers->end(),tower_ids[j]) == tmp_towers->end()) {
            throw "Type-1 MET Correction failed.\n\
                   Re: Towers in raw and corrected jets don't match.";
          }

        }

      }
      */
        
      // Get energy difference (E or P or Et or Pt?)
      double correction = ((*corrjets)[i].getE() - (*rawjets)[i].getE());

      // Store the corrections for the towers
      // Can the same tower be clustered in two jets?
      for (int j = 0; j < num_towers; j++)
        tower_corrections[tower_ids[j]] += correction;
        
    }

    // Cleanup vars for reuse
    cmet.clearMET();
    met_phi = 0.;
    sum_et = 0.;
    sum_ex = 0.;
    sum_ey = 0.;
    sum_ez = 0.;

    // Loop over CaloTowers and make corrections
    for (ct_iter = towers->begin(); ct_iter != towers->end(); ct_iter++) {
                                                                                                              
      // Get the relevant CaloTower data
      int index    = ct_iter->getTowerIndex();
      double e     = ct_iter->getE() + tower_corrections[index];
      double theta = ct_iter->getTheta();
      double phi   = ct_iter->getPhi();
                                                                                                              
      // Sum over the transverse energy
      sum_ez += e * cos(theta);
      sum_et += e * sin(theta);
      sum_ex += sum_et * cos(phi);
      sum_ey += sum_et * sin(phi);
                                                                                                              
    }

    // Calculate the Resultant MET Angle
    met_phi = calcMETPhi(-sum_ex,-sum_ey);

    // Set new TowerMET values
    cmet.setLabel("type1");
    cmet.setPhi(met_phi);
    cmet.setMET(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));
    cmet.setMETx(-sum_ex);
    cmet.setMETy(-sum_ey);
    cmet.setMETz(-sum_ez);
    cmet.setSumEt(sum_et);
  
    // Push the Type-1 correction into the vector
    metvec.push_back(cmet);
    
  }

}

double TowerMETAlgo::calcMETPhi(double METx, double METy) const {

  double phi = 0.;
  double metx = METx;
  double mety = METy;

  // Note: mapped out over [-pi,pi)
  if (metx == 0.) {
    if (mety == 0.) {
      throw "METPhi calculation failed (defaulted to 0.).\n\
             Re: sumEt is zero.";
      return 0.;
    } else if (mety > 0.)
      phi = M_PI_2;
    else
      phi = - M_PI_2;
  } else if (metx > 0.) {
    if (mety > 0.)
      phi = atan(mety/metx);
    else
      phi = atan(mety/metx);
  } else {
    if (mety > 0.)
      phi = atan(mety/metx) + M_PI;
    else
      phi = atan(mety/metx) - M_PI;
  }

  return phi;

}

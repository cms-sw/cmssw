// File: TestMETAlgo.cc
// Description:  see TestMETAlgo.h
// Author: Michael Schmitt, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------

#include "RecoMET/METAlgorithms/interface/TestMETAlgo.h"

using namespace std;

const int MAX_TOWERS = 100;

TestMETAlgo::TestMETAlgo(bool Type1) {

  doType1 = Type1;

}

TestMETAlgo::~TestMETAlgo() {}

void TestMETAlgo::run(const CaloTowerCollection *towers, METCollection &metvec) 
{

  // Clean up the EDProduct, it should be empty
  metvec.clear();

  double sum_et = 0.;
  double sum_ex = 0.;
  double sum_ey = 0.;
  double sum_ez = 0.;

  // Loop over CaloTowers
  CaloTowerCollection::const_iterator ct_iter;
  for (ct_iter = towers->begin(); ct_iter != towers->end(); ct_iter++) 
    {

      // Get the relevant CaloTower data
      //double e     = ct_iter->e;
      //double theta = ct_iter->getTheta();
      double phi   = ct_iter->phi;

      // Sum over the transverse energy
      double et = ct_iter->eT;//10.0 * e * sin(theta);
      sum_ez += 10.0;// * e * cos(theta);
      sum_et += et;
      sum_ex += et * cos(phi);
      sum_ey += et * sin(phi);

    }

  // Calculate the Resultant MET Angle
  double met_phi = calcMETPhi(-sum_ex,-sum_ey);

  // Create a holder for the MET object
  MET cmet;
  cmet.clearMET();

  // Set new TowerMET values
  cmet.setLabel("test");
  cmet.setMET(sqrt(sum_ex * sum_ex + sum_ey * sum_ey));
  cmet.setMETx(-sum_ex);
  cmet.setMETy(-sum_ey);
  cmet.setMETz(-sum_ez);
  cmet.setPhi(met_phi);
  cmet.setSumEt(sum_et);

  // Raw MET is pushed into the zero position
  metvec.push_back(cmet);
}

double TestMETAlgo::calcMETPhi(double METx, double METy) const {

  double phi = 0.;
  double metx = METx;
  double mety = METy;

  // Note: mapped out over [-pi,pi)
  if (metx == 0.) 
    {
      if (mety == 0.) 
	{
	  //throw "METPhi calculation failed (defaulted to 0.).\n\
          //   Re: sumEt is zero.";
	  return 0.;
	} 
      else if (mety > 0.)
	phi = M_PI_2;
      else
	phi = - M_PI_2;
    } 
  else if (metx > 0.) 
    {
      if (mety > 0.)
	phi = atan(mety/metx);
      else
	phi = atan(mety/metx);
    } 
  else 
    {
      if (mety > 0.)
	phi = atan(mety/metx) + M_PI;
      else
	phi = atan(mety/metx) - M_PI;
    }
  return phi;
}

// File: CorrMETAlgo.cc
// Description:  see CorrMETAlgo.h
// Author: R. Cavanaugh, The University of Florida
// Creation Date:  MHS May 31, 2005 Initial version.
//
//--------------------------------------------

#include "RecoMET/METAlgorithms/interface/CorrMETAlgo.h"

using namespace std;

const int MAX_TOWERS = 100;

CorrMETAlgo::CorrMETAlgo(bool Type1) {

  doType1 = Type1;

}

CorrMETAlgo::~CorrMETAlgo() {}

void CorrMETAlgo::run(const METCollection *rawmet, METCollection &metvec) 
{

  // Clean up the EDProduct, it should be empty
  metvec.clear();

  double sum_et = 0.;
  //double sum_ex = 0.;
  //double sum_ey = 0.;
  //double sum_ez = 0.;

  // Loop over CaloTowers
  METCollection::const_iterator met_iter;
  for (met_iter = rawmet->begin(); met_iter != rawmet->end(); met_iter++) 
    {
      // Get the relevant raw MET data
      sum_et     = met_iter->getMET();
    }

  // Calculate the Resultant MET Angle
  //double met_phi = calcMETPhi(-sum_ex,-sum_ey);

  // Create a holder for the MET object
  MET cmet;
  cmet.clearMET();

  // Set new TowerMET values
  cmet.setLabel("corr");
  cmet.setMET( sum_et );
  cmet.setMETx( sum_et );
  cmet.setMETy( sum_et );
  cmet.setMETz( sum_et );
  cmet.setPhi( sum_et );
  cmet.setSumEt( sum_et );

  // Raw MET is pushed into the zero position
  metvec.push_back(cmet);
}

double CorrMETAlgo::calcMETPhi(double METx, double METy) const {

  double phi = 0.;
  double metx = METx;
  double mety = METy;

  // Note: mapped out over [-pi,pi)
  if (metx == 0.) 
    {
      if (mety == 0.) 
	{
	  throw "METPhi calculation failed (defaulted to 0.).\n\
             Re: sumEt is zero.";
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

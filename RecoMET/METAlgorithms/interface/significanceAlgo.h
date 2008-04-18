#ifndef SIGMET_ASIGNIFICANCE_H
#define SIGMET_ASIGNIFICANCE_H
// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SigInputObj
// 
/**\class METSignificance SigInputObj.h RecoMET/METAlgorithms/include/SigInputObj.h

 Description: <one line class summary>

 Implementation:

  Purpose:
  
  This subroutine takes in a vector of type SigInputObj that includes
  all of the physics objects in the event.  It then calculates
  the log significance of the MET of the event.

  The Significance (S) is defined as:
    ln(S) = 1/2 Chisq_0
  where Chisq_0 is the value of Chi squared at MET=0.

*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id$
//
//

//  
//  Purpose:
//  
//  This subroutine takes in a vector of type SigInputObj that includes
//  all of the physics objects in the event.  It then calculates
//  the log significance of the MET of the event.
//
//  The Significance (S) is defined as:
//    ln(S) = 1/2 Chisq_0
//  where Chisq_0 is the value of Chi squared at MET=0.
//
//
// $Id: significanceAlgo.h,v 1.2 2008/03/26 17:52:19 kstory Exp $
// 
// Revision history
// 
// $Log: significanceAlgo.h,v $
// Revision 1.2  2008/03/26 17:52:19  kstory
// Tower Based Algorithm.  significanceAlgo now using matrix operations.
//
// Revision 1.1.1.1  2008/02/29 22:25:02  kstory
// Tower-Based Algorithm
//
// Revision 1.2  2008/02/13 13:14:37  fblekman
// updated version with electrons
//
// Revision 1.3  2007/12/07 00:20:10  kstory
// MET phi was changed to have a range of [-pi, pi]
//
// Revision 1.2  2007/11/30 22:02:33  kstory
// Changed the arguments to allow calculation of the total MET and the MET_phi.
//
//

#include <vector>
#include <string>
#include <iostream>
#include <sstream>

#include "RecoMET/METAlgorithms/interface/SigInputObj.h"
#include "TMatrixTBase.h"
#include "TMatrixD.h"
#include "TVectorD.h"

namespace metsig{
  //double chisq( double x,  double y,  double X0,  double Y0,  double rho,  double sigma_x,  double sigma_y);
  void rotateMatrix( Double_t theta, TMatrixD &v);  
  double ASignificance(const std::vector<metsig::SigInputObj>& EventVec, double& met_r, double& met_phi, double& met_set);
}

#endif

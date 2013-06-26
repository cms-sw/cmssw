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
// $Id: significanceAlgo.h,v 1.3 2012/09/11 11:40:12 veelken Exp $
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
// $Id: significanceAlgo.h,v 1.3 2012/09/11 11:40:12 veelken Exp $
// 
// Revision history
// 
// $Log: significanceAlgo.h,v $
// Revision 1.3  2012/09/11 11:40:12  veelken
// SigInputObj moved to DataFormats/METReco
//
// Revision 1.2  2009/10/21 11:27:11  fblekman
// merged version with cvs head - includes new interfaces for MET significance to make it possible to correct MET objects later and also correct the signficance.
//
// Revision 1.1  2008/04/18 10:12:55  fblekman
// First implementation (very preliminary) of missing ET significance algorithm.
// This code is currently still heavily under development so please bear with us.
//
// cheers,
//
//   Freya (for the Cornell MET significance group)
//
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

#include "DataFormats/METReco/interface/SigInputObj.h"
#include "TMatrixTBase.h"
#include "TMatrixD.h"
#include "TVectorD.h"

namespace metsig{
  class significanceAlgo{
  public:
    significanceAlgo();
    ~significanceAlgo();

    const void addSignifMatrix(const TMatrixD &input);
    const void setSignifMatrix(const TMatrixD &input,const double &met_r, const double &met_phi, const double &met_set);
    const double significance(double& met_r, double& met_phi, double& met_set);
    const void addObjects(const std::vector<metsig::SigInputObj>& EventVec);
    const void subtractObjects(const std::vector<metsig::SigInputObj>& EventVec);
    TMatrixD getSignifMatrix() const {return signifmatrix_;}
    //    const std::vector<metsig::SigInputObj> eventVec(){return eventVec_;}
  private:
    void rotateMatrix( Double_t theta, TMatrixD &v);  

    //    std::vector<metsig::SigInputObj> eventVec_;
    TMatrixD signifmatrix_;
    // workers:
    double set_worker_;
    double xmet_;
    double ymet_;
  };
}

#endif

#ifndef RECOMET_METALGORITHMS_SIGNALGORESOLUTIONS_H
#define RECOMET_METALGORITHMS_SIGNALGORESOLUTIONS_H

// -*- C++ -*-
//
// Package:    METAlgorithms
// Class:      SignAlgoResolutions
// 
/**\class METSignificance SignAlgoResolutions.h RecoMET/METAlgorithms/include/SignAlgoResolutions.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Kyle Story, Freya Blekman (Cornell University)
//         Created:  Fri Apr 18 11:58:33 CEST 2008
// $Id$
//
//

#include "DataFormats/Math/interface/LorentzVector.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <map>
#include <iostream>
#include <vector>

namespace metsig {

  enum resolutionType { caloEE, caloEB, caloHE, caloHO, caloHF, caloHB, jet, electron, tau, muon };
  enum resolutionFunc { ET, PHI };

  class SignAlgoResolutions{
    
  public:
    SignAlgoResolutions():functionmap_(){;}
    SignAlgoResolutions(const edm::ParameterSet &iConfig);

    void addResolutions(const edm::ParameterSet &iConfig);
    double eval(const resolutionType & type, const resolutionFunc & func, const double & et, const double & phi, const double & eta) const; // for example getvalue(caloHF,ET,et,phi,eta);
    
  private:
    double getfunc(const resolutionType & type,const resolutionFunc & func,  std::vector<double> & x) const;
    void addfunction(const resolutionType type, const resolutionFunc func, std::vector<double> parameters);
    
    typedef std::pair<metsig::resolutionType, metsig::resolutionFunc> functionCombo;
    typedef std::vector<double> functionPars;
    std::map<functionCombo,functionPars> functionmap_;
 
    double EtFunction( const functionPars &x,  const functionPars &  par) const;
    double PhiFunction( const functionPars &x,  const functionPars & par) const;
  };
}
#endif

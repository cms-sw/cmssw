// -*- C++ -*-
//
// Package:    Castor
// Class:      Castor
// 
/**\class Castor Egamma.cc RecoLocalCalo/Castor/src/Egamma.cc

 Description: Functions to check found CastorJets and take them as Egamma's given some conditions

 Implementation:
     . 
*/
//
// Original Author:  Hans Van Haevermaet
//         Created:  Sat May 24 12:00:56 CET 2008
// $Id: Egamma.cc,v 1.1.2.1 2008/08/30 20:46:31 hvanhaev Exp $
//
//

// includes
#include <iostream>
#include <algorithm>
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorJet.h"
#include "DataFormats/CastorReco/interface/CastorEgamma.h"
#include "RecoLocalCalo/Castor/interface/Egamma.h"

// namespaces
using namespace std;
using namespace reco;
using namespace math;
using namespace edm;

// typedefs
typedef math::XYZPointD Point;
typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;

// main public function being executed, is called to give results      
CastorEgammaCollection Egamma::runEgamma (const CastorJetCollection inputjets, const double minratio, const double maxwidth, const double
maxdepth) {
  
  // get and check input size
  int nJets = inputjets.size();
  if (nJets==0) {
  	cout << "Warning: You are trying to run the Egamma algorithm with 0 input jets. \n";
  }
  
  // define output
  CastorEgammaCollection egammas;
  egammas.reserve(inputjets.size());
  
  // check for Egamma conditions
  for (size_t i=0;i<inputjets.size();i++) {
  	if ( inputjets[i].emtotRatio() > minratio && (inputjets[i].width() < maxwidth && inputjets[i].depth() < maxdepth)) {
  		TowerPoint temp(1.,inputjets[i].eta(),inputjets[i].phi());
		Point position(temp);
		CastorJetCollection usedJets;
		usedJets.push_back(inputjets[i]);
  		egammas.push_back(CastorEgamma(inputjets[i].energy(),position,inputjets[i].emEnergy(),inputjets[i].hadEnergy(),inputjets[i].emtotRatio(),
		inputjets[i].width(),inputjets[i].depth(),usedJets));
	}
  } 
  
  return egammas;
}


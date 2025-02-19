#ifndef PatBTagCommonHistos_H_
#define PatBTagCommonHistos_H_

// -*- C++ -*-
//
// Package:    PatBTag
// Class:      PatBTagCommonHistos
// 
/**\class PatBTagCommonHistos PatBTagCommonHistos.h

 Description: <Define and Fill common set of histograms depending on flavor and tagger>

 Implementation:
 
 Create a container of histograms. 
*/
//
// Original Author:  J. E. Ramirez
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Common/interface/View.h"
#include "PhysicsTools/PatUtils/interface/bJetSelector.h"

#include "TH1D.h"
#include "TH2D.h"
#include <map>

//
// class declaration
//

class PatBTagCommonHistos {
   public:
      explicit PatBTagCommonHistos(const edm::ParameterSet&);
      ~PatBTagCommonHistos();

      void Set(std::string);
      void Sumw2();
      void Fill(edm::View<pat::Jet>::const_iterator&, std::string);
   private:

      // ----------member data ---------------------------


  std::string  flavor;
  std::map<std::string,TH1D*> histocontainer_;   // simple map to contain all histograms. Histograms are booked in the beginJob() method
  std::map<std::string,TH2D*> h2_; // simple map to contain 2D  histograms. Histograms are booked in the beginJob() method
  std::string  BTagdiscriminator_;
  std::string  BTagpurity_;
  double  BTagdisccut_;
  bJetSelector BTagger;
};

#endif

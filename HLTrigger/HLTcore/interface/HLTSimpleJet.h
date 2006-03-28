#ifndef HLTSimpleJet_h
#define HLTSimpleJet_h

// -*- C++ -*-
//
// Package:    HLTSimpleJet
// Class:      HLTSimpleJet
// 
/**\class HLTSimpleJet

 Description: A very basic HLT trigger for jets

 Implementation:
     A filter is provided cutting on the number of jets above a pt cut
*/
//
// Original Author:  Martin GRUNEWALD
//         Created:  Thu Mar 23 10:00:22 CET 2006
// $Id: HLTSimpleJet.h,v 1.2 2006/03/23 16:34:01 gruen Exp $
//
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class HLTSimpleJet : public edm::EDFilter {

   public:
      explicit HLTSimpleJet(const edm::ParameterSet&);
      ~HLTSimpleJet();

      virtual bool filter(const edm::Event&, const edm::EventSetup&);

   private:
      std::string module_;  // module label for input jets
      double ptcut_;        // pt cut in GeV 
      int    njcut_;        // number of jets required
};

#endif //HLTSimpleJet_h

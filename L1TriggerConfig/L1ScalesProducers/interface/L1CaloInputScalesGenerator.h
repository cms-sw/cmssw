// -*- C++ -*-
//
// Package:    L1ScalesProducers
// Class:      L1CaloInputScalesGenerator
// 
/**\class L1CaloInputScalesGenerator L1CaloInputScalesGenerator.cc L1TriggerConfig/L1ScalesProducers/src/L1CaloInputScalesGenerator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  pts/140
//         Created:  Wed Jun 25 16:40:01 CEST 2008
// $Id: L1CaloInputScalesGenerator.h,v 1.2 2009/12/18 20:44:59 wmtan Exp $
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class L1CaloInputScalesGenerator : public edm::EDAnalyzer {
   public:
      explicit L1CaloInputScalesGenerator(const edm::ParameterSet&);
      ~L1CaloInputScalesGenerator();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};

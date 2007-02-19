// -*- C++ -*-
//
// Package:    L1GctTestAnalyzer
// Class:      L1GctTestAnalyzer
// 
/**\class L1GctTestAnalyzer L1GctTestAnalyzer.cc L1Trigger/GlobalCaloTrigger/test/L1GctTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Thu May 18 16:45:23 CEST 2006
// $Id: L1GctTestAnalyzer.h,v 1.1 2006/05/18 16:52:34 jbrooke Exp $
//
//


// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class L1GctTestAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1GctTestAnalyzer(const edm::ParameterSet&);
      ~L1GctTestAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);

  void doEM(const edm::Event&, std::string label);

   private:
      // ----------member data ---------------------------

  std::string rawLabel;
  std::string emuLabel;
  std::string outFilename;

  //  std::ostream outFile;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//

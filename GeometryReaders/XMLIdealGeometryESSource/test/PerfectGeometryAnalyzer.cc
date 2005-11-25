// -*- C++ -*-
//
// Package:    PerfectGeometryAnalyzer
// Class:      PerfectGeometryAnalyzer
// 
/**\class PerfectGeometryAnalyzer PerfectGeometryAnalyzer.cc test/PerfectGeometryAnalyzer/src/PerfectGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tommaso Boccali
//         Created:  Tue Jul 26 08:47:57 CEST 2005
// $Id: PerfectGeometryAnalyzer.cc,v 1.4 2005/11/14 13:57:46 case Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//
// class decleration
//

class PerfectGeometryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit PerfectGeometryAnalyzer( const edm::ParameterSet& );
      ~PerfectGeometryAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
   private:
      // ----------member data ---------------------------
      std::string label_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PerfectGeometryAnalyzer::PerfectGeometryAnalyzer( const edm::ParameterSet& iConfig ) :
   label_(iConfig.getUntrackedParameter<std::string>("label",""))
{
   //now do what ever initialization is needed
   
}


PerfectGeometryAnalyzer::~PerfectGeometryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PerfectGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   //
   // get the DDCompactView
   //
   edm::ESHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get(label_, pDD );
   //std::cout <<" got the geometry"<<std::endl;
   try {
      DDExpandedView ex(*pDD);
      //std::cout <<" made the expanded view"<<std::endl;
      // ex.firstChild();
      std::cout << " Top node is a "<< ex.logicalPart() << std::endl;
   }catch(const DDLogicalPart& iException){
      throw cms::Exception("Geometry")
      <<"A DDLogicalPart was thrown \""<<iException<<"\"";
   }
}

//define this as a plug-in
DEFINE_FWK_MODULE(PerfectGeometryAnalyzer)

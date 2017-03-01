// -*- C++ -*-
//
// Package:    TestSpecParAnalyzer
// Class:      TestSpecParAnalyzer
// 
/**\class TestSpecParAnalyzer TestSpecParAnalyzer.cc test/TestSpecParAnalyzer/src/TestSpecParAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDRoot.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "CondFormats/Common/interface/FileBlob.h"
#include "Geometry/Records/interface/GeometryFileRcd.h"

//
// class decleration
//

class TestSpecParAnalyzer : public edm::EDAnalyzer {
public:
  explicit TestSpecParAnalyzer( const edm::ParameterSet& );
  ~TestSpecParAnalyzer();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
  std::string specName_;
  std::string specStrValue_;
  double specDblValue_;
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
TestSpecParAnalyzer::TestSpecParAnalyzer( const edm::ParameterSet& iConfig ) :
  specName_(iConfig.getParameter<std::string>("specName")),
  specStrValue_(iConfig.getUntrackedParameter<std::string>("specStrValue", "frederf")),
  specDblValue_(iConfig.getUntrackedParameter<double>("specDblValue", 0.0))
{ }


TestSpecParAnalyzer::~TestSpecParAnalyzer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TestSpecParAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   std::cout << "Here I am " << std::endl;
   edm::ESTransientHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get("", pDD );
   const DDCompactView& cpv(*pDD);
   if ( specStrValue_ != "frederf" ) {
     std::cout << "specName = " << specName_ << " and specStrValue = " << specStrValue_ << std::endl;
     DDValue fval(specName_, specStrValue_, 0.0);
     DDSpecificsFilter filter;
     filter.setCriteria(fval, // name & value of a variable 
			DDCompOp::equals,
			DDLogOp::AND, 
			true, // compare strings otherwise doubles
			true // use merged-specifics or simple-specifics
			);
     DDFilteredView fv(cpv);
     fv.addFilter(filter);
     bool doit = fv.firstChild();
     std::vector<const DDsvalues_type *> spec = fv.specifics();
     std::vector<const DDsvalues_type *>::const_iterator spit = spec.begin();
     while (doit) {
       spec = fv.specifics();
       spit = spec.begin();
       std::cout << fv.geoHistory() << std::endl;
       for ( ; spit != spec.end() ; ++spit ) {
	 DDsvalues_type::const_iterator it = (**spit).begin();
	 for (;  it != (**spit).end(); it++) {
	   std::cout << "\t" << it->second.name() << std::endl;
	   if ( it->second.isEvaluated() ) {
	     for ( size_t i = 0; i < it->second.doubles().size(); ++i) {
	       std::cout << "\t\t" << it->second.doubles()[i] << std::endl;
	     }
	   } else {
	     for ( size_t i = 0 ; i < it->second.strings().size(); ++i) {
	       std::cout << "\t\t" << it->second.strings()[i] << std::endl;
	     }
	   }
	 }
       }
       doit = fv.next();
     }

   } else {
     std::cout << "double spec value not implemented" << std::endl;
   }

   std::cout << "finished" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(TestSpecParAnalyzer);

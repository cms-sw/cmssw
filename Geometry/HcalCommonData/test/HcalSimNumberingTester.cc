// -*- C++ -*-
//
// Package:    HcalSimNumberingTester
// Class:      HcalSimNumberingTester
// 
/**\class HcalSimNumberingTester HcalSimNumberingTester.cc test/HcalSimNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2013/12/26
// $Id: HcalSimNumberingTester.cc,v 1.0 2013/12/26 14:06:07 sunanda Exp $
//
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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDSimConstants.h"

#include "CoralBase/Exception.h"

//
// class decleration
//

class HcalSimNumberingTester : public edm::EDAnalyzer {
public:
  explicit HcalSimNumberingTester( const edm::ParameterSet& );
  ~HcalSimNumberingTester();

  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  // ----------member data ---------------------------
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
HcalSimNumberingTester::HcalSimNumberingTester(const edm::ParameterSet& ) {}


HcalSimNumberingTester::~HcalSimNumberingTester() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void HcalSimNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
   using namespace edm;

   std::cout << "Here I am " << std::endl;

   edm::ESHandle<HcalDDDSimConstants> pHSNDC;
   edm::ESTransientHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get( pDD );
   iSetup.get<HcalSimNumberingRecord>().get( pHSNDC );

   try {
      DDExpandedView epv(*pDD);
      std::cout << " without firstchild or next... epv.logicalPart() =" << epv.logicalPart() << std::endl;
   }catch(const DDLogicalPart& iException){
      throw cms::Exception("Geometry")
	<<"DDORAReader::readDB caught a DDLogicalPart exception: \""<<iException<<"\"";
   } catch (const coral::Exception& e) {
      throw cms::Exception("Geometry")
	<<"DDORAReader::readDB caught coral::Exception: \""<<e.what()<<"\"";
   } catch ( std::exception& e ) {
     throw cms::Exception("Geometry")
       <<  "DDORAReader::readDB caught std::exception: \"" << e.what() << "\"";
   } catch ( ... ) {
     throw cms::Exception("Geometry")
       <<  "DDORAReader::readDB caught UNKNOWN!!! exception." << std::endl;
   }
   std::cout << "about to de-reference the edm::ESHandle<HcalDDDSimConstants> pHSNDC" << std::endl;
   const HcalDDDSimConstants hdc (*pHSNDC);
   std::cout << "about to getPhiOff and getPhiBin for 0..2" << std::endl;
   for (int i=0; i<3; ++i) {
     double foff = hdc.getPhiOff(i);
     double fbin = hdc.getPhiBin(i);
     std::cout << "PhiOff[" << i << "] = " << foff << "  PhiBin[" << i << "] = " << fbin << std::endl;
   }
   hdc.printTiles();
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalSimNumberingTester);

// -*- C++ -*-
//
// Package:    ShashlikNumberingTester
// Class:      ShashlikNumberingTester
// 
/**\class ShashlikNumberingTester ShashlikNumberingTester.cc test/ShashlikNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2014/03/21
// $Id: ShashlikNumberingTester.cc,v 1.0 2014/032/21 14:06:07 sunanda Exp $
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
#include "Geometry/Records/interface/ShashlikNumberingRecord.h"
#include "Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h"

#include "CoralBase/Exception.h"

//
// class decleration
//

class ShashlikNumberingTester : public edm::EDAnalyzer {
public:
  explicit ShashlikNumberingTester( const edm::ParameterSet& );
  ~ShashlikNumberingTester();

  
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
ShashlikNumberingTester::ShashlikNumberingTester(const edm::ParameterSet& ) {}


ShashlikNumberingTester::~ShashlikNumberingTester() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void ShashlikNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
  
  edm::ESHandle<ShashlikDDDConstants> pSNDC;
  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get( pDD );
  iSetup.get<ShashlikNumberingRecord>().get( pSNDC );

  try {
    DDExpandedView epv(*pDD);
    std::cout << " without firstchild or next... epv.logicalPart() =" << epv.logicalPart() << std::endl;
  } catch(const DDLogicalPart& iException) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught a DDLogicalPart exception: \""<<iException<<"\"";
  } catch (const coral::Exception& e) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught coral::Exception: \""<<e.what()<<"\"";
  } catch ( std::exception& e ) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught std::exception: \"" << e.what() << "\"";
  } catch ( ... ) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught UNKNOWN!!! exception." << std::endl;
  }
  std::cout << "about to de-reference the edm::ESHandle<ShashlikDDDConstants> pSNDC" << std::endl;
  const ShashlikDDDConstants sdc(*pSNDC);
  std::cout << "SuperModules = " << sdc.getSuperModules() << " Modules = " 
	    << sdc.getModules() << "  Number of Rows/Columns = " 
	    << sdc.getCols() << std::endl;
  for (int sm=1; sm<=sdc.getSuperModules(); ++sm) {
    for (int mod=1; mod<=sdc.getModules(); ++mod) {
      if (sdc.isValidSMM(sm,mod)) {
	std::pair<int,int> ixy = sdc.getXY(sm,mod);
	std::pair<int,int> ism = sdc.getSMM(ixy.first,ixy.second);
	std::string flag = (ism.first == sm && ism.second == mod) ? "OK" : "ERROR";
	std::cout << "Input SM/Module " << sm << ":" << mod
		  << " iX/iY " << ixy.first << ":" << ixy.second
		  << " o/p SM/Module " << ism.first << ":" << ism.second
		  << " Valid " << sdc.isValidXY(ixy.first,ixy.second)
		  << " Flag " << flag << std::endl;
      }
    }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(ShashlikNumberingTester);

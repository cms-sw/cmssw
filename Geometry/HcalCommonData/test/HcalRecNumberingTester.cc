// -*- C++ -*-
//
// Package:    HcalRecNumberingTester
// Class:      HcalRecNumberingTester
// 
/**\class HcalRecNumberingTester HcalRecNumberingTester.cc test/HcalRecNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2013/12/26
// $Id: HcalRecNumberingTester.cc,v 1.0 2013/12/26 14:06:07 sunanda Exp $
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
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"

#include "CoralBase/Exception.h"

//
// class decleration
//

class HcalRecNumberingTester : public edm::EDAnalyzer {
public:
  explicit HcalRecNumberingTester( const edm::ParameterSet& );
  ~HcalRecNumberingTester();

  
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
HcalRecNumberingTester::HcalRecNumberingTester(const edm::ParameterSet& ) {}


HcalRecNumberingTester::~HcalRecNumberingTester() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void HcalRecNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ) {
   using namespace edm;

   std::cout << "Here I am " << std::endl;

   edm::ESHandle<HcalDDDRecConstants> pHSNDC;
   edm::ESTransientHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get( pDD );
   iSetup.get<HcalRecNumberingRecord>().get( pHSNDC );

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
   std::cout << "about to de-reference the edm::ESHandle<HcalDDDRecConstants> pHSNDC" << std::endl;
   const HcalDDDRecConstants hdc (*pHSNDC);
   std::cout << "about to getPhiOff and getPhiBin for 0..2" << std::endl;
   int neta = hdc.getNEta();
   std::cout << neta << " eta bins with phi off set for barrel = " 
	     << hdc.getPhiOff(0) << ", endcap = " << hdc.getPhiOff(1) 
	     << std::endl;
   for (int i=0; i<neta; ++i) {
     std::pair<double,double> etas   = hdc.getEtaLimit(i);
     double                   fbin   = hdc.getPhiBin(i);
     std::vector<int>         depths = hdc.getDepth(i);
     std::cout << "EtaBin[" << i << "]: EtaLimit = (" << etas.first << ":"
	       << etas.second << ")  phiBin = " << fbin << " depths = (";
     for (unsigned int k=0; k<depths.size(); ++k) {
       if (k == 0) std::cout << depths[k];
       else        std::cout << ", " << depths[k];
     }
     std::cout << ")" << std::endl;
   }
   std::vector<HcalDDDRecConstants::HcalEtaBin> hbar = hdc.getEtaBins(0);
   std::vector<HcalDDDRecConstants::HcalEtaBin> hcap = hdc.getEtaBins(1);
   std::cout << "Topology Mode " << hdc.getTopoMode() 
	     << " HB with " << hbar.size() << " eta sectors and HE with "
	     << hcap.size() << " eta sectors" << std::endl;
   std::vector<HcalCellType> hbcell = hdc.HcalCellTypes(HcalBarrel);
   std::vector<HcalCellType> hecell = hdc.HcalCellTypes(HcalEndcap);
   std::cout << "HB with " << hbcell.size() << " cells and HE with "
	     << hecell.size() << " cells" << std::endl;
}


//define this as a plug-in
DEFINE_FWK_MODULE(HcalRecNumberingTester);

// -*- C++ -*-
//
// Package:    MuonNumberingTester
// Class:      MuonNumberingTester
// 
/**\class MuonNumberingTester MuonNumberingTester.cc test/MuonNumberingTester/src/MuonNumberingTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Michael Case
//         Created:  Mon 2006/10/02
// $Id: MuonNumberingTester.cc,v 1.4 2010/08/10 14:06:07 innocent Exp $
//
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include "CoralBase/Exception.h"

class MuonNumberingTester : public edm::EDAnalyzer {
public:
  explicit MuonNumberingTester( const edm::ParameterSet& );
  ~MuonNumberingTester();
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
};

MuonNumberingTester::MuonNumberingTester( const edm::ParameterSet& iConfig )
{
}

MuonNumberingTester::~MuonNumberingTester()
{
}

void
MuonNumberingTester::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   ESHandle<MuonDDDConstants> pMNDC;
   ESTransientHandle<DDCompactView> pDD;
   iSetup.get<IdealGeometryRecord>().get( pDD );
   iSetup.get<MuonNumberingRecord>().get( pMNDC );

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
   std::cout << "set the toFind string to \"level\"" << std::endl;
   std::string toFind("level");
   std::cout << "about to de-reference the edm::ESHandle<MuonDDDConstants> pMNDC" << std::endl;
   const MuonDDDConstants mdc (*pMNDC);
   std::cout << "about to getValue( toFind )" << std::endl;
   int level = mdc.getValue( toFind );
   std::cout << "level = " <<  level << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonNumberingTester);

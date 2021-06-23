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
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/MuonNumberingRecord.h"
#include "Geometry/MuonNumbering/interface/MuonDDDConstants.h"

#include "CoralBase/Exception.h"

class MuonNumberingTester : public edm::one::EDAnalyzer<> {
public:
  explicit MuonNumberingTester(const edm::ParameterSet&);
  ~MuonNumberingTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> tokDDD_;
  edm::ESGetToken<MuonDDDConstants, MuonNumberingRecord> tokMuon_;
};

MuonNumberingTester::MuonNumberingTester(const edm::ParameterSet& iConfig)
    : tokDDD_{esConsumes<DDCompactView, IdealGeometryRecord>(edm::ESInputTag{})},
      tokMuon_{esConsumes<MuonDDDConstants, MuonNumberingRecord>(edm::ESInputTag{})} {}

MuonNumberingTester::~MuonNumberingTester() {}

// ------------ method called to produce the data  ------------
void MuonNumberingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  std::cout << "Here I am " << std::endl;

  const auto& pDD = iSetup.getData(tokDDD_);
  const auto& pMNDC = iSetup.getData(tokMuon_);

  try {
    DDExpandedView epv(pDD);
    std::cout << " without firstchild or next... epv.logicalPart() =" << epv.logicalPart() << std::endl;
  } catch (const DDLogicalPart& iException) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught a DDLogicalPart exception: \"" << iException
                                     << "\"";
  } catch (const coral::Exception& e) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught coral::Exception: \"" << e.what() << "\"";
  } catch (std::exception& e) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught std::exception: \"" << e.what() << "\"";
  } catch (...) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught UNKNOWN!!! exception." << std::endl;
  }
  std::cout << "set the toFind string to \"level\"" << std::endl;
  std::string toFind("level");
  std::cout << "about to de-reference the edm::ESHandle<MuonDDDConstants> pMNDC" << std::endl;
  const MuonDDDConstants mdc(pMNDC);
  std::cout << "about to getValue( toFind )" << std::endl;
  int level = mdc.getValue(toFind);
  std::cout << "level = " << level << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonNumberingTester);

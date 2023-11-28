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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
  ~MuonNumberingTester() override = default;

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

// ------------ method called to produce the data  ------------
void MuonNumberingTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::LogVerbatim("MuonNumbering") << "Here I am ";

  const auto& pDD = iSetup.getData(tokDDD_);
  const auto& pMNDC = iSetup.getData(tokMuon_);

  try {
    DDExpandedView epv(pDD);
    edm::LogVerbatim("MuonNumbering") << " without firstchild or next... epv.logicalPart() =" << epv.logicalPart();
  } catch (const DDLogicalPart& iException) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught a DDLogicalPart exception: \"" << iException
                                     << "\"";
  } catch (const coral::Exception& e) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught coral::Exception: \"" << e.what() << "\"";
  } catch (std::exception& e) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught std::exception: \"" << e.what() << "\"";
  } catch (...) {
    throw cms::Exception("Geometry") << "DDORAReader::readDB caught UNKNOWN!!! exception.";
  }
  edm::LogVerbatim("MuonNumbering") << "set the toFind string to \"level\"";
  std::string toFind("level");
  edm::LogVerbatim("MuonNumbering") << "about to de-reference the edm::ESHandle<MuonDDDConstants> pMNDC";
  const MuonDDDConstants mdc(pMNDC);
  edm::LogVerbatim("MuonNumbering") << "about to getValue( toFind )";
  int level = mdc.getValue(toFind);
  edm::LogVerbatim("MuonNumbering") << "level = " << level;
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonNumberingTester);

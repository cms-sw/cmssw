// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

//STL headers
#include <vector>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>

class CSCGeometryDump : public edm::one::EDAnalyzer<> {
public:
  explicit CSCGeometryDump(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool verbose_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> tokCSC_;
  const CSCGeometry* cscGeometry_;
};

CSCGeometryDump::CSCGeometryDump(const edm::ParameterSet& iC)
    : verbose_(iC.getParameter<bool>("verbose")),
      tokCSC_{esConsumes<CSCGeometry, MuonGeometryRecord>(edm::ESInputTag{})} {}

void CSCGeometryDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("verbose", false);
  descriptions.add("cscGeometryDump", desc);
}

void CSCGeometryDump::analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {
  cscGeometry_ = &eventSetup.getData(tokCSC_);
  auto const& chambers = cscGeometry_->chambers();
  edm::LogVerbatim("CSCGeometry") << "CSCGeometry found with " << chambers.size() << " chambers\n";
  for (unsigned int k1 = 0; k1 < chambers.size(); ++k1) {
    auto const& layers = chambers[k1]->layers();
    auto spec = chambers[k1]->specs();
    auto const& id = chambers[k1]->id();
    edm::LogVerbatim("CSCGeometry") << "\nChamber " << k1 << ":" << spec->chamberType() << ":"
                                    << spec->chamberTypeName() << ": (E:" << id.endcap() << " S:" << id.station()
                                    << " R:" << id.ring() << " C:" << id.chamber() << " L:" << id.layer() << ") with "
                                    << layers.size() << " layers ";

    if (verbose_)
      edm::LogVerbatim("CSCGeometry") << "\nStrip Resolution " << spec->stripResolution() << "\nWire resolution "
                                      << spec->wireResolution() << "\nEfficiency " << spec->efficiency()
                                      << "\nTime Window " << spec->timeWindow()
                                      << "\nNeutron hit rate per CSC layer per event " << spec->neutronRate()
                                      << "\nNumber of strips in one chamber " << spec->nStrips()
                                      << "\nNumber of strips nodes " << spec->nNodes() << "\nNumber of wires per group "
                                      << spec->nWiresPerGroup() << "\nNumber of floating strips "
                                      << spec->nFloatingStrips() << "\nStrip pitch in phi, in radians "
                                      << spec->stripPhiPitch() << "\nOffset to centre to intersection, in cm "
                                      << spec->ctiOffset() << "\nWire spacing, in cm " << spec->wireSpacing()
                                      << "\nDistance from anode to cathode, in cm " << spec->anodeCathodeSpacing()
                                      << "\nGas gain " << spec->gasGain() << "\nVoltage " << spec->voltage()
                                      << "\nCalibration uncertainty " << spec->calibrationError()
                                      << "\nElectron attraction " << spec->electronAttraction()
                                      << "\nFraction of the charge that survives to reach the cathode "
                                      << spec->fractionQS() << "\nADC calibration, in fC " << spec->chargePerCount()
                                      << "\nAnode wire radius, in cm " << spec->wireRadius()
                                      << "\nFast shaper peaking time (ns) " << spec->shaperPeakingTime()
                                      << "\nThe constant term in the electronics noise, in # of electrons "
                                      << spec->constantNoise()
                                      << "\nThe # of noise electrons per picofarad of capacitance " << spec->e_pF();

    for (unsigned int k2 = 0; k2 < layers.size(); ++k2) {
      auto const& id = layers[k2]->id();
      edm::LogVerbatim("CSCGeometry") << "\nLayer " << k2 << ":" << layers[k2]->type().name() << ": (E:" << id.endcap()
                                      << " S:" << id.station() << " R:" << id.ring() << " C:" << id.chamber()
                                      << " L:" << id.layer() << ")";

      if (verbose_)
        edm::LogVerbatim("CSCGeometry") << "\nLayer Geometry:\n" << *(layers[k2]->geometry());
    }
  }
}

DEFINE_FWK_MODULE(CSCGeometryDump);

/****************************************************************************
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionData.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

#include "Geometry/Records/interface/VeryForwardRealGeometryRecord.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h"

#include "CalibPPS/AlignmentRelative/interface/AlignmentTask.h"
#include "CalibPPS/AlignmentRelative/interface/Utilities.h"

/**
 *\brief Modifies the alignment modes unconstrained by the track-based alignment.
 **/
class PPSModifySingularModes : public edm::stream::EDAnalyzer<> {
public:
  PPSModifySingularModes(const edm::ParameterSet &ps);

private:
  edm::ParameterSet ps_;

  edm::ESGetToken<CTPPSGeometry, VeryForwardRealGeometryRecord> tokenRealGeometry_;

  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  void analyze(const edm::Event &e, const edm::EventSetup &es) override {}
};

using namespace std;
using namespace edm;

//----------------------------------------------------------------------------------------------------

PPSModifySingularModes::PPSModifySingularModes(const ParameterSet &ps)
    : ps_(ps), tokenRealGeometry_(esConsumes<edm::Transition::BeginRun>()) {}

//----------------------------------------------------------------------------------------------------

void PPSModifySingularModes::beginRun(edm::Run const &, edm::EventSetup const &es) {
  // get config parameters
  const double z1 = ps_.getUntrackedParameter<double>("z1");
  const double z2 = ps_.getUntrackedParameter<double>("z2");
  const double de_x1 = ps_.getUntrackedParameter<double>("de_x1");
  const double de_x2 = ps_.getUntrackedParameter<double>("de_x2");
  const double de_y1 = ps_.getUntrackedParameter<double>("de_y1");
  const double de_y2 = ps_.getUntrackedParameter<double>("de_y2");
  const double de_rho1 = ps_.getUntrackedParameter<double>("de_rho1");
  const double de_rho2 = ps_.getUntrackedParameter<double>("de_rho2");

  FileInPath inputFileInPath(ps_.getUntrackedParameter<string>("inputFile"));
  const string inputFile = inputFileInPath.fullPath();
  const string outputFile = ps_.getUntrackedParameter<string>("outputFile");

  // validate config parameters
  if (z1 == z2)
    throw cms::Exception("PPS") << "z1 equals z2";

  // calculate slopes and intercepts
  const double a_x = (de_x2 - de_x1) / (z2 - z1), b_x = de_x1 - a_x * z1;
  const double a_y = (de_y2 - de_y1) / (z2 - z1), b_y = de_y1 - a_y * z1;
  const double a_rho = (de_rho2 - de_rho1) / (z2 - z1), b_rho = de_rho1 - a_rho * z1;

  // get geometry
  const auto &geometry = es.getData(tokenRealGeometry_);

  // get input alignments
  CTPPSRPAlignmentCorrectionsDataSequence inputSequence = CTPPSRPAlignmentCorrectionsMethods::loadFromXML(inputFile);
  const auto &input = inputSequence.begin()->second;

  // modify the singular modes
  CTPPSRPAlignmentCorrectionsData output = input;

  for (auto &it : input.getSensorMap()) {
    const auto &sensorId = it.first;

    const auto &c = geometry.sensorTranslation(sensorId);

    // pixels cannot be described by one single value of z, but no better approxiamtion can be made
    const double z = c.z();

    double de_ShX = a_x * z + b_x;
    double de_ShY = a_y * z + b_y;
    const double de_RotZ = a_rho * z + b_rho;

    // add the effect of global rotation (about origin, not sensor centre)
    de_ShX -= +de_RotZ * (c.y() + de_ShY);
    de_ShY -= -de_RotZ * (c.x() + de_ShX);

    CTPPSRPAlignmentCorrectionData d = it.second;
    d.setShX(d.getShX() + de_ShX);
    d.setShY(d.getShY() + de_ShY);
    d.setRotZ(d.getRotZ() + de_RotZ);

    output.setSensorCorrection(sensorId, d);
  }

  // build list of RPs
  vector<unsigned int> rps;
  unsigned int last_rp = 123456;
  for (auto &it : input.getSensorMap()) {
    CTPPSDetId senId(it.first);
    unsigned int rpDecId = senId.arm() * 100 + senId.station() * 10 + senId.rp();

    if (last_rp != rpDecId) {
      rps.push_back(rpDecId);
      last_rp = rpDecId;
    }
  }

  // build alignment geometry (needed for the factorisation below)
  AlignmentGeometry alignmentGeometry;
  vector<unsigned int> excludePlanes;
  AlignmentTask::buildGeometry(rps, excludePlanes, &geometry, 0., alignmentGeometry);

  // factorise output
  CTPPSRPAlignmentCorrectionsData outputExpanded;
  CTPPSRPAlignmentCorrectionsData outputFactored;

  const bool equalWeights = false;
  factorRPFromSensorCorrections(output, outputExpanded, outputFactored, alignmentGeometry, equalWeights, 1);

  // save output
  CTPPSRPAlignmentCorrectionsMethods::writeToXML(outputFactored, outputFile, false, false, true, true, true, true);
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPSModifySingularModes);

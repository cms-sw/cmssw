// -*- C++ -*-
//
// Package:    Alignment/GREAtAlgorithm
// Class:      GREAtAlgorithm
//
/**\class GREAtAlgorithm GREAtAlgorithm.cc Alignment/GREAtAlgorithm/plugins/GREAtAlgorithm.cc

   Description: *G*regor's *R*andom-*E*nhanced *A*lignmen*t* Algorithm

   Implementation:

   This algorithm consistently introduces the notion of randomness into the
   determination of tracker alignment conditions.

   It simultaneously determines the parameters of the TrackerAlignmentRcd,
   TrackerAlignmentErrorExtendedRcd, TrackerSurfaceDeformationRcd,
   SiPixelLorentzAngleRcd, SiStripLorentzAngleRcd, and
   SiStripBackPlaneCorrectionRcd.

   The algorithm is GREAt because it does not even require data. Hence it is the
   natural choice when predicting the alignment prior to data taking.

   For the same reason it allows to disentangle the CMS and LHC schedule from
   the alignment expert's vacation plans because the alignment can be derived at
   any point in time.

   In addition to the above-mentioned convenience features it also shows
   unprecedented performance improvements:

   - It's extremely fast! All the above-mentioned records are derived almost
     instantly, thereby dramatically reducing the time budget of normally 1-2
     weeks.  
   - Due to its inherent GREAt-ness, validation is not required anymore. Thus
     the total time between the request of a new alignment and the delivery can
     be reduced to a few seconds.

   It also solves the typical problem of conventional algorithms that have to
   deal with complaints about delays caused by the missing tracker alignment
   conditions. The first complaints about delay typically arise several days,
   sometimes weeks, before all inputs of the conventional algorithms are
   available.  The extreme reduction of the required inputs for the
   GREAtAlgorithm fully avoids this problem in a straight forward fashion.

   NOTE: The implementation of this alignment algorithm is unique as it requires
         no data. Thus the overhead of the AligmentProducer framework is
         elegantly circumvented.

*/
//
// Original Author:  Gregor Mittag
//         Created:  Sun, 01 Apr 2018 00:00:00 GMT
//
//


// system include files

// user include files
#include "CLHEP/Random/DRand48Engine.h"
#include "CLHEP/Random/RandGauss.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/Alignment/interface/Alignments.h"
#include "CondFormats/Alignment/interface/AlignTransform.h"
#include "CondFormats/Alignment/interface/AlignmentErrorsExtended.h"
#include "CondFormats/Alignment/interface/AlignTransformError.h"
#include "CondFormats/Alignment/interface/AlignTransformErrorExtended.h"
#include "CondFormats/Alignment/interface/AlignmentSurfaceDeformations.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerAlignmentErrorExtendedRcd.h"
#include "CondFormats/AlignmentRecord/interface/TrackerSurfaceDeformationRcd.h"
#include "CondFormats/DataRecord/interface/SiPixelLorentzAngleRcd.h"
#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CondFormats/SiStripObjects/interface/SiStripBackPlaneCorrection.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
//
// class declaration
//


class GREAtAlgorithm : public edm::one::EDAnalyzer<>  {
public:
  explicit GREAtAlgorithm(const edm::ParameterSet&);
  ~GREAtAlgorithm() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void determineParameters(const edm::EventSetup&);
  void writeToDB();

  // ----------member data ---------------------------
  const double inverseLevelOfTrustworthiness_;
  const std::string pixelLorentzAngleLabel_;
  const std::string apvMode_;

  CLHEP::DRand48Engine randomEngine_;
  Alignments alignments_;
  AlignmentErrorsExtended alignmentErrors_;
  AlignmentSurfaceDeformations alignmentSurfaceDeformations_;
  SiPixelLorentzAngle siPixelLorentzAngles_;
  SiStripLorentzAngle siStripLorentzAngles_;
  SiStripBackPlaneCorrection siStripBackPlaneCorrections_;
  bool firstEvent_{true};
};

//
// constructor
//
GREAtAlgorithm::GREAtAlgorithm(const edm::ParameterSet& iConfig) :
  inverseLevelOfTrustworthiness_{iConfig.getParameter<double>("inverseLevelOfTrustworthiness")},
  pixelLorentzAngleLabel_{iConfig.getParameter<std::string>("pixelLorentzAngleLabel")},
  apvMode_{iConfig.getParameter<std::string>("apvMode")}
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
GREAtAlgorithm::analyze(const edm::Event&, const edm::EventSetup& iSetup)
{
  if (firstEvent_) {
    determineParameters(iSetup);
    writeToDB();
    firstEvent_ = false;
  }
}


void
GREAtAlgorithm::determineParameters(const edm::EventSetup& iSetup)
{
  edm::ESHandle<Alignments> alignments;
  edm::ESHandle<AlignmentErrorsExtended> alignmentErrors;
  edm::ESHandle<AlignmentSurfaceDeformations> surfaceDeformations;
  edm::ESHandle<SiPixelLorentzAngle> pixelLorentzAngles;
  edm::ESHandle<SiStripLorentzAngle> stripLorentzAngles;
  edm::ESHandle<SiStripBackPlaneCorrection> backPlaneCorrections;

  iSetup.get<TrackerAlignmentRcd>().get(alignments);
  iSetup.get<TrackerAlignmentErrorExtendedRcd>().get(alignmentErrors);
  iSetup.get<TrackerSurfaceDeformationRcd>().get(surfaceDeformations);
  iSetup.get<SiPixelLorentzAngleRcd>().get(pixelLorentzAngleLabel_, pixelLorentzAngles);
  iSetup.get<SiStripLorentzAngleRcd>().get(apvMode_, stripLorentzAngles);
  iSetup.get<SiStripBackPlaneCorrectionRcd>().get(apvMode_, backPlaneCorrections);

  CLHEP::RandGauss aliG{randomEngine_, 0.0, inverseLevelOfTrustworthiness_};

  for (const auto& item: alignments->m_align) {
    const AlignTransform::Translation trans{item.translation().x()* (1 + aliG.fire()),
        item.translation().y()* (1 + aliG.fire()),
        item.translation().z()* (1 + aliG.fire())};
    const AlignTransform::Rotation rot{item.rotation().phi()* (1 + aliG.fire()),
        item.rotation().theta()* (1 + aliG.fire()),
        item.rotation().psi()* (1 + aliG.fire())};
    alignments_.m_align.emplace_back(trans, rot, item.rawId());
  }

  for (const auto& item: alignmentErrors->m_alignError) {
    alignmentErrors_.m_alignError.emplace_back(item.matrix() * (1 + aliG.fire()),
                                               item.rawId());
  }

  for (size_t i = 0; i < surfaceDeformations->items().size(); ++i) {
    const auto beginEndPair = surfaceDeformations->parameters(i);
    std::vector<align::Scalar> params;
    params.reserve(std::distance(beginEndPair.first, beginEndPair.second));
    for (auto p = beginEndPair.first; p != beginEndPair.second; ++p) {
      params.emplace_back(*p* (1 + aliG.fire()));
    }
    const auto& item = surfaceDeformations->items()[i];
    alignmentSurfaceDeformations_.add(item.m_rawId,
                                      item.m_parametrizationType,
                                      params);
  }

  for (const auto& item: pixelLorentzAngles->getLorentzAngles()) {
    auto value = static_cast<float>(item.second* (1 + aliG.fire()));
    siPixelLorentzAngles_.putLorentzAngle(item.first, value);
  }

  for (const auto& item: stripLorentzAngles->getLorentzAngles()) {
    siStripLorentzAngles_.putLorentzAngle(item.first, item.second* (1 + aliG.fire()));
  }

  for (const auto& item: backPlaneCorrections->getBackPlaneCorrections()) {
    siStripBackPlaneCorrections_.putBackPlaneCorrection(item.first,
                                                        item.second* (1 + aliG.fire()));
  }
}


void
GREAtAlgorithm::writeToDB()
{
  const auto& since = cond::timeTypeSpecs[cond::runnumber].beginValue;

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (!poolDb.isAvailable()) {
    throw cms::Exception("NotAvailable") << "PoolDBOutputService not available";
  }

  edm::LogInfo("Alignment")
    << "Writing tracker-alignment records.";
  poolDb->writeOne(&alignments_, since, "TrackerAlignmentRcd");
  poolDb->writeOne(&alignmentErrors_, since, "TrackerAlignmentErrorExtendedRcd");
  poolDb->writeOne(&alignmentSurfaceDeformations_, since, "TrackerSurfaceDeformationRcd");
  poolDb->writeOne(&siPixelLorentzAngles_, since, "SiPixelLorentzAngleRcd");
  poolDb->writeOne(&siStripLorentzAngles_, since, "SiStripLorentzAngleRcd");
  poolDb->writeOne(&siStripBackPlaneCorrections_, since, "SiStripBackPlaneCorrectionRcd");
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
GREAtAlgorithm::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Produces all possible tracker-alignment-related "
                  "database payloads you can imagine. "
                  "PoolDBOutputService must be set up for the corresponding records.");
  desc.add<double>("inverseLevelOfTrustworthiness", 0.0); // By default we infinitely trust this algorithm!
  desc.add<std::string>("pixelLorentzAngleLabel", "fromAlignment");
  desc.add<std::string>("apvMode", "deconvolution");
  descriptions.add("greatAlgorithm", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GREAtAlgorithm);

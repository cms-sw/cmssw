// -*- C++ -*-
//
// Package:    Alignment/CommonAlignment
// Class:      MagneticFieldFilter
//
/**\class MagneticFieldFilter MagneticFieldFilter.cc Alignment/CommonAlignment/plugins/MagneticFieldFilter.cc

 Description: Plugin to filter events based on the magnetic field value

 Implementation:
     Takes the magnet current from the RunInfoRcd and translates it into a
     magnetic field value using the parameterization given here:

     https://hypernews.cern.ch/HyperNews/CMS/get/magnetic-field/63/1/1/1.html

     Agrees within an accuracy of ~20 mT with results of:
     https://cmswbm.web.cern.ch/cmswbm/cmsdb/servlet/RunSummary

*/
//
// Original Author:  Gregor Mittag
//         Created:  Wed, 25 Nov 2015 12:59:02 GMT
//
//


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

//
// class declaration
//

class MagneticFieldFilter : public edm::stream::EDFilter<> {
public:
  explicit MagneticFieldFilter(const edm::ParameterSet&);
  ~MagneticFieldFilter() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  virtual bool filter(edm::Event&, const edm::EventSetup&) override;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /// convert Ampere (A) to Tesla (T)
  float currentToField(const float& current) const;

  // ----------member data ---------------------------
  /// see: https://hypernews.cern.ch/HyperNews/CMS/get/magnetic-field/63/1/1/1.html
  static constexpr float linearCoeffCurrentToField_ = 2.084287e-04;
  /// see: https://hypernews.cern.ch/HyperNews/CMS/get/magnetic-field/63/1/1/1.html
  static constexpr float constantTermCurrentToField_ = 1.704418e-02;

  const int magneticField_;     /// magnetic field that is filtered
  int magneticFieldCurrentRun_; /// magnetic field estimate of the current run
};

//
// static data member definitions
//
constexpr float MagneticFieldFilter::linearCoeffCurrentToField_;
constexpr float MagneticFieldFilter::constantTermCurrentToField_;


//
// constructor
//
MagneticFieldFilter::MagneticFieldFilter(const edm::ParameterSet& iConfig) :
  magneticField_(iConfig.getUntrackedParameter<int>("magneticField")),
  magneticFieldCurrentRun_(-10000) {
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
MagneticFieldFilter::filter(edm::Event&, const edm::EventSetup&) {
  return magneticField_ == magneticFieldCurrentRun_;
}

// ------------ method called when starting to processes a run  ------------

void
MagneticFieldFilter::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  edm::ESHandle<RunInfo> sum;
  iSetup.get<RunInfoRcd>().get(sum);
  auto summary = sum.product();
  // convert from Tesla to kGauss (multiply with 10) and
  // round off to whole kGauss (add 0.5 and cast to int) as is done in
  // 'MagneticField::computeNominalValue()':
  magneticFieldCurrentRun_ =
    static_cast<int>(currentToField(summary->m_avg_current)*10.0 + 0.5);
}


float
MagneticFieldFilter::currentToField(const float& current) const {
  return linearCoeffCurrentToField_ * current + constantTermCurrentToField_;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
MagneticFieldFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Filters events with a magnetic field of 'magneticField'.");
  desc.addUntracked<int>("magneticField", 38)
    ->setComment("In units of kGauss (= 0.1 Tesla).");
  descriptions.add("magneticFieldFilter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MagneticFieldFilter);

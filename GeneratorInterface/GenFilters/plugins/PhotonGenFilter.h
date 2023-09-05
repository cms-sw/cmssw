#ifndef PhotonGenFilter_h
#define PhotonGenFilter_h

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


namespace edm {
  class HepMCProduct;
}

class PhotonGenFilter : public edm::global::EDFilter<> {
public:
    explicit PhotonGenFilter(const edm::ParameterSet&);
    ~PhotonGenFilter() override;
    
    bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
    // Private member variables and functions

  const edm::EDGetTokenT<edm::HepMCProduct> token_;
  std::vector<int> particleID;
  std::vector<double> ptMin;
  std::vector<double> etaMin;
  std::vector<double> etaMax;
  std::vector<double> drMin;
  std::vector<int> status;
  const double betaBoost;

};

#endif
#include "CommonTools/RecoAlgos/interface/MassiveCandidateConverter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HepPDT/ParticleData.hh"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include <algorithm>
using namespace edm;
using namespace std;
using namespace converter;

MassiveCandidateConverter::MassiveCandidateConverter(const edm::ParameterSet& cfg, edm::ConsumesCollector iC)
    : massSqr_(0),
      particle_(cfg.getParameter<PdtEntry>("particleType")),
      tableToken_(iC.esConsumes<edm::Transition::BeginRun>()) {}

void MassiveCandidateConverter::beginFirstRun(const EventSetup& es) {
  auto const& pdt = es.getData(tableToken_);
  particle_.setup(pdt);
  massSqr_ = particle_.data().mass();
  massSqr_ *= massSqr_;
}

void MassiveCandidateConverter::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.addNode(edm::ParameterDescription<int>("particleType", true) xor
               edm::ParameterDescription<std::string>("particleType", true))
      ->setComment("the PdtEntry can be specified as either an 'int' or via its name using a 'string'");
}

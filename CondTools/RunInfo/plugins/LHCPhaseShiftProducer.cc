#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <CLHEP/Units/SystemOfUnits.h>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

class LHCPhaseShiftProducer : public edm::global::EDProducer<> {
public:
  explicit LHCPhaseShiftProducer(const edm::ParameterSet& iConfig)
      : collisionPhaseShift_(computePhaseShift(iConfig)), collidingBunches_(getCollidingBunches()) {
    produces<std::vector<float>>();
  };
  ~LHCPhaseShiftProducer() override {};

  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  int getBunchCrossing(const edm::Event& iEvent) const {
    if (iEvent.isRealData())
      return iEvent.bunchCrossing();
    const auto idx = iEvent.id().event() % collidingBunches_.size();
    return collidingBunches_[idx];
  };

  float getPhaseShift(int bx) const { return collisionPhaseShift_[bx].first; };

private:
  static constexpr size_t MAXBX = 3564;
  static constexpr double RF = 400800000;  //RF radio frequency [Hz]
  static constexpr double RoverQ = 45;     //R/Q of the cavity [Ohm]
  static constexpr double RFxPI = RF * M_PI;
  typedef std::array<std::pair<double, bool>, MAXBX> INFOTYPE;

  const INFOTYPE collisionPhaseShift_;
  const std::vector<int> collidingBunches_;

  void createBeams(std::vector<bool>& beam1, std::vector<bool>& beam2, const edm::ParameterSet& iConfig) {
    std::ifstream lhcfile(iConfig.getParameter<edm::FileInPath>("file").fullPath());
    const nlohmann::ordered_json lhc_data = nlohmann::ordered_json::parse(lhcfile, nullptr, true, true);
    for (const auto& beam : lhc_data["beam1"])
      for (const auto& batch : beam["batches"])
        for (size_t i = 0; i < batch["bunches"]; i++)
          beam1[int(beam["lhcbunch"]) + int(batch["injBunch"]) + i] = true;
    for (const auto& beam : lhc_data["beam2"])
      for (const auto& batch : beam["batches"])
        for (size_t i = 0; i < batch["bunches"]; i++)
          beam2[int(beam["lhcbunch"]) + int(batch["injBunch"]) + i] = true;
  };

  std::vector<double> calculatePhase(const std::vector<bool>& beams, double b_len, double ppb, double V0) {
    int nbunch(0);
    for (const auto& beam : beams)
      nbunch += beam;
    const auto t1 = RFxPI * b_len;
    const auto fbp = 3 * (-t1 * std::cos(t1) + std::sin(t1)) / (t1 * t1 * t1);
    const auto ibt = 2 * fbp * ppb * CLHEP::e_SI / 25.0e-9;
    const auto fbx = nbunch / double(MAXBX);
    const auto domega = -RFxPI * RoverQ * ibt * fbx / V0;
    std::vector<double> timevec(MAXBX);
    double ssum(0.), integ(0.);
    for (size_t i = 0; i < MAXBX; i++) {
      integ += domega * 25.0e-9 * (1. - beams[i] / fbx);
      const auto time = -0.5 * integ * 1.0e9 / RFxPI;  // [ns]
      timevec[i] = time;
      ssum += time;
    }
    const auto mean = ssum / MAXBX;
    for (auto& time : timevec)
      time -= mean;
    return timevec;
  };

  INFOTYPE computePhaseShift(const edm::ParameterSet& iConfig) {
    std::vector<bool> beam1(MAXBX, false), beam2(MAXBX, false);
    createBeams(beam1, beam2, iConfig);
    const auto& V0 = iConfig.getParameter<double>("V0");
    const auto& ppb = iConfig.getParameter<double>("ppb");
    const auto& b_len = iConfig.getParameter<double>("b_len");
    const auto phaseshift1 = calculatePhase(beam1, b_len, ppb, V0);
    const auto phaseshift2 = calculatePhase(beam2, b_len, ppb, V0);
    INFOTYPE collisionTimePhase = {};
    for (size_t i = 0; i < MAXBX; i++) {
      collisionTimePhase[i].first = (phaseshift1[i] + phaseshift2[i]) / 2.;
      collisionTimePhase[i].second = beam1[i] && beam2[i];
    }
    return collisionTimePhase;
  };

  std::vector<int> getCollidingBunches() {
    std::vector<int> collBX;
    collBX.reserve(MAXBX);
    for (size_t i = 0; i < MAXBX; i++)
      if (collisionPhaseShift_[i].second)
        collBX.push_back(i);
    return collBX;
  };
};

void LHCPhaseShiftProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto bx = getBunchCrossing(iEvent);
  const auto ps = getPhaseShift(bx);
  auto output = std::make_unique<std::vector<float>>(std::vector<float>({float(bx), ps}));
  iEvent.put(std::move(output));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LHCPhaseShiftProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("file")->setComment("LHC schema json file");
  desc.add<double>("V0", 1500000)->setComment("Voltage of one cavity [V]");
  desc.add<double>("ppb", 1.15e11)->setComment("Protons per bunch");
  desc.add<double>("b_len", 1e-9)->setComment("Bunch length in seconds (4 sigma) [sec]");
  descriptions.add("lhcPhaseShift", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(LHCPhaseShiftProducer);

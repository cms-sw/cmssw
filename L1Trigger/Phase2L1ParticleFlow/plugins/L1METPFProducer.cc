#include <vector>
#include <ap_int.h>
#include <ap_fixed.h>
#include <TVector2.h>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace l1t;

class L1METPFProducer : public edm::global::EDProducer<> {
public:
  explicit L1METPFProducer(const edm::ParameterSet&);
  ~L1METPFProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  edm::EDGetTokenT<vector<l1t::PFCandidate>> _l1PFToken;

  int maxCands_ = 128;

  // quantization controllers
  typedef ap_ufixed<14, 12, AP_RND, AP_WRAP> pt_t;  // LSB is 0.25 and max is 4 TeV
  typedef ap_int<12> phi_t;                         // LSB is pi/720 ~ 0.0044 and max is +/-8.9
  const float ptLSB_ = 0.25;                        // GeV
  const float phiLSB_ = M_PI / 720;                 // rad

  // derived, helper types
  typedef ap_fixed<pt_t::width + 1, pt_t::iwidth + 1, AP_RND, AP_SAT> pxy_t;
  typedef ap_fixed<2 * pt_t::width, 2 * pt_t::iwidth, AP_RND, AP_SAT> pt2_t;
  // derived, helper constants
  const float maxPt_ = ((1 << pt_t::width) - 1) * ptLSB_;
  const phi_t hwPi_ = round(M_PI / phiLSB_);
  const phi_t hwPiOverTwo_ = round(M_PI / (2 * phiLSB_));

  typedef ap_ufixed<pt_t::width, 0> inv_t;  // can't easily use the MAXPT/pt trick with ap_fixed

  // to make configurable...
  const int dropBits_ = 2;
  const int dropFactor_ = (1 << dropBits_);
  const int invTableBits_ = 10;
  const int invTableSize_ = (1 << invTableBits_);

  void Project(pt_t pt, phi_t phi, pxy_t& pxy, bool isX, bool debug = false) const;
  void PhiFromXY(pxy_t px, pxy_t py, phi_t& phi, bool debug = false) const;

  void CalcMetHLS(const std::vector<float>& pt,
                  const std::vector<float>& phi,
                  reco::Candidate::PolarLorentzVector& metVector) const;
};

L1METPFProducer::L1METPFProducer(const edm::ParameterSet& cfg)
    : _l1PFToken(consumes<std::vector<l1t::PFCandidate>>(cfg.getParameter<edm::InputTag>("L1PFObjects"))),
      maxCands_(cfg.getParameter<int>("maxCands")) {
  produces<std::vector<l1t::EtSum>>();
}

void L1METPFProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(_l1PFToken, l1PFCandidates);

  std::vector<float> pt;
  std::vector<float> phi;

  for (int i = 0; i < int(l1PFCandidates->size()) && (i < maxCands_ || maxCands_ < 0); i++) {
    const auto& l1PFCand = l1PFCandidates->at(i);
    pt.push_back(l1PFCand.pt());
    phi.push_back(l1PFCand.phi());
  }

  reco::Candidate::PolarLorentzVector metVector;

  CalcMetHLS(pt, phi, metVector);

  l1t::EtSum theMET(metVector, l1t::EtSum::EtSumType::kTotalHt, 0, 0, 0, 0);

  std::unique_ptr<std::vector<l1t::EtSum>> metCollection(new std::vector<l1t::EtSum>(0));
  metCollection->push_back(theMET);
  iEvent.put(std::move(metCollection));
}

void L1METPFProducer::CalcMetHLS(const std::vector<float>& pt,
                                 const std::vector<float>& phi,
                                 reco::Candidate::PolarLorentzVector& metVector) const {
  pxy_t hw_px = 0;
  pxy_t hw_py = 0;
  pxy_t hw_sumx = 0;
  pxy_t hw_sumy = 0;

  for (uint i = 0; i < pt.size(); i++) {
    pt_t hw_pt = min(pt[i], maxPt_);
    phi_t hw_phi = float(TVector2::Phi_mpi_pi(phi[i]) / phiLSB_);

    Project(hw_pt, hw_phi, hw_px, true);
    Project(hw_pt, hw_phi, hw_py, false);

    hw_sumx = hw_sumx - hw_px;
    hw_sumy = hw_sumy - hw_py;
  }

  pt2_t hw_met = pt2_t(hw_sumx) * pt2_t(hw_sumx) + pt2_t(hw_sumy) * pt2_t(hw_sumy);
  hw_met = sqrt(int(hw_met));  // stand-in for HLS::sqrt

  phi_t hw_met_phi = 0;
  PhiFromXY(hw_sumx, hw_sumy, hw_met_phi);

  metVector.SetPt(hw_met.to_double());
  metVector.SetPhi(hw_met_phi.to_double() * phiLSB_);
  metVector.SetEta(0);
}

void L1METPFProducer::Project(pt_t pt, phi_t phi, pxy_t& pxy, bool isX, bool debug) const {
  /*
      Convert pt and phi to px (py)
      1) Map phi to the first quadrant to reduce LUT size
      2) Lookup sin(phiQ1), where the result is in [0,maxPt]
      which is used to encode [0,1].
      3) Multiply pt by sin(phiQ1) to get px. Result will be px*maxPt, but
      wrapping multiplication is 'mod maxPt' so the correct value is returned.
      4) Check px=-|px|.
    */

  // set phi to first quadrant
  phi_t phiQ1 = (phi > 0) ? phi : phi_t(-phi);  // Q1/Q4
  if (phiQ1 >= hwPiOverTwo_)
    phiQ1 = hwPi_ - phiQ1;

  if (phiQ1 > hwPiOverTwo_) {
    edm::LogWarning("L1METPFProducer") << "unexpected phi (high)";
    phiQ1 = hwPiOverTwo_;
  } else if (phiQ1 < 0) {
    edm::LogWarning("L1METPFProducer") << "unexpected phi (low)";
    phiQ1 = 0;
  }
  if (isX) {
    typedef ap_ufixed<14, 12, AP_RND, AP_WRAP> pt_t;  // LSB is 0.25 and max is 4 TeV
    ap_ufixed<pt_t::width, 0> cosPhi = cos(phiQ1.to_double() / hwPiOverTwo_.to_double() * M_PI / 2);
    pxy = pt * cosPhi;
    if (phi > hwPiOverTwo_ || phi < -hwPiOverTwo_)
      pxy = -pxy;
  } else {
    ap_ufixed<pt_t::width, 0> sinPhi = sin(phiQ1.to_double() / hwPiOverTwo_.to_double() * M_PI / 2);
    pxy = pt * sinPhi;
    if (phi < 0)
      pxy = -pxy;
  }
}

void L1METPFProducer::PhiFromXY(pxy_t px, pxy_t py, phi_t& phi, bool debug) const {
  if (px == 0 && py == 0) {
    phi = 0;
    return;
  }
  if (px == 0) {
    phi = py > 0 ? hwPiOverTwo_ : phi_t(-hwPiOverTwo_);
    return;
  }
  if (py == 0) {
    phi = px > 0 ? phi_t(0) : phi_t(-hwPi_);
    return;
  }

  // get q1 coordinates
  pt_t x = px > 0 ? pt_t(px) : pt_t(-px);  //px>=0 ? px : -px;
  pt_t y = py > 0 ? pt_t(py) : pt_t(-py);  //px>=0 ? px : -px;
  // transform so a<b
  pt_t a = x < y ? x : y;
  pt_t b = x < y ? y : x;

  if (b.to_double() > maxPt_ / dropFactor_)
    b = maxPt_ / dropFactor_;
  // map [0,max/4) to inv table size
  int index = round((b.to_double() / (maxPt_ / dropFactor_)) * invTableSize_);
  float bcheck = (float(index) / invTableSize_) * (maxPt_ / dropFactor_);
  inv_t inv_b = 1. / ((float(index) / invTableSize_) * (maxPt_ / dropFactor_));

  inv_t a_over_b = a * inv_b;

  if (debug) {
    LogDebug("L1METPFProducer") << "  a, b = \n  " << a.to_double() << " , " << b.to_double()
                                << ";   index, inv = " << index << ", " << inv_b.to_double()
                                << "; ratio= " << a_over_b.to_double() << " \n"
                                << std::endl;
    LogDebug("L1METPFProducer") << "bcheck, 1/bc = " << bcheck << ", " << 1. / bcheck << " -- " << invTableSize_ << " "
                                << maxPt_ << " " << dropFactor_ << " \n"
                                << std::endl;
  }

  const int atanTableBits_ = 7;
  const int atanTableSize_ = (1 << atanTableBits_);
  index = round(a_over_b.to_double() * atanTableSize_);
  phi = atan(float(index) / atanTableSize_) / phiLSB_;

  if (debug) {
    LogDebug("L1METPFProducer") << "    atan index, phi = " << index << ", " << phi.to_double() << " ("
                                << phi.to_double() * (M_PI / hwPi_.to_double())
                                << " rad) real atan(a/b)= " << atan(a.to_double() / b.to_double()) << " \n"
                                << std::endl;
  }

  // rotate from (0,pi/4) to full quad1
  if (y > x)
    phi = hwPiOverTwo_ - phi;  //phi = pi/2 - phi
  // other quadrants
  if (px < 0 && py > 0)
    phi = hwPi_ - phi;  // Q2 phi = pi - phi
  if (px > 0 && py < 0)
    phi = -phi;  // Q4 phi = -phi
  if (px < 0 && py < 0)
    phi = -(hwPi_ - phi);  // Q3 composition of both

  if (debug) {
    LogDebug("L1METPFProducer") << "    phi hw, float, real = " << phi.to_double() << ", "
                                << phi.to_double() * (M_PI / hwPi_.to_double()) << "     ("
                                << atan2(py.to_double(), px.to_double()) << " rad from x,y = " << px.to_double() << ", "
                                << py.to_double() << ") \n"
                                << std::endl;
  }
}

void L1METPFProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("maxCandidates", 128);
  desc.add<edm::InputTag>("L1PFObjects", edm::InputTag("L1PFProducer", "l1pfCandidates"));
  descriptions.add("L1METPFProducer", desc);
}

L1METPFProducer::~L1METPFProducer() {}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1METPFProducer);

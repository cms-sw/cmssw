// This class header:
#include "FastSimDataFormats/L1GlobalMuonTrigger/interface/SimpleL1MuGMTCand.h"

// Fast Simulation headers
#include "SimDataFormats/Track/interface/SimTrack.h"

// The muon scales

//CMSSW headers

// STL headers
#include <iomanip>

SimpleL1MuGMTCand::SimpleL1MuGMTCand()
    : m_name("FastL1MuCand"),
      m_empty(true),
      m_phi(0),
      m_eta(31),
      m_pt(0),
      m_charge(0),
      m_quality(0),
      m_rank(0),
      m_smearedPt(0) {}

SimpleL1MuGMTCand::SimpleL1MuGMTCand(const SimpleL1MuGMTCand& mu)
    : L1MuGMTExtendedCand::L1MuGMTExtendedCand(mu),
      m_name(mu.m_name),
      m_empty(mu.m_empty),
      m_phi(mu.m_phi),
      m_eta(mu.m_eta),
      m_pt(mu.m_pt),
      m_charge(mu.m_charge),
      m_quality(mu.m_quality),
      m_rank(mu.m_rank),
      m_smearedPt(mu.m_smearedPt) {
  setMomentum(mu.getMomentum());
  setQuality(m_quality & 7);
  setEtaPacked(m_eta & 63);
  setPhiPacked(m_phi & 255);
  setCharge(m_charge);
  setPtPacked(m_pt & 31);
}

SimpleL1MuGMTCand::SimpleL1MuGMTCand(const SimpleL1MuGMTCand* mu)
    : m_name(mu->m_name),
      m_empty(mu->m_empty),
      m_phi(mu->m_phi),
      m_eta(mu->m_eta),
      m_pt(mu->m_pt),
      m_charge(mu->m_charge),
      m_quality(mu->m_quality),
      m_rank(mu->m_rank),
      m_smearedPt(mu->m_smearedPt) {
  setMomentum(mu->getMomentum());
  setQuality(m_quality & 7);
  setEtaPacked(m_eta & 63);
  setPhiPacked(m_phi & 255);
  setCharge(m_charge);
  setPtPacked(m_pt & 31);
}

SimpleL1MuGMTCand::SimpleL1MuGMTCand(const SimTrack* p) {
  //  setMomentum(p->momentum());
  LorentzVector toBeRemoved(p->momentum().x(), p->momentum().y(), p->momentum().z(), p->momentum().t());
  setMomentum(toBeRemoved);
  m_name = "FastL1MuCand";
  m_empty = false;
  m_quality = 7;
  setQuality(m_quality);
  m_rank = 0;
  setEta(myMomentum.Eta());
  setPhi(myMomentum.Phi());
  setCharge(int(p->charge()));
  setPt(myMomentum.Pt());
  setBx(0);
  if (fabs(myMomentum.eta()) > 1.04)
    setFwdBit(1);
  else
    setFwdBit(0);
  setRPCBit(0);
}

SimpleL1MuGMTCand::SimpleL1MuGMTCand(const SimTrack* p,
                                     unsigned etaIndex,
                                     unsigned phiIndex,
                                     unsigned pTIndex,
                                     float etaValue,
                                     float phiValue,
                                     float pTValue) {
  //  setMomentum(p->momentum());
  LorentzVector toBeRemoved(p->momentum().x(), p->momentum().y(), p->momentum().z(), p->momentum().t());
  setMomentum(toBeRemoved);
  m_name = "FastL1MuCand";
  m_empty = false;
  m_quality = 7;
  setQuality(m_quality);
  m_rank = 0;
  m_phi = phiIndex;
  setPhiPacked(phiIndex);
  setPhiValue(phiValue);
  m_eta = etaIndex;
  setEtaPacked(etaIndex);
  setEtaValue(etaValue);
  setCharge(int(p->charge()));
  m_pt = pTIndex;
  setPtPacked(pTIndex);
  setPtValue(pTValue);
  m_smearedPt = myMomentum.Pt();
  setBx(0);
  if (fabs(etaValue) > 1.04)
    setFwdBit(1);
  else
    setFwdBit(0);
  setRPCBit(0);
}

SimpleL1MuGMTCand::~SimpleL1MuGMTCand() { reset(); }

//
void SimpleL1MuGMTCand::reset() {
  m_empty = true;
  m_phi = 0;
  m_eta = 31;
  m_pt = 0;
  m_charge = 0;
  m_quality = 0;
  m_rank = 0;
  m_smearedPt = 0;
}

//
// set phi-value of muon candidate
//
void SimpleL1MuGMTCand::setPhi(float phi) {
  int index = 0;
  float mindiff = 1000.0;

  if (phi < 0.) {
    phi = 2 * M_PI + phi;
  }
  for (int i = 0; i < 144; i++) {
    float diff = fabs(SimpleL1MuGMTCand::phiScale[i] - phi);
    if (diff <= mindiff) {
      mindiff = diff;
      index = i;
    }
  }

  m_phi = index;
  setPhiPacked(m_phi & 255);
  setPhiValue(phiScale[m_phi]);
}

//
// set eta-value of muon candidate
//
void SimpleL1MuGMTCand::setEta(float eta) {
  int index = 0;
  float mindiff = 1000.0;

  for (int i = 0; i < 63; i++) {
    float diff = fabs(SimpleL1MuGMTCand::etaScale[i] - eta);
    if (diff <= mindiff) {
      mindiff = diff;
      index = i;
    }
  }

  m_eta = index;
  setEtaPacked(m_eta & 63);
  setEtaValue(etaScale[m_eta]);
}

//
// set pt (value!!) of muon candidate
//
void SimpleL1MuGMTCand::setPt(float pt) {
  int index = 0;
  m_smearedPt = pt;

  float mindiff = 1000.0;

  for (int i = 0; i < 32; i++) {
    float diff = fabs(SimpleL1MuGMTCand::ptScale[i] - pt);
    if (diff <= mindiff) {
      mindiff = diff;
      index = i;
    }
  }

  m_pt = index;
  setPtPacked(m_pt & 31);
  setPtValue(ptScale[m_pt]);
}

//
// set charge and packed code of muon candidate
//
void SimpleL1MuGMTCand::setCharge(int charge) {
  m_charge = charge;
  setChargePacked(charge == 1 ? 0 : 1);
}

//
// set generator particle of muon candidate
//
/*
void SimpleL1MuGMTCand::setGenPart(const HepMC::GenParticle * rhp) {

   myGenParticle = rhp;
}
*/

//
// Assignment operator
//
SimpleL1MuGMTCand& SimpleL1MuGMTCand::operator=(const SimpleL1MuGMTCand& cand) {
  if (this != &cand) {
    m_empty = cand.m_empty;
    m_phi = cand.m_phi;
    m_eta = cand.m_eta;
    m_pt = cand.m_pt;
    m_charge = cand.m_charge;
    m_quality = cand.m_quality;
    m_rank = cand.m_rank;
    m_smearedPt = cand.m_smearedPt;
  }
  return *this;
}

//
// Assignment operator for SimTrack's
//
SimpleL1MuGMTCand* SimpleL1MuGMTCand::operator=(const SimTrack* p) {
  m_empty = false;
  setEta(p->momentum().eta());
  setPhi(p->momentum().phi());
  setCharge(int(p->charge()));
  setPt(std::sqrt(p->momentum().perp2()));

  return this;
}

//
// Equal operator
//
bool SimpleL1MuGMTCand::operator==(const SimpleL1MuGMTCand& cand) const {
  if (m_empty != cand.m_empty)
    return false;
  if (m_phi != cand.m_phi)
    return false;
  if (m_eta != cand.m_eta)
    return false;
  if (m_pt != cand.m_pt)
    return false;
  if (m_charge != cand.m_charge)
    return false;
  if (m_quality != cand.m_quality)
    return false;
  if (m_rank != cand.m_rank)
    return false;
  return true;
}

//
// Unequal operator
//
bool SimpleL1MuGMTCand::operator!=(const SimpleL1MuGMTCand& cand) const {
  if (m_empty != cand.m_empty)
    return true;
  if (m_phi != cand.m_phi)
    return true;
  if (m_eta != cand.m_eta)
    return true;
  if (m_pt != cand.m_pt)
    return true;
  if (m_charge != cand.m_charge)
    return true;
  if (m_quality != cand.m_quality)
    return true;
  if (m_rank != cand.m_rank)
    return true;
  return false;
}

//
// print parameters of track candidate
//
void SimpleL1MuGMTCand::print() const {
  using namespace std;

  if (!empty()) {
    cout.setf(ios::showpoint);
    cout.setf(ios::right, ios::adjustfield);
    cout << setiosflags(ios::showpoint | ios::fixed) << "pt = " << setw(5) << setprecision(1) << ptValue() << " GeV  "
         << "charge = " << setw(2) << charge() << " "
         << "eta = " << setw(5) << setprecision(2) << etaValue() << "  "
         << "phi = " << setw(5) << setprecision(3) << phiValue() << " rad  "
         << "rank = " << setw(6) << rank() << endl;
  }
}

//
// output stream operator for track candidate
//
std::ostream& operator<<(std::ostream& s, const SimpleL1MuGMTCand& id) {
  using namespace std;

  if (!id.empty()) {
    s << setiosflags(ios::showpoint | ios::fixed) << "pt = " << setw(5) << setprecision(1) << id.ptValue() << " GeV  "
      << "charge = " << setw(2) << id.charge() << " "
      << "eta = " << setw(5) << setprecision(2) << id.etaValue() << "  "
      << "phi = " << setw(5) << setprecision(3) << id.phiValue() << " rad  ";
  }
  return s;
}

//static

// pt scale in GeV
// low edges of pt bins
const float SimpleL1MuGMTCand::ptScale[32] = {0.0,  0.0,  1.5,  2.0,  2.5,  3.0,  3.5,  4.0,   4.5,   5.0,  6.0,
                                              7.0,  8.0,  10.0, 12.0, 14.0, 16.0, 18.0, 20.0,  25.0,  30.0, 35.0,
                                              40.0, 45.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0, 140.0};

// eta scale
const float SimpleL1MuGMTCand::etaScale[63] = {
    -2.40, -2.35, -2.30, -2.25, -2.20, -2.15, -2.10, -2.05, -2.00, -1.95, -1.90, -1.85, -1.80, -1.75, -1.70, -1.60,
    -1.50, -1.40, -1.30, -1.20, -1.10, -1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00,
    0.10,  0.20,  0.30,  0.40,  0.50,  0.60,  0.70,  0.80,  0.90,  1.00,  1.10,  1.20,  1.30,  1.40,  1.50,  1.60,
    1.70,  1.75,  1.80,  1.85,  1.90,  1.95,  2.00,  2.05,  2.10,  2.15,  2.20,  2.25,  2.30,  2.35,  2.40};

// phi scale
const float SimpleL1MuGMTCand::phiScale[144] = {
    0.0000, 0.0436, 0.0873, 0.1309, 0.1745, 0.2182, 0.2618, 0.3054, 0.3491, 0.3927, 0.4363, 0.4800, 0.5236, 0.5672,
    0.6109, 0.6545, 0.6981, 0.7418, 0.7854, 0.8290, 0.8727, 0.9163, 0.9599, 1.0036, 1.0472, 1.0908, 1.1345, 1.1781,
    1.2217, 1.2654, 1.3090, 1.3526, 1.3963, 1.4399, 1.4835, 1.5272, 1.5708, 1.6144, 1.6581, 1.7017, 1.7453, 1.7890,
    1.8326, 1.8762, 1.9199, 1.9635, 2.0071, 2.0508, 2.0944, 2.1380, 2.1817, 2.2253, 2.2689, 2.3126, 2.3562, 2.3998,
    2.4435, 2.4871, 2.5307, 2.5744, 2.6180, 2.6616, 2.7053, 2.7489, 2.7925, 2.8362, 2.8798, 2.9234, 2.9671, 3.0107,
    3.0543, 3.0980, 3.1416, 3.1852, 3.2289, 3.2725, 3.3161, 3.3598, 3.4034, 3.4470, 3.4907, 3.5343, 3.5779, 3.6216,
    3.6652, 3.7088, 3.7525, 3.7961, 3.8397, 3.8834, 3.9270, 3.9706, 4.0143, 4.0579, 4.1015, 4.1452, 4.1888, 4.2324,
    4.2761, 4.3197, 4.3633, 4.4070, 4.4506, 4.4942, 4.5379, 4.5815, 4.6251, 4.6688, 4.7124, 4.7560, 4.7997, 4.8433,
    4.8869, 4.9306, 4.9742, 5.0178, 5.0615, 5.1051, 5.1487, 5.1924, 5.2360, 5.2796, 5.3233, 5.3669, 5.4105, 5.4542,
    5.4978, 5.5414, 5.5851, 5.6287, 5.6723, 5.7160, 5.7596, 5.8032, 5.8469, 5.8905, 5.9341, 5.9778, 6.0214, 6.0650,
    6.1087, 6.1523, 6.1959, 6.2396};

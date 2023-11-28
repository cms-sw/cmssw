
/**\class L1TrackerHTMissEmulatorProducer L1TrackerHTMissEmulatorProducer.cc
 L1Trigger/L1TTrackMatch/plugins/L1TrackerHTMissEmulatorProducer.cc
 Description: Takes L1TTkJets and performs a integer emulation of Track-based missing HT, outputting a collection of EtSum 
*/

// Original Author:  Hardik Routray
//         Created:  Mon, 11 Oct 2021

// system include files
#include <memory>
#include <numeric>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMiss.h"
#include "DataFormats/L1TCorrelator/interface/TkHTMissFwd.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/TkJetWord.h"
#include "L1Trigger/L1TTrackMatch/interface/L1TkHTMissEmulatorProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace l1t;

class L1TkHTMissEmulatorProducer : public edm::stream::EDProducer<> {
public:
  explicit L1TkHTMissEmulatorProducer(const edm::ParameterSet&);
  ~L1TkHTMissEmulatorProducer() override = default;

private:
  virtual void beginJob();
  void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob();

  // ----------member data ---------------------------

  bool debug_ = false;
  bool displaced_;

  int cosLUTbins;
  int aSteps = 8;

  l1tmhtemu::pt_t jetMinPt_;
  l1tmhtemu::eta_t jetMaxEta_;
  l1tmhtemu::ntracks_t minNtracksHighPt_;
  l1tmhtemu::ntracks_t minNtracksLowPt_;

  std::vector<l1tmhtemu::phi_t> cosLUT_;
  std::vector<l1tmhtemu::MHTphi_t> atanLUT_;
  std::vector<l1tmhtemu::Et_t> magNormalisationLUT_;

  std::string L1MHTCollectionName_;

  const edm::EDGetTokenT<TkJetWordCollection> jetToken_;
};

L1TkHTMissEmulatorProducer::L1TkHTMissEmulatorProducer(const edm::ParameterSet& iConfig)
    : jetToken_(consumes<TkJetWordCollection>(iConfig.getParameter<edm::InputTag>("L1TkJetEmulationInputTag"))) {
  debug_ = iConfig.getParameter<bool>("debug");
  displaced_ = iConfig.getParameter<bool>("displaced");

  jetMinPt_ = l1tmhtemu::digitizeSignedValue<l1tmhtemu::pt_t>(
      (float)iConfig.getParameter<double>("jet_minPt"), l1tmhtemu::kInternalPtWidth, l1tmhtemu::kStepPt);
  jetMaxEta_ = l1tmhtemu::digitizeSignedValue<l1tmhtemu::eta_t>(
      (float)iConfig.getParameter<double>("jet_maxEta"), l1tmhtemu::kInternalEtaWidth, l1tmhtemu::kStepEta);
  minNtracksHighPt_ = (l1tmhtemu::ntracks_t)iConfig.getParameter<int>("jet_minNtracksHighPt");
  minNtracksLowPt_ = (l1tmhtemu::ntracks_t)iConfig.getParameter<int>("jet_minNtracksLowPt");

  cosLUTbins = floor(l1tmhtemu::kMaxCosLUTPhi / l1tmhtemu::kStepPhi);
  cosLUT_ = l1tmhtemu::generateCosLUT(cosLUTbins);

  atanLUT_ = l1tmhtemu::generateaTanLUT(aSteps);
  magNormalisationLUT_ = l1tmhtemu::generatemagNormalisationLUT(aSteps);

  // Name of output ED Product
  L1MHTCollectionName_ = (std::string)iConfig.getParameter<std::string>("L1MHTCollectionName");

  produces<std::vector<EtSum>>(L1MHTCollectionName_);

  if (debug_) {
    edm::LogVerbatim("L1TrackerHTMissEmulatorProducer")
        << "-------------------------------------------------------------------------\n"
        << "====BITWIDTHS====\n"
        << "pt: " << l1t::TkJetWord::TkJetBitWidths::kPtSize << " eta: " << l1t::TkJetWord::TkJetBitWidths::kGlbEtaSize
        << " phi:" << l1t::TkJetWord::TkJetBitWidths::kGlbPhiSize << "\n"
        << "====CUT AP_INTS====\n"
        << "minpt: " << jetMinPt_ << " maxeta: " << jetMaxEta_ << " minNtracksHighPt: " << minNtracksHighPt_
        << " minNtracksLowPt: " << minNtracksLowPt_ << "\n"
        << "====CUT AP_INTS TO FLOATS====\n"
        << "minpt: " << (float)jetMinPt_ * l1tmhtemu::kStepPt << " maxeta: " << (float)jetMaxEta_ * l1tmhtemu::kStepEta
        << " minNtracksHighPt: " << (int)minNtracksHighPt_ << " minNtracksLowPt: " << (int)minNtracksLowPt_ << "\n"
        << "-------------------------------------------------------------------------\n";
  }
}

void L1TkHTMissEmulatorProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  std::unique_ptr<std::vector<l1t::EtSum>> MHTCollection(new std::vector<l1t::EtSum>(0));

  // L1 track-trigger jets
  edm::Handle<TkJetWordCollection> L1TkJetsHandle;
  iEvent.getByToken(jetToken_, L1TkJetsHandle);
  std::vector<TkJetWord>::const_iterator jetIter;

  if (!L1TkJetsHandle.isValid() && !displaced_) {
    LogError("TkHTMissEmulatorProducer") << "\nWarning: TkJetCollection not found in the event. Exit\n";
    return;
  }

  if (!L1TkJetsHandle.isValid() && displaced_) {
    LogError("TkHTMissEmulatorProducer") << "\nWarning: TkJetExtendedCollection not found in the event. Exit\n";
    return;
  }

  // floats used for debugging
  float sumPx_ = 0;
  float sumPy_ = 0;
  float HT_ = 0;

  l1tmhtemu::Et_t sumPx = 0;
  l1tmhtemu::Et_t sumPy = 0;
  l1tmhtemu::MHT_t HT = 0;

  // loop over jets
  int jetn = 0;

  for (jetIter = L1TkJetsHandle->begin(); jetIter != L1TkJetsHandle->end(); ++jetIter) {
    // floats used for debugging
    float tmp_jet_px_ = jetIter->pt() * cos(jetIter->glbphi());
    float tmp_jet_py_ = jetIter->pt() * sin(jetIter->glbphi());
    //float tmp_jet_et_ = jetIter->pt();  // FIXME Get Et from the emulated jets
    float tmp_jet_pt_ = jetIter->pt();

    l1tmhtemu::pt_t tmp_jet_pt =
        l1tmhtemu::digitizeSignedValue<l1tmhtemu::pt_t>(jetIter->pt(), l1tmhtemu::kInternalPtWidth, l1tmhtemu::kStepPt);
    l1tmhtemu::eta_t tmp_jet_eta = l1tmhtemu::digitizeSignedValue<l1tmhtemu::eta_t>(
        jetIter->glbeta(), l1tmhtemu::kInternalEtaWidth, l1tmhtemu::kStepEta);
    l1tmhtemu::phi_t tmp_jet_phi = l1tmhtemu::digitizeSignedValue<l1tmhtemu::phi_t>(
        jetIter->glbphi(), l1tmhtemu::kInternalPhiWidth, l1tmhtemu::kStepPhi);
    l1tmhtemu::ntracks_t tmp_jet_nt = l1tmhtemu::ntracks_t(jetIter->nt());

    l1tmhtemu::phi_t tmp_jet_cos_phi = l1tmhtemu::phi_t(-999);
    l1tmhtemu::phi_t tmp_jet_sin_phi = l1tmhtemu::phi_t(-999);

    if (tmp_jet_phi >= 0) {
      tmp_jet_cos_phi = cosLUT_[tmp_jet_phi];

      if (cosLUTbins / 2 + 1 - tmp_jet_phi >= 0)
        tmp_jet_sin_phi = cosLUT_[cosLUTbins / 2 + 1 - tmp_jet_phi];
      else
        tmp_jet_sin_phi = cosLUT_[-1 * (cosLUTbins / 2 + 1 - tmp_jet_phi)];

    } else {
      tmp_jet_cos_phi = cosLUT_[-1 * tmp_jet_phi];

      if (cosLUTbins / 2 + 1 - (-1 * tmp_jet_phi) >= 0)
        tmp_jet_sin_phi = -1 * cosLUT_[cosLUTbins / 2 + 1 - (-1 * tmp_jet_phi)];
      else
        tmp_jet_sin_phi = -1 * cosLUT_[-1 * (cosLUTbins / 2 + 1 - (-1 * tmp_jet_phi))];
    }

    l1tmhtemu::Et_t tmp_jet_px = tmp_jet_pt * tmp_jet_cos_phi;
    l1tmhtemu::Et_t tmp_jet_py = tmp_jet_pt * tmp_jet_sin_phi;

    jetn++;

    if (debug_) {
      edm::LogVerbatim("L1TrackerHTMissEmulatorProducer")
          << "****JET EMULATION" << jetn << "****\n"
          << "FLOATS ORIGINAL\n"
          << "PT: " << jetIter->pt() << "| ETA: " << jetIter->glbeta() << "| PHI: " << jetIter->glbphi()
          << "| NTRACKS: " << jetIter->nt() << "| COS(PHI): " << cos(jetIter->glbphi())
          << "| SIN(PHI): " << sin(jetIter->glbphi()) << "| Px: " << jetIter->pt() * cos(jetIter->glbphi())
          << "| Py: " << jetIter->pt() * sin(jetIter->glbphi()) << "\n"
          << "AP_INTS RAW\n"
          << "PT: " << jetIter->ptWord() << "| ETA: " << jetIter->glbEtaWord() << "| PHI: " << jetIter->glbPhiWord()
          << "| NTRACKS: " << jetIter->ntWord() << "\n"
          << "AP_INTS NEW\n"
          << "PT: " << tmp_jet_pt << "| ETA: " << tmp_jet_eta << "| PHI: " << tmp_jet_phi << "| NTRACKS: " << tmp_jet_nt
          << "| COS(PHI): " << tmp_jet_cos_phi << "| SIN(PHI): " << tmp_jet_sin_phi << "| Px: " << tmp_jet_px
          << "| Py: " << tmp_jet_py << "\n"
          << "AP_INTS NEW TO FLOATS\n"
          << "PT: " << (float)tmp_jet_pt * l1tmhtemu::kStepPt << "| ETA: " << (float)tmp_jet_eta * l1tmhtemu::kStepEta
          << "| PHI: " << (float)tmp_jet_phi * l1tmhtemu::kStepPhi << "| NTRACKS: " << (int)tmp_jet_nt
          << "| COS(PHI): " << (float)tmp_jet_cos_phi * l1tmhtemu::kStepPhi
          << "| SIN(PHI): " << (float)tmp_jet_sin_phi * l1tmhtemu::kStepPhi
          << "| Px: " << (float)tmp_jet_px * l1tmhtemu::kStepPt * l1tmhtemu::kStepPhi
          << "| Py: " << (float)tmp_jet_py * l1tmhtemu::kStepPt * l1tmhtemu::kStepPhi << "\n"
          << "-------------------------------------------------------------------------\n";
    }

    if (tmp_jet_pt < jetMinPt_)
      continue;
    if (tmp_jet_eta > jetMaxEta_ or tmp_jet_eta < -1 * jetMaxEta_)
      continue;
    if (tmp_jet_nt < minNtracksLowPt_ && tmp_jet_pt > 200)
      continue;
    if (tmp_jet_nt < minNtracksHighPt_ && tmp_jet_pt > 400)
      continue;

    if (debug_) {
      sumPx_ += tmp_jet_px_;
      sumPy_ += tmp_jet_py_;
      HT_ += tmp_jet_pt_;
    }

    sumPx += tmp_jet_pt * tmp_jet_cos_phi;
    sumPy += tmp_jet_pt * tmp_jet_sin_phi;
    HT += tmp_jet_pt;

  }  // end jet loop

  // define missing HT

  // Perform cordic sqrt, take x,y and converts to polar coordinate r,phi where
  // r=sqrt(x**2+y**2) and phi = atan(y/x)
  l1tmhtemu::EtMiss EtMiss = l1tmhtemu::cordicSqrt(sumPx, sumPy, aSteps, atanLUT_, magNormalisationLUT_);
  math::XYZTLorentzVector missingEt(-sumPx, -sumPy, 0, EtMiss.Et);

  l1tmhtemu::MHTphi_t phi = 0;

  if ((sumPx < 0) && (sumPy < 0))
    phi = EtMiss.Phi - l1tmhtemu::kMHTPhiBins / 2;
  else if ((sumPx >= 0) && (sumPy >= 0))
    phi = (EtMiss.Phi) + l1tmhtemu::kMHTPhiBins / 2;
  else if ((sumPx >= 0) && (sumPy < 0))
    phi = EtMiss.Phi - l1tmhtemu::kMHTPhiBins / 2;
  else if ((sumPx < 0) && (sumPy >= 0))
    phi = EtMiss.Phi - 3 * l1tmhtemu::kMHTPhiBins / 2;

  if (debug_) {
    edm::LogVerbatim("L1TrackerHTMissEmulatorProducer")
        << "-------------------------------------------------------------------------\n"
        << "====MHT FLOATS====\n"
        << "sumPx: " << sumPx_ << "| sumPy: " << sumPy_ << "| ET: " << sqrt(sumPx_ * sumPx_ + sumPy_ * sumPy_)
        << "| HT: " << HT_ << "| PHI: " << atan2(sumPy_, sumPx_) << "\n"
        << "====MHT AP_INTS====\n"
        << "sumPx: " << sumPx << "| sumPy: " << sumPy << "| ET: " << EtMiss.Et << "| HT: " << HT << "| PHI: " << phi
        << "\n"
        << "====MHT AP_INTS TO FLOATS====\n"
        << "sumPx: " << (float)sumPx * l1tmhtemu::kStepPt * l1tmhtemu::kStepPhi
        << "| sumPy: " << (float)sumPy * l1tmhtemu::kStepPt * l1tmhtemu::kStepPhi << "| ET: " << EtMiss.Et.to_double()
        << "| HT: " << (float)HT * l1tmhtemu::kStepPt << "| PHI: " << (float)phi * l1tmhtemu::kStepMHTPhi - M_PI << "\n"
        << "-------------------------------------------------------------------------\n";
  }
  //rescale HT to correct output range
  HT = HT / (int)(1 / l1tmhtemu::kStepPt);

  EtSum L1HTSum(missingEt, EtSum::EtSumType::kMissingHt, (int)HT.range(), 0, (int)phi, (int)jetn);

  MHTCollection->push_back(L1HTSum);
  iEvent.put(std::move(MHTCollection), L1MHTCollectionName_);

}  //end producer

void L1TkHTMissEmulatorProducer::beginJob() {}

void L1TkHTMissEmulatorProducer::endJob() {}

DEFINE_FWK_MODULE(L1TkHTMissEmulatorProducer);

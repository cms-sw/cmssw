// -*- C++ -*-
//
// Package:    DQM/DetectorStatus
// Class:      DetectorStatus
//
/**\class DetectorStatus DetectorStatus.cc DQM/DetectorStatus/plugins/DetectorStatus.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrius Kirilovas
//         Created:  Thu, 06 Dec 2018 11:49:14 GMT
//

#include <cstdint>
#include <string>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Scalers/interface/DcsStatus.h"
#include "DataFormats/TCDS/interface/TCDSRecord.h"

//
// class declaration
//

class DetectorStatus : public DQMEDAnalyzer {
public:
  explicit DetectorStatus(const edm::ParameterSet&);
  ~DetectorStatus() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker &,
                              edm::Run const&,
                              edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  edm::EDGetTokenT<DcsStatusCollection> dcsStatusCollection_;
  edm::EDGetTokenT<TCDSRecord> tcdsRecord_;

  std::string folder_;
  MonitorElement* perLuminosity_;

  // Numbers of each of the vertical bins
  const static int VBIN_VALID = 1;
  const static int VBIN_PHYSICS_DECLARED = 2;
  const static int VBIN_STABLE_BEAM = 3;
  const static int VBIN_MOMENTUM = 4;

  const static int VBIN_EB_P = 5;
  const static int VBIN_EB_M = 6;
  const static int VBIN_EE_P = 7;
  const static int VBIN_EE_M = 8;
  const static int VBIN_HBHE_A = 9;
  const static int VBIN_HBHE_B = 10;
  const static int VBIN_HBHE_C = 11;
  const static int VBIN_HF = 12;
  const static int VBIN_HO = 13;
  const static int VBIN_RPC = 14;
  const static int VBIN_DT_0 = 15;
  const static int VBIN_DT_P = 16;
  const static int VBIN_DT_M = 17;
  const static int VBIN_CSC_P = 18;
  const static int VBIN_CSC_M = 19;
  const static int VBIN_CASTOR = 20;
  const static int VBIN_TOB = 21;
  const static int VBIN_TIBTID = 22;
  const static int VBIN_TEC_P = 23;
  const static int VBIN_TEC_M = 24;
  const static int VBIN_BPIX = 25;
  const static int VBIN_FPIX = 26;
  const static int VBIN_ES_P = 27;
  const static int VBIN_ES_M = 28;
  const static int VBIN_ZDC = 29;

  // To max amount of lumisections we foresee for the plots
  // DQM GUI renderplugins provide scaling to actual amount
  const static int MAX_LUMIS = 2000; //2000
  const static int MAX_VBINS = 29;

  // Beam momentum at flat top, used to determine if collisions are
  // occurring with the beams at the energy allowed for physics production.
  const static int MAX_MOMENTUM = 6500;

  // Beam momentum allowed offset: it is a momentum value subtracted to
  // maximum momentum in order to decrease the threshold for beams going to
  // collisions for physics production. This happens because BST sends from
  // time to time a value of the beam momentum slightly below the nominal values,
  // even during stable collisions: in this way, we provide a correct information
  // at the cost of not requiring the exact momentum being measured by BST.
  const static int MOMENTUM_OFFSET = 1;

  bool dcsBits_[MAX_VBINS];
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DetectorStatus::DetectorStatus(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")) {

  dcsStatusCollection_ =
      consumes<DcsStatusCollection>(iConfig.getUntrackedParameter<edm::InputTag>(
          "dcsStatusCollection", edm::InputTag("scalersRawToDigi")));

  // Used to get the BST record from the TCDS information
  tcdsRecord_ = consumes<TCDSRecord>(iConfig.getUntrackedParameter<edm::InputTag>(
    "tcdsData", edm::InputTag("tcdsDigis", "tcdsRecord")));
}

DetectorStatus::~DetectorStatus() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void DetectorStatus::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::Handle<DcsStatusCollection> dcsStatus;
  edm::Handle<TCDSRecord> tcdsData;

  iEvent.getByToken(dcsStatusCollection_, dcsStatus);
  iEvent.getByToken(tcdsRecord_, tcdsData);

  int currentLS = iEvent.getLuminosityBlock().id().luminosityBlock();

  // Set Valid for every event
  // Value will be green if at least one event was processed for lumisection
  perLuminosity_->Fill(currentLS, VBIN_VALID, 1.);
  
  // Set every DCS bit to false initially
  for(int i = 0; i < MAX_VBINS; i++) {
    dcsBits_[i] = false;
  }

  // Set every DCS bin
  for (auto const & dcsStatusItr : *dcsStatus) {
    perLuminosity_->Fill(currentLS, VBIN_EB_P, dcsStatusItr.ready(DcsStatus::EBp));
    perLuminosity_->Fill(currentLS, VBIN_EB_M, dcsStatusItr.ready(DcsStatus::EBm));
    perLuminosity_->Fill(currentLS, VBIN_EE_P, dcsStatusItr.ready(DcsStatus::EEp));
    perLuminosity_->Fill(currentLS, VBIN_EE_M, dcsStatusItr.ready(DcsStatus::EEm));
    perLuminosity_->Fill(currentLS, VBIN_HBHE_A, dcsStatusItr.ready(DcsStatus::HBHEa));
    perLuminosity_->Fill(currentLS, VBIN_HBHE_B, dcsStatusItr.ready(DcsStatus::HBHEb));
    perLuminosity_->Fill(currentLS, VBIN_HBHE_C, dcsStatusItr.ready(DcsStatus::HBHEc));
    perLuminosity_->Fill(currentLS, VBIN_HF, dcsStatusItr.ready(DcsStatus::HF));
    perLuminosity_->Fill(currentLS, VBIN_HO, dcsStatusItr.ready(DcsStatus::HO));
    perLuminosity_->Fill(currentLS, VBIN_RPC, dcsStatusItr.ready(DcsStatus::RPC));
    perLuminosity_->Fill(currentLS, VBIN_DT_0, dcsStatusItr.ready(DcsStatus::DT0));
    perLuminosity_->Fill(currentLS, VBIN_DT_P, dcsStatusItr.ready(DcsStatus::DTp));
    perLuminosity_->Fill(currentLS, VBIN_DT_M, dcsStatusItr.ready(DcsStatus::DTm));
    perLuminosity_->Fill(currentLS, VBIN_CSC_P, dcsStatusItr.ready(DcsStatus::CSCp));
    perLuminosity_->Fill(currentLS, VBIN_CSC_M, dcsStatusItr.ready(DcsStatus::CSCm));
    perLuminosity_->Fill(currentLS, VBIN_CASTOR, dcsStatusItr.ready(DcsStatus::CASTOR));
    perLuminosity_->Fill(currentLS, VBIN_TOB, dcsStatusItr.ready(DcsStatus::TOB));
    perLuminosity_->Fill(currentLS, VBIN_TIBTID, dcsStatusItr.ready(DcsStatus::TIBTID));
    perLuminosity_->Fill(currentLS, VBIN_TEC_P, dcsStatusItr.ready(DcsStatus::TECp));
    perLuminosity_->Fill(currentLS, VBIN_TEC_M, dcsStatusItr.ready(DcsStatus::TECm));
    perLuminosity_->Fill(currentLS, VBIN_BPIX, dcsStatusItr.ready(DcsStatus::BPIX));
    perLuminosity_->Fill(currentLS, VBIN_FPIX, dcsStatusItr.ready(DcsStatus::FPIX));
    perLuminosity_->Fill(currentLS, VBIN_ES_P, dcsStatusItr.ready(DcsStatus::ESp));
    perLuminosity_->Fill(currentLS, VBIN_ES_M, dcsStatusItr.ready(DcsStatus::ESm));
    perLuminosity_->Fill(currentLS, VBIN_ZDC, dcsStatusItr.ready(DcsStatus::ZDC));

    // Collect information for Physics Declared bin
    dcsBits_[VBIN_BPIX] &= dcsStatusItr.ready(DcsStatus::BPIX);
    dcsBits_[VBIN_FPIX] &= dcsStatusItr.ready(DcsStatus::FPIX);
    dcsBits_[VBIN_TIBTID] &= dcsStatusItr.ready(DcsStatus::TIBTID);
    dcsBits_[VBIN_TOB] &= dcsStatusItr.ready(DcsStatus::TOB);
    dcsBits_[VBIN_TEC_P] &= dcsStatusItr.ready(DcsStatus::TECp);
    dcsBits_[VBIN_TEC_M] &= dcsStatusItr.ready(DcsStatus::TECm);
    dcsBits_[VBIN_CSC_P] &= dcsStatusItr.ready(DcsStatus::CSCp);
    dcsBits_[VBIN_CSC_M] &= dcsStatusItr.ready(DcsStatus::CSCm);
    dcsBits_[VBIN_DT_0] &= dcsStatusItr.ready(DcsStatus::DT0);
    dcsBits_[VBIN_DT_P] &= dcsStatusItr.ready(DcsStatus::DTp);
    dcsBits_[VBIN_DT_M] &= dcsStatusItr.ready(DcsStatus::DTm);
    dcsBits_[VBIN_RPC] &= dcsStatusItr.ready(DcsStatus::RPC);
  }

  if(tcdsData.isValid()) {
    uint16_t beamMode = tcdsData->getBST().getBeamMode();
    int32_t momentum = tcdsData->getBST().getBeamMomentum();

    // Set Stable Beam bin
    dcsBits_[VBIN_STABLE_BEAM] = beamMode == 11;

    // Set Momentum bin
    dcsBits_[VBIN_MOMENTUM] = momentum >= MAX_MOMENTUM - MOMENTUM_OFFSET;

    // Set Physics Declared bin
    dcsBits_[VBIN_PHYSICS_DECLARED] = (beamMode == 11)
                                      && (dcsBits_[VBIN_BPIX] && dcsBits_[VBIN_FPIX] && dcsBits_[VBIN_TIBTID] 
                                          && dcsBits_[VBIN_TOB] && dcsBits_[VBIN_TEC_P] && dcsBits_[VBIN_TEC_M])
                                      && (dcsBits_[VBIN_CSC_P] || dcsBits_[VBIN_CSC_M]
                                          || dcsBits_[VBIN_DT_0] || dcsBits_[VBIN_DT_P] || dcsBits_[VBIN_DT_M]
                                          || dcsBits_[VBIN_RPC]);
  }
  else {
    edm::LogWarning("DetectorStatus") << "TCDS Data inaccessible.";
  }

  perLuminosity_->Fill(currentLS, VBIN_STABLE_BEAM, dcsBits_[VBIN_STABLE_BEAM] ? 1. : 0.);
  perLuminosity_->Fill(currentLS, VBIN_PHYSICS_DECLARED, dcsBits_[VBIN_PHYSICS_DECLARED] ? 1. : 0.);
  perLuminosity_->Fill(currentLS, VBIN_MOMENTUM, dcsBits_[VBIN_MOMENTUM] ? 1. : 0.);
}

void DetectorStatus::bookHistograms(DQMStore::IBooker & ibook,
                               edm::Run const& run,
                               edm::EventSetup const & iSetup) {
  ibook.setCurrentFolder(folder_);

  perLuminosity_ = ibook.bookProfile2D("perLumisection", "Detector and LHC Status per Lumisection", 
                                MAX_LUMIS, 1., MAX_LUMIS, MAX_VBINS, 0.5, MAX_VBINS + 0.5, 0., 1.);
  
  perLuminosity_->setAxisTitle("Luminosity Section");

  perLuminosity_->setBinLabel(VBIN_VALID, "Valid", 2);
  perLuminosity_->setBinLabel(VBIN_PHYSICS_DECLARED, "PhysDecl", 2);
  perLuminosity_->setBinLabel(VBIN_STABLE_BEAM, "Stable B", 2);
  perLuminosity_->setBinLabel(VBIN_MOMENTUM, "13 TeV", 2);
  perLuminosity_->setBinLabel(VBIN_EB_P, "EB+", 2);
  perLuminosity_->setBinLabel(VBIN_EB_M, "EB-", 2);
  perLuminosity_->setBinLabel(VBIN_EE_P, "EE+", 2);
  perLuminosity_->setBinLabel(VBIN_EE_M, "EE-", 2);
  perLuminosity_->setBinLabel(VBIN_HBHE_A, "HBHEa", 2);
  perLuminosity_->setBinLabel(VBIN_HBHE_B, "HBHEb", 2);
  perLuminosity_->setBinLabel(VBIN_HBHE_C, "HBHEc", 2);
  perLuminosity_->setBinLabel(VBIN_HF, "HF", 2);
  perLuminosity_->setBinLabel(VBIN_HO, "HO", 2);
  perLuminosity_->setBinLabel(VBIN_RPC, "RPC", 2);
  perLuminosity_->setBinLabel(VBIN_DT_0, "DT0", 2);
  perLuminosity_->setBinLabel(VBIN_DT_P, "DT+", 2);
  perLuminosity_->setBinLabel(VBIN_DT_M, "DT-", 2);
  perLuminosity_->setBinLabel(VBIN_CSC_P, "CSC+", 2);
  perLuminosity_->setBinLabel(VBIN_CSC_M, "CSC-", 2);
  perLuminosity_->setBinLabel(VBIN_CASTOR, "CASTOR", 2);
  perLuminosity_->setBinLabel(VBIN_TOB, "TOB", 2);
  perLuminosity_->setBinLabel(VBIN_TIBTID, "TIBTID", 2);
  perLuminosity_->setBinLabel(VBIN_TEC_P, "TECp", 2);
  perLuminosity_->setBinLabel(VBIN_TEC_M , "TECm", 2);
  perLuminosity_->setBinLabel(VBIN_BPIX, "BPIX", 2);
  perLuminosity_->setBinLabel(VBIN_FPIX, "FPIX", 2);
  perLuminosity_->setBinLabel(VBIN_ES_P, "ES+", 2);
  perLuminosity_->setBinLabel(VBIN_ES_M, "ES-", 2);
  perLuminosity_->setBinLabel(VBIN_ZDC, "ZDC", 2);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void DetectorStatus::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // The following says we do not know what parameters are allowed so do no
  // validation
  // Please change this to state exactly what you do use, even if it is no
  // parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("folder", "DQM/DetectorStatus");
  descriptions.add("detectorstatus", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DetectorStatus);

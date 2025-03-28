#include "L1Trigger/L1TTrackMatch/interface/DisplacedVertexProducer.h"
#include "L1TrackUnpacker.h"

using namespace l1trackunpacker;

double DisplacedVertexProducer::FloatPtFromBits(const L1TTTrackType& track) const {
  ap_uint<14> ptEmulationBits = track.getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1,
                                                     TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
  ap_ufixed<14, 9> ptEmulation;
  ptEmulation.V = (ptEmulationBits.range());
  return ptEmulation.to_double();
}

double DisplacedVertexProducer::FloatEtaFromBits(const L1TTTrackType& track) const {
  TTTrack_TrackWord::tanl_t etaBits = track.getTanlWord();
  glbeta_intern digieta;
  digieta.V = etaBits.range();
  return (double)digieta;
}

double DisplacedVertexProducer::FloatPhiFromBits(const L1TTTrackType& track) const {
  int Sector = track.phiSector();
  double sector_phi_value = 0;
  if (Sector < 5) {
    sector_phi_value = 2.0 * M_PI * Sector / 9.0;
  } else {
    sector_phi_value = (-1.0 * M_PI + M_PI / 9.0 + (Sector - 5) * 2.0 * M_PI / 9.0);
  }
  glbphi_intern trkphiSector = DoubleToBit(
      sector_phi_value, TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
  glbphi_intern local_phiBits = 0;
  local_phiBits.V = track.getPhiWord();

  glbphi_intern local_phi =
      DoubleToBit(BitToDouble(local_phiBits, TTTrack_TrackWord::TrackBitWidths::kPhiSize, TTTrack_TrackWord::stepPhi0),
                  TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit,
                  TTTrack_TrackWord::stepPhi0);
  glbphi_intern digiphi = local_phi + trkphiSector;
  return BitToDouble(
      digiphi, TTTrack_TrackWord::TrackBitWidths::kPhiSize + kExtraGlobalPhiBit, TTTrack_TrackWord::stepPhi0);
}

double DisplacedVertexProducer::FloatZ0FromBits(const L1TTTrackType& track) const {
  z0_intern trkZ = track.getZ0Word();
  return BitToDouble(trkZ, TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0);
}

double DisplacedVertexProducer::FloatD0FromBits(const L1TTTrackType& track) const {
  d0_intern trkD0 = track.getD0Word();
  return BitToDouble(trkD0, TTTrack_TrackWord::TrackBitWidths::kD0Size, TTTrack_TrackWord::stepD0);
}

int DisplacedVertexProducer::ChargeFromBits(const L1TTTrackType& track) const {
  ap_uint<1> chargeBit = track.getTrackWord()[TTTrack_TrackWord::TrackBitLocations::kRinvMSB];
  return 1 - (2 * chargeBit.to_uint());
}

double convertPtToR(double pt) {
  return 100.0 * (1.0 / (0.3 * 3.8)) * pt;  //returns R in cm
}

bool ComparePtTrack(std::pair<Track_Parameters, edm::Ptr<TrackingParticle>> a,
                    std::pair<Track_Parameters, edm::Ptr<TrackingParticle>> b) {
  return a.first.pt > b.first.pt;
}

Double_t dist(Double_t x1, Double_t y1, Double_t x2 = 0, Double_t y2 = 0) {  // Distance between 2 points
  return (TMath::Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)));
}

Double_t dist_TPs(Track_Parameters a, Track_Parameters b) {
  float x1 = a.x0;   //   Centers of the circles
  float y1 = a.y0;   //
  float x2 = b.x0;   //
  float y2 = b.y0;   //
  float R1 = a.rho;  // Radii of the circles
  float R2 = b.rho;
  float R = dist(x1, y1, x2, y2);  // Distance between centers
  if ((R >= fabs(R1 - R2)) && (R <= (R1 + R2))) {
    return (0);
  } else if (R == 0) {
    return (-99999.0);
  } else {
    return (R - R1 - R2);
  }
}

Int_t calcVertex(Track_Parameters a, Track_Parameters b, Double_t& x_vtx, Double_t& y_vtx, Double_t& z_vtx) {
  float x1 = a.x0;   //   Centers of the circles
  float y1 = a.y0;   //
  float x2 = b.x0;   //
  float y2 = b.y0;   //
  float R1 = a.rho;  // Radii of the circles
  float R2 = b.rho;
  float R = dist(x1, y1, x2, y2);  // Distance between centers
  if (R == 0)
    return -1;
  float co1 = (pow(R1, 2) - pow(R2, 2)) / (2 * pow(R, 2));
  float radicand = (2 / pow(R, 2)) * (pow(R1, 2) + pow(R2, 2)) - (pow(pow(R1, 2) - pow(R2, 2), 2) / pow(R, 4)) - 1;
  float co2 = 0;
  if (radicand > 0)
    co2 = 0.5 * TMath::Sqrt(radicand);
  float ix1_x = 0.5 * (x1 + x2) + co1 * (x2 - x1) + co2 * (y2 - y1);
  float ix2_x = 0.5 * (x1 + x2) + co1 * (x2 - x1) - co2 * (y2 - y1);
  float ix1_y = 0.5 * (y1 + y2) + co1 * (y2 - y1) + co2 * (x1 - x2);
  float ix2_y = 0.5 * (y1 + y2) + co1 * (y2 - y1) - co2 * (x1 - x2);
  float ix1_z1 = a.z(ix1_x, ix1_y);
  float ix1_z2 = b.z(ix1_x, ix1_y);
  float ix1_delz = fabs(ix1_z1 - ix1_z2);
  float ix2_z1 = a.z(ix2_x, ix2_y);
  float ix2_z2 = b.z(ix2_x, ix2_y);
  float ix2_delz = fabs(ix2_z1 - ix2_z2);
  float trk1_POCA[2] = {a.d0 * sin(a.phi), -1 * a.d0 * cos(a.phi)};
  float trk2_POCA[2] = {b.d0 * sin(b.phi), -1 * b.d0 * cos(b.phi)};
  float trk1_ix1_delxy[2] = {ix1_x - trk1_POCA[0], ix1_y - trk1_POCA[1]};
  float trk1_ix2_delxy[2] = {ix2_x - trk1_POCA[0], ix2_y - trk1_POCA[1]};
  float trk2_ix1_delxy[2] = {ix1_x - trk2_POCA[0], ix1_y - trk2_POCA[1]};
  float trk2_ix2_delxy[2] = {ix2_x - trk2_POCA[0], ix2_y - trk2_POCA[1]};
  float trk1_traj[2] = {cos(a.phi), sin(a.phi)};
  float trk2_traj[2] = {cos(b.phi), sin(b.phi)};
  bool trk1_ix1_inTraj = ((trk1_ix1_delxy[0] * trk1_traj[0] + trk1_ix1_delxy[1] * trk1_traj[1]) > 0) ? true : false;
  bool trk1_ix2_inTraj = ((trk1_ix2_delxy[0] * trk1_traj[0] + trk1_ix2_delxy[1] * trk1_traj[1]) > 0) ? true : false;
  bool trk2_ix1_inTraj = ((trk2_ix1_delxy[0] * trk2_traj[0] + trk2_ix1_delxy[1] * trk2_traj[1]) > 0) ? true : false;
  bool trk2_ix2_inTraj = ((trk2_ix2_delxy[0] * trk2_traj[0] + trk2_ix2_delxy[1] * trk2_traj[1]) > 0) ? true : false;
  if (trk1_ix1_inTraj && trk2_ix1_inTraj && trk1_ix2_inTraj && trk2_ix2_inTraj) {
    if (ix1_delz < ix2_delz) {
      x_vtx = ix1_x;
      y_vtx = ix1_y;
      z_vtx = (ix1_z1 + ix1_z2) / 2;
      return 0;
    } else {
      x_vtx = ix2_x;
      y_vtx = ix2_y;
      z_vtx = (ix2_z1 + ix2_z2) / 2;
      return 0;
    }
  }
  if (trk1_ix1_inTraj && trk2_ix1_inTraj) {
    x_vtx = ix1_x;
    y_vtx = ix1_y;
    z_vtx = (ix1_z1 + ix1_z2) / 2;
    return 1;
  }
  if (trk1_ix2_inTraj && trk2_ix2_inTraj) {
    x_vtx = ix2_x;
    y_vtx = ix2_y;
    z_vtx = (ix2_z1 + ix2_z2) / 2;
    return 2;
  } else {
    if (ix1_delz < ix2_delz) {
      x_vtx = ix1_x;
      y_vtx = ix1_y;
      z_vtx = (ix1_z1 + ix1_z2) / 2;
      return 3;
    } else {
      x_vtx = ix2_x;
      y_vtx = ix2_y;
      z_vtx = (ix2_z1 + ix2_z2) / 2;
      return 3;
    }
  }
  return 4;
}

DisplacedVertexProducer::DisplacedVertexProducer(const edm::ParameterSet& iConfig)
    : ttTrackMCTruthToken_(consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>>(
          iConfig.getParameter<edm::InputTag>("mcTruthTrackInputTag"))),
      trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(
          iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
      trackGTTToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>>(
          iConfig.getParameter<edm::InputTag>("l1TracksGTTInputTag"))),
      outputVertexCollectionName_(iConfig.getParameter<std::string>("l1TrackVertexCollectionName")),
      model_(iConfig.getParameter<edm::FileInPath>("model")),
      runEmulation_(iConfig.getParameter<bool>("runEmulation")),
      cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),
      chi2rzMax_(cutSet_.getParameter<double>("chi2rzMax")),
      promptMVAMin_(cutSet_.getParameter<double>("promptMVAMin")),
      ptMin_(cutSet_.getParameter<double>("ptMin")),
      etaMax_(cutSet_.getParameter<double>("etaMax")),
      dispD0Min_(cutSet_.getParameter<double>("dispD0Min")),
      promptMVADispTrackMin_(cutSet_.getParameter<double>("promptMVADispTrackMin")),
      overlapEtaMin_(cutSet_.getParameter<double>("overlapEtaMin")),
      overlapEtaMax_(cutSet_.getParameter<double>("overlapEtaMax")),
      overlapNStubsMin_(cutSet_.getParameter<int>("overlapNStubsMin")),
      diskEtaMin_(cutSet_.getParameter<double>("diskEtaMin")),
      diskD0Min_(cutSet_.getParameter<double>("diskD0Min")),
      barrelD0Min_(cutSet_.getParameter<double>("barrelD0Min")),
      RTMin_(cutSet_.getParameter<double>("RTMin")),
      RTMax_(cutSet_.getParameter<double>("RTMax")) {
  //--- Define EDM output to be written to file (if required)
  produces<l1t::DisplacedTrackVertexCollection>(outputVertexCollectionName_);
}

void DisplacedVertexProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackGTTHandle;
  iEvent.getByToken(trackGTTToken_, TTTrackGTTHandle);
  std::vector<TTTrack<Ref_Phase2TrackerDigi_>>::const_iterator iterL1Track;
  int this_l1track = 0;
  std::vector<std::pair<Track_Parameters, edm::Ptr<TrackingParticle>>> selectedTracksWithTruth;

  //track selection loop
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> l1track_ptr(TTTrackHandle, this_l1track);
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> gtttrack_ptr(TTTrackGTTHandle, this_l1track);
    this_l1track++;

    float pt, eta, phi, z0, d0, rho, chi2rphi, chi2rz, bendchi2, MVA1;
    int nstub;

    if (runEmulation_) {
      pt = FloatPtFromBits(*gtttrack_ptr);
      eta = FloatEtaFromBits(*gtttrack_ptr);
      phi = FloatPhiFromBits(*gtttrack_ptr);
      z0 = gtttrack_ptr->getZ0();  //cm
      d0 = gtttrack_ptr->getD0();
      int charge = ChargeFromBits(*gtttrack_ptr);
      rho = charge * convertPtToR(pt);
      chi2rphi = gtttrack_ptr->getChi2RPhi();
      chi2rz = gtttrack_ptr->getChi2RZ();
      bendchi2 = gtttrack_ptr->getBendChi2();
      MVA1 = gtttrack_ptr->getMVAQuality();
      nstub = gtttrack_ptr->getNStubs();
    } else {
      pt = l1track_ptr->momentum().perp();
      eta = l1track_ptr->momentum().eta();
      phi = l1track_ptr->momentum().phi();
      z0 = l1track_ptr->z0();  //cm
      float x0 = l1track_ptr->POCA().x();
      float y0 = l1track_ptr->POCA().y();
      d0 = x0 * sin(phi) - y0 * cos(phi);
      rho = 1 / l1track_ptr->rInv();
      chi2rphi = l1track_ptr->chi2XYRed();
      chi2rz = l1track_ptr->chi2ZRed();
      bendchi2 = l1track_ptr->stubPtConsistency();
      MVA1 = l1track_ptr->trkMVA1();
      std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
          stubRefs = l1track_ptr->getStubRefs();
      nstub = (int)stubRefs.size();
    }

    if (chi2rz < chi2rzMax_ && MVA1 > promptMVAMin_ && pt > ptMin_ && fabs(eta) < etaMax_) {
      if (fabs(d0) > dispD0Min_) {
        if (MVA1 <= promptMVADispTrackMin_)
          continue;
      }
      if (fabs(eta) > overlapEtaMin_ && fabs(eta) < overlapEtaMax_) {
        if (nstub <= overlapNStubsMin_)
          continue;
      }
      if (fabs(eta) > diskEtaMin_) {
        if (fabs(d0) <= diskD0Min_)
          continue;
      }
      if (fabs(eta) <= diskEtaMin_) {
        if (fabs(d0) <= barrelD0Min_)
          continue;
      }

      Track_Parameters track =
          Track_Parameters(pt, d0, z0, eta, phi, rho, (this_l1track - 1), nstub, chi2rphi, chi2rz, bendchi2, MVA1);

      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);
      selectedTracksWithTruth.push_back(std::make_pair(track, my_tp));
    }
  }

  //vertex loop
  std::unique_ptr<l1t::DisplacedTrackVertexCollection> product(new std::vector<l1t::DisplacedTrackVertex>());
  for (int i = 0; i < int(selectedTracksWithTruth.size() - 1); i++) {
    for (int j = i + 1; j < int(selectedTracksWithTruth.size()); j++) {
      if (dist_TPs(selectedTracksWithTruth[i].first, selectedTracksWithTruth[j].first) != 0)
        continue;
      Double_t x_dv_trk = -9999.0;
      Double_t y_dv_trk = -9999.0;
      Double_t z_dv_trk = -9999.0;
      edm::Ptr<TrackingParticle> tp_i = selectedTracksWithTruth[i].second;
      edm::Ptr<TrackingParticle> tp_j = selectedTracksWithTruth[j].second;
      bool isReal = false;
      if (!tp_i.isNull() && !tp_j.isNull()) {
        bool isHard_i = false;
        bool isHard_j = false;
        if (!tp_i->genParticles().empty() && !tp_j->genParticles().empty()) {
          isHard_i = tp_i->genParticles()[0]->isHardProcess() || tp_i->genParticles()[0]->fromHardProcessFinalState();
          isHard_j = tp_j->genParticles()[0]->isHardProcess() || tp_j->genParticles()[0]->fromHardProcessFinalState();
        }

        if (tp_i->eventId().event() == 0 && tp_j->eventId().event() == 0 && fabs(tp_i->vx() - tp_j->vx()) < 0.0001 &&
            fabs(tp_i->vy() - tp_j->vy()) < 0.0001 && fabs(tp_i->vz() - tp_j->vz()) < 0.0001 && isHard_i && isHard_j &&
            ((tp_i->charge() + tp_j->charge()) == 0)) {
          isReal = true;
        }
      }

      int inTraj =
          calcVertex(selectedTracksWithTruth[i].first, selectedTracksWithTruth[j].first, x_dv_trk, y_dv_trk, z_dv_trk);
      Vertex_Parameters vertex = Vertex_Parameters(
          x_dv_trk, y_dv_trk, z_dv_trk, selectedTracksWithTruth[i].first, selectedTracksWithTruth[j].first);

      if (vertex.R_T > RTMax_)
        continue;
      if (vertex.R_T < RTMin_)
        continue;

      l1t::DisplacedTrackVertex outputVertex = l1t::DisplacedTrackVertex(selectedTracksWithTruth[i].first.index,
                                                                         selectedTracksWithTruth[j].first.index,
                                                                         inTraj,
                                                                         vertex.d_T,
                                                                         vertex.R_T,
                                                                         vertex.cos_T,
                                                                         vertex.delta_z,
                                                                         vertex.x_dv,
                                                                         vertex.y_dv,
                                                                         vertex.z_dv,
                                                                         vertex.openingAngle,
                                                                         vertex.p_mag,
                                                                         isReal);

      //Rescaling input features so they all fall within [-20,20]. This reduces bits needed in emulation by 2. See this presentation for more information: https://indico.cern.ch/event/1476881/contributions/6219913/attachments/2964052/5214060/GTT%20Displaced%20Vertexing%20November%208%202024.pdf
      float ptRescaling = 0.25;
      float deltaZRescaling = 0.125;
      if (runEmulation_) {
        std::vector<ap_fixed<13, 8, AP_RND_CONV, AP_SAT>> Transformed_features = {
            selectedTracksWithTruth[i].first.pt * ptRescaling,
            selectedTracksWithTruth[j].first.pt * ptRescaling,
            selectedTracksWithTruth[i].first.eta,
            selectedTracksWithTruth[j].first.eta,
            selectedTracksWithTruth[i].first.phi,
            selectedTracksWithTruth[j].first.phi,
            selectedTracksWithTruth[i].first.d0,
            selectedTracksWithTruth[j].first.d0,
            selectedTracksWithTruth[i].first.z0,
            selectedTracksWithTruth[j].first.z0,
            selectedTracksWithTruth[i].first.chi2rz,
            selectedTracksWithTruth[j].first.chi2rz,
            selectedTracksWithTruth[i].first.bendchi2,
            selectedTracksWithTruth[j].first.bendchi2,
            selectedTracksWithTruth[i].first.MVA1,
            selectedTracksWithTruth[j].first.MVA1,
            vertex.d_T,
            vertex.R_T,
            vertex.cos_T,
            vertex.delta_z * deltaZRescaling};
        conifer::BDT<ap_fixed<13, 8, AP_RND_CONV, AP_SAT>, ap_fixed<13, 8, AP_RND_CONV, AP_SAT>> bdt(
            this->model_.fullPath());
        std::vector<ap_fixed<13, 8, AP_RND_CONV, AP_SAT>> output = bdt.decision_function(Transformed_features);
        outputVertex.setScore(output.at(0).to_float());
      } else {
        std::vector<float> Transformed_features = {float(selectedTracksWithTruth[i].first.pt * ptRescaling),
                                                   float(selectedTracksWithTruth[j].first.pt * ptRescaling),
                                                   selectedTracksWithTruth[i].first.eta,
                                                   selectedTracksWithTruth[j].first.eta,
                                                   selectedTracksWithTruth[i].first.phi,
                                                   selectedTracksWithTruth[j].first.phi,
                                                   selectedTracksWithTruth[i].first.d0,
                                                   selectedTracksWithTruth[j].first.d0,
                                                   selectedTracksWithTruth[i].first.z0,
                                                   selectedTracksWithTruth[j].first.z0,
                                                   float(selectedTracksWithTruth[i].first.chi2rz),
                                                   float(selectedTracksWithTruth[j].first.chi2rz),
                                                   float(selectedTracksWithTruth[i].first.bendchi2),
                                                   float(selectedTracksWithTruth[j].first.bendchi2),
                                                   float(selectedTracksWithTruth[i].first.MVA1),
                                                   float(selectedTracksWithTruth[j].first.MVA1),
                                                   vertex.d_T,
                                                   vertex.R_T,
                                                   vertex.cos_T,
                                                   float(vertex.delta_z * deltaZRescaling)};
        conifer::BDT<float, float> bdt(this->model_.fullPath());
        std::vector<float> output = bdt.decision_function(Transformed_features);
        outputVertex.setScore(output.at(0));
      }

      product->emplace_back(outputVertex);
    }
  }

  // //=== Store output
  iEvent.put(std::move(product), outputVertexCollectionName_);
}

DEFINE_FWK_MODULE(DisplacedVertexProducer);

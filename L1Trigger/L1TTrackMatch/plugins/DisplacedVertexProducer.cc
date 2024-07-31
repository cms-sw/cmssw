#include "L1Trigger/L1TTrackMatch/interface/DisplacedVertexProducer.h"

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
      outputTrackCollectionName_(iConfig.getParameter<std::string>("l1TrackVertexCollectionName")),
      ONNXmodel_(iConfig.getParameter<std::string>("ONNXmodel")),
      ONNXInputName_(iConfig.getParameter<std::string>("ONNXInputName")),
      cutSet_(iConfig.getParameter<edm::ParameterSet>("cutSet")),
      chi2rzMax_(cutSet_.getParameter<double>("chi2rzMax")),
      dispMVAMin_(cutSet_.getParameter<double>("dispMVAMin")),
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
      barrelD0Min_(cutSet_.getParameter<double>("barrelD0Min")) {
  //--- Define EDM output to be written to file (if required)
  produces<l1t::DisplacedTrackVertexCollection>(outputTrackCollectionName_);
  runTime_ = std::make_unique<cms::Ort::ONNXRuntime>(this->ONNXmodel_);
}

void DisplacedVertexProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_>> MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_>>> TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  std::vector<TTTrack<Ref_Phase2TrackerDigi_>>::const_iterator iterL1Track;
  int this_l1track = 0;
  std::vector<std::pair<Track_Parameters, edm::Ptr<TrackingParticle>>> selectedTracksWithTruth;

  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> l1track_ptr(TTTrackHandle, this_l1track);
    this_l1track++;

    float pt = l1track_ptr->momentum().perp();
    float eta = l1track_ptr->momentum().eta();
    float phi = l1track_ptr->momentum().phi();
    float z0 = l1track_ptr->z0();  //cm
    float x0 = l1track_ptr->POCA().x();
    float y0 = l1track_ptr->POCA().y();
    float d0 = -x0 * sin(phi) + y0 * cos(phi);
    float rinv = l1track_ptr->rInv();
    float chi2 = l1track_ptr->chi2Red();
    float chi2rphi = l1track_ptr->chi2XYRed();
    float chi2rz = l1track_ptr->chi2ZRed();
    float bendchi2 = l1track_ptr->stubPtConsistency();
    float MVA1 = l1track_ptr->trkMVA1();
    float MVA2 = l1track_ptr->trkMVA2();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        stubRefs = l1track_ptr->getStubRefs();
    int nstub = (int)stubRefs.size();
    if (chi2rz < chi2rzMax_ && MVA2 > dispMVAMin_ && MVA1 > promptMVAMin_ && pt > ptMin_ && fabs(eta) < etaMax_) {
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

      Track_Parameters track = Track_Parameters(pt,
                                                -d0,
                                                z0,
                                                eta,
                                                phi,
                                                -99999,
                                                -999,
                                                -999,
                                                -999,
                                                rinv,
                                                (this_l1track - 1),
                                                nullptr,
                                                nstub,
                                                chi2rphi,
                                                chi2rz,
                                                bendchi2,
                                                MVA1,
                                                MVA2);

      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);
      selectedTracksWithTruth.push_back(std::make_pair(track, my_tp));
    }
  }

  sort(selectedTracksWithTruth.begin(), selectedTracksWithTruth.end(), ComparePtTrack);
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
      l1t::DisplacedTrackVertex outputVertex = l1t::DisplacedTrackVertex(selectedTracksWithTruth[i].first.index,
                                                                         selectedTracksWithTruth[j].first.index,
                                                                         i,
                                                                         j,
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
                                                                         fabs(i - j),
                                                                         isReal);
      std::vector<std::string> ortinput_names;
      std::vector<std::string> ortoutput_names;
      cms::Ort::FloatArrays ortinput;
      cms::Ort::FloatArrays ortoutputs;
      float minD0 = vertex.a.d0;
      if (fabs(vertex.b.d0) < fabs(minD0))
        minD0 = vertex.b.d0;
      std::vector<float> Transformed_features = {selectedTracksWithTruth[i].first.pt,
                                                 selectedTracksWithTruth[j].first.pt,
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
                                                 selectedTracksWithTruth[i].first.MVA2,
                                                 selectedTracksWithTruth[j].first.MVA2,
                                                 vertex.d_T,
                                                 vertex.R_T,
                                                 vertex.cos_T,
                                                 vertex.delta_z};
      ortinput_names.push_back(this->ONNXInputName_);
      ortoutput_names = runTime_->getOutputNames();
      ortinput.push_back(Transformed_features);
      int batch_size = 1;
      ortoutputs = runTime_->run(ortinput_names, ortinput, {}, ortoutput_names, batch_size);
      outputVertex.setScore(ortoutputs[1][1]);
      product->emplace_back(outputVertex);
    }
  }
  // //=== Store output

  iEvent.put(std::move(product), outputTrackCollectionName_);
}

DEFINE_FWK_MODULE(DisplacedVertexProducer);

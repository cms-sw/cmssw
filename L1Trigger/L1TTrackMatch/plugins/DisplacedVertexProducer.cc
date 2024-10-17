#include "L1Trigger/L1TTrackMatch/interface/DisplacedVertexProducer.h"
#include "L1TrackUnpacker.h"

using namespace l1trackunpacker;

double DisplacedVertexProducer::FloatPtFromBits(const L1TTTrackType &track) const {
  ap_uint<14> ptEmulationBits = track.getTrackWord()(TTTrack_TrackWord::TrackBitLocations::kRinvMSB - 1, TTTrack_TrackWord::TrackBitLocations::kRinvLSB);
  ap_ufixed<14, 9> ptEmulation;
  ptEmulation.V = (ptEmulationBits.range());
  return ptEmulation.to_double();
}

double DisplacedVertexProducer::FloatEtaFromBits(const L1TTTrackType &track) const {
  TTTrack_TrackWord::tanl_t etaBits = track.getTanlWord();
  glbeta_intern digieta;
  digieta.V = etaBits.range();
  return (double)digieta;
}

double DisplacedVertexProducer::FloatPhiFromBits(const L1TTTrackType &track) const {
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

double DisplacedVertexProducer::FloatZ0FromBits(const L1TTTrackType &track) const {
  z0_intern trkZ = track.getZ0Word();
  return BitToDouble(trkZ, TTTrack_TrackWord::TrackBitWidths::kZ0Size, TTTrack_TrackWord::stepZ0);
}

double DisplacedVertexProducer::FloatD0FromBits(const L1TTTrackType &track) const {
  d0_intern trkD0 = track.getD0Word();
  return BitToDouble(trkD0, TTTrack_TrackWord::TrackBitWidths::kD0Size, TTTrack_TrackWord::stepD0);
}

int DisplacedVertexProducer::ChargeFromBits(const L1TTTrackType &track) const {
  ap_uint<1> chargeBit = track.getTrackWord()[TTTrack_TrackWord::TrackBitLocations::kRinvMSB];
  return 1 - (2*chargeBit.to_uint());
}

double convertPtToR(double pt){
  return 100.0 * (1.0 / (0.3 * 3.8)) * pt; //returns R in cm
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
      outputTrackCollectionName_(iConfig.getParameter<std::string>("l1TrackVertexCollectionName")),
      outputTrackEmulationCollectionName_(iConfig.getParameter<std::string>("l1TrackEmulationVertexCollectionName")),
      model_(iConfig.getParameter<std::string>("model")),
      runEmulation_(iConfig.getParameter<bool>("runEmulation")),
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
      barrelD0Min_(cutSet_.getParameter<double>("barrelD0Min")),
      RTMin_(cutSet_.getParameter<double>("RTMin")),
      RTMax_(cutSet_.getParameter<double>("RTMax")) {
  //--- Define EDM output to be written to file (if required)
  produces<l1t::DisplacedTrackVertexCollection>(outputTrackCollectionName_);
  if(runEmulation_) produces<l1t::DisplacedTrackVertexCollection>(outputTrackEmulationCollectionName_);
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
  std::vector<std::pair<Track_Parameters, edm::Ptr<TrackingParticle>>> selectedTracksEmulationWithTruth;

  //Simulation track selection loop
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
    
    //float MVA2 = l1track_ptr->trkMVA2();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        stubRefs = l1track_ptr->getStubRefs();
    int nstub = (int)stubRefs.size();
    //std::cout<<"simulation track pt: "<<pt<<" eta: "<<eta<<" phi: "<<phi<<" z0: "<<z0<<" d0: "<<d0<<" rinv: "<<rinv<<" chi2rphi: "<<chi2rphi<<" chi2rz: "<<chi2rz<<" bendchi2: "<<bendchi2<<" MVA: "<<MVA1<<" nstub: "<<nstub<<" rho: "<<(1/rinv)<<std::endl;
    
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

      Track_Parameters track = Track_Parameters(pt,
                                                -d0,
                                                z0,
                                                eta,
                                                phi,
                                                -99999,
                                                -999,
                                                -999,
                                                -999,
                                                (1/rinv),
                                                (this_l1track - 1),
                                                nullptr,
                                                nstub,
                                                chi2rphi,
                                                chi2rz,
                                                bendchi2,
                                                MVA1);

      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);
      selectedTracksWithTruth.push_back(std::make_pair(track, my_tp));
    }
  }

  //Emulation track selection loop
  this_l1track = 0;
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> l1track_ptr(TTTrackGTTHandle, this_l1track);
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_>> l1track_ref(TTTrackHandle, this_l1track);
    this_l1track++;

#if 0
    float ptSim = l1track_ptr->momentum().perp();
    float etaSim = l1track_ptr->momentum().eta();
    float phiSim = l1track_ptr->momentum().phi();
    float z0Sim = l1track_ptr->z0();  //cm
    float x0Sim = l1track_ptr->POCA().x();
    float y0Sim = l1track_ptr->POCA().y();
    float d0Sim = -x0Sim * sin(phiSim) + y0Sim * cos(phiSim);
    float d0SimAlt = l1track_ptr->d0();
    int chargeSim = (int)(l1track_ptr->rInv() / fabs(l1track_ptr->rInv()));
    float chi2Sim = l1track_ptr->chi2Red();
    float chi2rphiSim = l1track_ptr->chi2XYRed();
    float chi2rzSim = l1track_ptr->chi2ZRed();
    float bendchi2Sim = l1track_ptr->stubPtConsistency();
    float MVA1Sim = l1track_ptr->trkMVA1();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
        stubRefsSim = l1track_ptr->getStubRefs();
    int nstubSim = (int)stubRefsSim.size();

    //std::cout<<"simulation track pt: "<<ptSim<<" eta: "<<etaSim<<" phi: "<<phiSim<<" z0: "<<z0Sim<<" d0: "<<d0Sim<<" charge: "<<chargeSim<<" chi2rphi: "<<chi2rphiSim<<" chi2rz: "<<chi2rzSim<<" bendchi2: "<<bendchi2Sim<<" MVA: "<<MVA1Sim<<" nstub: "<<nstubSim<<" d0 alt: "<<d0SimAlt<<std::endl;
#endif
    
    float pt = FloatPtFromBits(*l1track_ptr);
    float eta = FloatEtaFromBits(*l1track_ptr);
    float phi = FloatPhiFromBits(*l1track_ptr);
    //float z0 = FloatZ0FromBits(*l1track_ptr); //cm
    float z0 = l1track_ptr->getZ0(); //cm
    float d0 = l1track_ptr->getD0();
    int charge = ChargeFromBits(*l1track_ptr);
    float rho = charge*convertPtToR(pt);
    float chi2rphi = l1track_ptr->getChi2RPhi();
    float chi2rz = l1track_ptr->getChi2RZ();
    float bendchi2 = l1track_ptr->getBendChi2();
    float MVA1 = l1track_ptr->getMVAQuality();
    //float MVA2 = l1track_ptr->trkMVA2();
    //std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_>>, TTStub<Ref_Phase2TrackerDigi_>>>
    //stubRefs = l1track_ptr->getStubRefs();
    int nstub = l1track_ptr->getNStubs();

    //std::cout<<"emulation track pt: "<<pt<<" eta: "<<eta<<" phi: "<<phi<<" z0: "<<z0<<" d0: "<<d0<<" charge: "<<charge<<" chi2rphi: "<<chi2rphi<<" chi2rz: "<<chi2rz<<" bendchi2: "<<bendchi2<<" MVA: "<<MVA1<<" nstub: "<<nstub<<" rho: "<<rho<<std::endl;
    
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

      Track_Parameters track = Track_Parameters(pt,
                                                -d0,
                                                z0,
                                                eta,
                                                phi,
                                                -99999,
                                                -999,
                                                -999,
                                                -999,
                                                rho,
                                                (this_l1track - 1),
                                                nullptr,
                                                nstub,
                                                chi2rphi,
                                                chi2rz,
                                                bendchi2,
                                                MVA1);

      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ref);
      selectedTracksEmulationWithTruth.push_back(std::make_pair(track, my_tp));
    }
  }

  
  //int numIntersections = 0;
  //int numVertices = 0;
  //std::cout<<"simulated number of tracks passing cuts: "<<selectedTracksWithTruth.size()<<std::endl;
  //Simulation vertex loop
  std::unique_ptr<l1t::DisplacedTrackVertexCollection> product(new std::vector<l1t::DisplacedTrackVertex>());
  for (int i = 0; i < int(selectedTracksWithTruth.size() - 1); i++) {
    for (int j = i + 1; j < int(selectedTracksWithTruth.size()); j++) {
      if (dist_TPs(selectedTracksWithTruth[i].first, selectedTracksWithTruth[j].first) != 0)
        continue;
      //numIntersections++;
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

      if(vertex.R_T>RTMax_) continue;
      if(vertex.R_T<RTMin_) continue;
      
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
                                                 vertex.d_T,
                                                 vertex.R_T,
                                                 vertex.cos_T,
                                                 vertex.delta_z};
 
      //std::vector<float> TransformedFloat_features;
      //for(auto feature: Transformed_features){
      //TransformedFloat_features.push_back(feature);
      //}
      
      if(isReal){
	std::cout<<"simulation features: "<<std::endl;
	for(auto feature: Transformed_features){
	  std::cout<<feature<<" ";
	}
	std::cout<<std::endl;
      }


      
      conifer::BDT<float, float> bdt(this->model_);
      std::vector<float> output = bdt.decision_function(Transformed_features);
      //outputVertex.setScore(1. / (1. + exp(-output.at(0))));
      outputVertex.setScore(1. / (1. + exp(-output.at(0))));
      if(isReal) std::cout<<"vertex true position: "<<tp_i->vx()<<" "<<tp_i->vy()<<" "<<tp_i->vz()<<" score: "<<(1. / (1. + exp(-output.at(0))))<<" raw score: "<<output.at(0)<<std::endl;
      product->emplace_back(outputVertex);
      //numVertices++;
    }
  }
  //std::cout<<"simulation number of intersections: "<<numIntersections<<" number of vertices: "<<numVertices<<std::endl;
  // //=== Store output
  iEvent.put(std::move(product), outputTrackCollectionName_);
  if(!runEmulation_) return;
  
  //std::cout<<"emulated number of tracks passing cuts: "<<selectedTracksEmulationWithTruth.size()<<std::endl;
  //int numIntersectionsEmu = 0;
  //int numVerticesEmu = 0;
  //Emulation vertex loop
  std::unique_ptr<l1t::DisplacedTrackVertexCollection> productEmulation(new std::vector<l1t::DisplacedTrackVertex>());
  for (int i = 0; i < int(selectedTracksEmulationWithTruth.size() - 1); i++) {
    for (int j = i + 1; j < int(selectedTracksEmulationWithTruth.size()); j++) {
      if (dist_TPs(selectedTracksEmulationWithTruth[i].first, selectedTracksEmulationWithTruth[j].first) != 0)
        continue;
      //numIntersectionsEmu++;
      Double_t x_dv_trk = -9999.0;
      Double_t y_dv_trk = -9999.0;
      Double_t z_dv_trk = -9999.0;
      edm::Ptr<TrackingParticle> tp_i = selectedTracksEmulationWithTruth[i].second;
      edm::Ptr<TrackingParticle> tp_j = selectedTracksEmulationWithTruth[j].second;
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
          calcVertex(selectedTracksEmulationWithTruth[i].first, selectedTracksEmulationWithTruth[j].first, x_dv_trk, y_dv_trk, z_dv_trk);
      Vertex_Parameters vertex = Vertex_Parameters(
          x_dv_trk, y_dv_trk, z_dv_trk, selectedTracksEmulationWithTruth[i].first, selectedTracksEmulationWithTruth[j].first);

      if(vertex.R_T>RTMax_) continue;
      if(vertex.R_T<RTMin_) continue;
      
      l1t::DisplacedTrackVertex outputVertex = l1t::DisplacedTrackVertex(selectedTracksEmulationWithTruth[i].first.index,
                                                                         selectedTracksEmulationWithTruth[j].first.index,
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

      float minD0 = vertex.a.d0;
      if (fabs(vertex.b.d0) < fabs(minD0))
        minD0 = vertex.b.d0;
      //std::cout<<"first pt: "<<selectedTracksEmulationWithTruth[i].first.pt<<" first eta: "<<selectedTracksEmulationWithTruth[i].first.eta<<" first phi: "<<selectedTracksEmulationWithTruth[i].first.phi<<" first d0: "<<selectedTracksEmulationWithTruth[i].first.d0<<" first z0: "<<selectedTracksEmulationWithTruth[i].first.z0<<" first chi2rz: "<<selectedTracksEmulationWithTruth[i].first.chi2rz<<" first bendchi2: "<<selectedTracksEmulationWithTruth[i].first.bendchi2<<" first MVA: "<<selectedTracksEmulationWithTruth[i].first.MVA1<<" d_T: "<<vertex.d_T<<" R_T: "<<vertex.R_T<<" cos_T: "<<vertex.cos_T<<" delta_z: "<<vertex.delta_z<<std::endl;
      std::vector<ap_fixed<32,16,AP_RND_CONV, AP_SAT>> Transformed_features = {selectedTracksEmulationWithTruth[i].first.pt,
                                                 selectedTracksEmulationWithTruth[j].first.pt,
                                                 selectedTracksEmulationWithTruth[i].first.eta,
                                                 selectedTracksEmulationWithTruth[j].first.eta,
                                                 selectedTracksEmulationWithTruth[i].first.phi,
                                                 selectedTracksEmulationWithTruth[j].first.phi,
                                                 selectedTracksEmulationWithTruth[i].first.d0,
                                                 selectedTracksEmulationWithTruth[j].first.d0,
                                                 selectedTracksEmulationWithTruth[i].first.z0,
                                                 selectedTracksEmulationWithTruth[j].first.z0,
                                                 selectedTracksEmulationWithTruth[i].first.chi2rz,
                                                 selectedTracksEmulationWithTruth[j].first.chi2rz,
                                                 selectedTracksEmulationWithTruth[i].first.bendchi2,
                                                 selectedTracksEmulationWithTruth[j].first.bendchi2,
                                                 selectedTracksEmulationWithTruth[i].first.MVA1,
                                                 selectedTracksEmulationWithTruth[j].first.MVA1,
                                                 vertex.d_T,
                                                 vertex.R_T,
                                                 vertex.cos_T,
                                                 vertex.delta_z};
      if(isReal){
	std::cout<<"emulation features: "<<std::endl;
	for(auto feature: Transformed_features){
	  std::cout<<feature<<" ";
	}
	std::cout<<std::endl;
      }
      conifer::BDT<ap_fixed<32,16,AP_RND_CONV, AP_SAT>, ap_fixed<32,16,AP_RND_CONV, AP_SAT>, true> bdt(this->model_);
      std::vector<ap_fixed<32,16,AP_RND_CONV, AP_SAT>> output = bdt.decision_function(Transformed_features);
      //std::cout<<"transformed first pt: "<<Transformed_features[0]<<" transformed first eta: "<<Transformed_features[2]<<" transformed first phi: "<<Transformed_features[4]<<" transformed first d0: "<<Transformed_features[6]<<" transformed first z0: "<<Transformed_features[8]<<" transformed first chi2rz: "<<Transformed_features[10]<<" transformed first bendchi2: "<<Transformed_features[12]<<" transformed first MVA: "<<Transformed_features[14]<<" transformed d_T: "<<Transformed_features[16]<<" transformed R_T: "<<Transformed_features[17]<<" transformed cos_T: "<<Transformed_features[18]<<" transformed delta_z: "<<Transformed_features[19]<<std::endl;
      //std::cout<<"transformed second pt: "<<Transformed_features[1]<<" transformed second eta: "<<Transformed_features[3]<<" transformed second phi: "<<Transformed_features[5]<<" transformed second d0: "<<Transformed_features[7]<<" transformed second z0: "<<Transformed_features[9]<<" transformed second chi2rz: "<<Transformed_features[11]<<" transformed second bendchi2: "<<Transformed_features[13]<<" transformed second MVA: "<<Transformed_features[15]<<" transformed d_T: "<<Transformed_features[16]<<" transformed R_T: "<<Transformed_features[17]<<" transformed cos_T: "<<Transformed_features[18]<<" transformed delta_z: "<<Transformed_features[19]<<std::endl;
      outputVertex.setScore(1. / (1. + exp(-output.at(0).to_float())));
      outputVertex.setScoreEmu(output.at(0));
      if(isReal) std::cout<<"emulation vertex true position: "<<tp_i->vx()<<" "<<tp_i->vy()<<" "<<tp_i->vz()<<" score: "<<(1. / (1. + exp(-output.at(0).to_float())))<<" raw score: "<<output.at(0).to_float()<<std::endl;
      productEmulation->emplace_back(outputVertex);
      //numVerticesEmu++;
    }
  }
  //std::cout<<"emulation number of intersections: "<<numIntersectionsEmu<<" number of vertices: "<<numVerticesEmu<<std::endl;
  iEvent.put(std::move(productEmulation), outputTrackEmulationCollectionName_);
}

DEFINE_FWK_MODULE(DisplacedVertexProducer);

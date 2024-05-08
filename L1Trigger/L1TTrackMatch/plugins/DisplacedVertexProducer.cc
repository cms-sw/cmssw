#include "L1Trigger/L1TTrackMatch/interface/DisplacedVertexProducer.h"

bool ComparePtTrack(Track_Parameters a, Track_Parameters b) { return a.pt > b.pt; }

Double_t dist(Double_t x1, Double_t y1 , Double_t x2=0, Double_t y2=0){ // Distance between 2 points
  return (TMath::Sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)));
}

Double_t dist_TPs(Track_Parameters a, Track_Parameters b){
  float x1 = a.x0; //   Centers of the circles
  float y1 = a.y0; //
  float x2 = b.x0; //
  float y2 = b.y0; //
  float R1 = a.rho;// Radii of the circles
  float R2 = b.rho;
  float R = dist(x1,y1,x2,y2); // Distance between centers
  if((R>=(R1-R2)) && (R<=(R1+R2))){
    return (0);
  }
  else if(R==0){
    return (-99999.0);
  }
  else{
    return(R-R1-R2);
  }
}

Int_t calcVertex(Track_Parameters a, Track_Parameters b, Double_t &x_vtx, Double_t &y_vtx, Double_t &z_vtx){
  float x1 = a.x0; //   Centers of the circles
  float y1 = a.y0; //
  float x2 = b.x0; //
  float y2 = b.y0; //
  float R1 = a.rho;   // Radii of the circles
  float R2 = b.rho;
  float R = dist(x1,y1,x2,y2); // Distance between centers
  if(R==0) return -1;
  float co1 = (pow(R1,2)-pow(R2,2))/(2*pow(R,2));
  float radicand = (2/pow(R,2))*(pow(R1,2)+pow(R2,2))-(pow(pow(R1,2)-pow(R2,2),2)/pow(R,4))-1;
  float co2 = 0;
  if(radicand>0) co2 = 0.5*TMath::Sqrt(radicand);
  float ix1_x = 0.5*(x1+x2)+co1*(x2-x1)+co2*(y2-y1);
  float ix2_x = 0.5*(x1+x2)+co1*(x2-x1)-co2*(y2-y1);
  float ix1_y = 0.5*(y1+y2)+co1*(y2-y1)+co2*(x1-x2);
  float ix2_y = 0.5*(y1+y2)+co1*(y2-y1)-co2*(x1-x2);
  float ix1_z1 = a.z(ix1_x,ix1_y);
  float ix1_z2 = b.z(ix1_x,ix1_y);
  float ix1_delz = fabs(ix1_z1-ix1_z2);
  float ix2_z1 = a.z(ix2_x,ix2_y);
  float ix2_z2 = b.z(ix2_x,ix2_y);
  float ix2_delz = fabs(ix2_z1-ix2_z2);
  float trk1_POCA[2] = {a.d0*sin(a.phi),-1*a.d0*cos(a.phi)};
  float trk2_POCA[2] = {b.d0*sin(b.phi),-1*b.d0*cos(b.phi)};
  float trk1_ix1_delxy[2] = {ix1_x-trk1_POCA[0],ix1_y-trk1_POCA[1]};
  float trk1_ix2_delxy[2] = {ix2_x-trk1_POCA[0],ix2_y-trk1_POCA[1]};
  float trk2_ix1_delxy[2] = {ix1_x-trk2_POCA[0],ix1_y-trk2_POCA[1]};
  float trk2_ix2_delxy[2] = {ix2_x-trk2_POCA[0],ix2_y-trk2_POCA[1]};
  float trk1_traj[2] = {cos(a.phi),sin(a.phi)};
  float trk2_traj[2] = {cos(b.phi),sin(b.phi)};
  bool trk1_ix1_inTraj = ((trk1_ix1_delxy[0]*trk1_traj[0]+trk1_ix1_delxy[1]*trk1_traj[1])>0) ? true : false;
  bool trk1_ix2_inTraj = ((trk1_ix2_delxy[0]*trk1_traj[0]+trk1_ix2_delxy[1]*trk1_traj[1])>0) ? true : false;
  bool trk2_ix1_inTraj = ((trk2_ix1_delxy[0]*trk2_traj[0]+trk2_ix1_delxy[1]*trk2_traj[1])>0) ? true : false;
  bool trk2_ix2_inTraj = ((trk2_ix2_delxy[0]*trk2_traj[0]+trk2_ix2_delxy[1]*trk2_traj[1])>0) ? true : false;
  if(trk1_ix1_inTraj&&trk2_ix1_inTraj&&trk1_ix2_inTraj&&trk2_ix2_inTraj){
    if(ix1_delz<ix2_delz){
      x_vtx = ix1_x;
      y_vtx = ix1_y;
      z_vtx = (ix1_z1+ix1_z2)/2;
      return 0;
    }
    else{
      x_vtx = ix2_x;
      y_vtx = ix2_y;
      z_vtx = (ix2_z1+ix2_z2)/2;
      return 0;
    }
  }
  if(trk1_ix1_inTraj&&trk2_ix1_inTraj){
    x_vtx = ix1_x;
    y_vtx = ix1_y;
    z_vtx = (ix1_z1+ix1_z2)/2;
    return 1;
  }
  if(trk1_ix2_inTraj&&trk2_ix2_inTraj){
    x_vtx = ix2_x;
    y_vtx = ix2_y;
    z_vtx = (ix2_z1+ix2_z2)/2;
    return 2;
  }
  else{
    if(ix1_delz<ix2_delz){
      x_vtx = ix1_x;
      y_vtx = ix1_y;
      z_vtx = (ix1_z1+ix1_z2)/2;
      return 3;
    }
    else{
      x_vtx = ix2_x;
      y_vtx = ix2_y;
      z_vtx = (ix2_z1+ix2_z2)/2;
      return 3;
    }
  }
  return 4;
}

DisplacedVertexProducer::DisplacedVertexProducer(const edm::ParameterSet& iConfig)
  : ttTrackMCTruthToken_(consumes<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> >(iConfig.getParameter<edm::InputTag>("mcTruthTrackInputTag"))),
    trackToken_(consumes<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > >(iConfig.getParameter<edm::InputTag>("l1TracksInputTag"))),
    outputTrackCollectionName_(iConfig.getParameter<std::string>("l1TrackVertexCollectionName")),
    ONNXmodel_(iConfig.getParameter<std::string>("ONNXmodel")),
    ONNXInputName_(iConfig.getParameter<std::string>("ONNXInputName")),
    featureNames_(iConfig.getParameter<std::vector<std::string>>("featureNames"))
{
  //--- Define EDM output to be written to file (if required)
  produces<l1t::DisplacedTrackVertexCollection>(outputTrackCollectionName_);
  runTime_ = std::make_unique<cms::Ort::ONNXRuntime>(this->ONNXmodel_);
}

void DisplacedVertexProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<TTTrackAssociationMap<Ref_Phase2TrackerDigi_> > MCTruthTTTrackHandle;
  iEvent.getByToken(ttTrackMCTruthToken_, MCTruthTTTrackHandle);
  edm::Handle<std::vector<TTTrack<Ref_Phase2TrackerDigi_> > > TTTrackHandle;
  iEvent.getByToken(trackToken_, TTTrackHandle);
  std::vector<TTTrack<Ref_Phase2TrackerDigi_> >::const_iterator iterL1Track;
  int this_l1track = 0;
  std::vector<Track_Parameters> selectedTracks;
  std::vector<edm::Ptr<TrackingParticle>> selectedTPs;
  
  for (iterL1Track = TTTrackHandle->begin(); iterL1Track != TTTrackHandle->end(); iterL1Track++) {
    edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> > l1track_ptr(TTTrackHandle, this_l1track);
    this_l1track++;
    float pt = iterL1Track->momentum().perp();
    float eta = iterL1Track->momentum().eta();
    float phi = iterL1Track->momentum().phi();
    float z0 = iterL1Track->z0();  //cm
    float x0 = iterL1Track->POCA().x();
    float y0 = iterL1Track->POCA().y();
    float d0 = -x0 * sin(phi) + y0 * cos(phi);
    float rinv = iterL1Track->rInv();
    float chi2 = iterL1Track->chi2Red();
    float chi2rphi = iterL1Track->chi2XYRed();
    float chi2rz = iterL1Track->chi2ZRed();
    float bendchi2 = iterL1Track->stubPtConsistency();
    float MVA1 = iterL1Track->trkMVA1();
    float MVA2 = iterL1Track->trkMVA2();
    std::vector<edm::Ref<edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> >, TTStub<Ref_Phase2TrackerDigi_> > > stubRefs = iterL1Track->getStubRefs();
    int nstub = (int)stubRefs.size();
    if( chi2rz<3.0 && MVA2>0.2 && MVA1>0.2 && pt>3.0 && fabs(eta)<2.4){

      if(fabs(d0)>1.0){
	if(MVA1<=0.5) continue;
      }
      if(fabs(eta)>1.1 && fabs(eta)<1.7){
	if(nstub<=4) continue;
      }
      if(fabs(eta)>0.95){
	if(fabs(d0)<=0.08) continue;
      }
      if(fabs(eta)<=0.95){
	if(fabs(d0)<=0.06) continue;
      }

      //std::cout<<"track params: "<<pt<<" "<<-d0<<" "<<z0<<" "<<eta<<" "<<phi<<" "<<rinv<<" "<<this_l1track<<" "<<nstub<<" "<<chi2rphi<<" "<<chi2rz<<" "<<bendchi2<<" "<<MVA1<<" "<<MVA2<<std::endl;
      Track_Parameters track = Track_Parameters(pt, -d0, z0, eta, phi, -99999, -999, -999, -999, rinv, (this_l1track - 1), nullptr, nstub, chi2rphi, chi2rz, bendchi2, MVA1, MVA2);
      selectedTracks.push_back(track);
      edm::Ptr<TrackingParticle> my_tp = MCTruthTTTrackHandle->findTrackingParticlePtr(l1track_ptr);
      selectedTPs.push_back(my_tp);
    }
  }
  //std::cout<<"num selected tracks: "<<selectedTracks.size()<<std::endl;
  sort(selectedTracks.begin(), selectedTracks.end(), ComparePtTrack); 
  std::unique_ptr<l1t::DisplacedTrackVertexCollection> product(new std::vector<l1t::DisplacedTrackVertex>());
  for(int i=0; i<int(selectedTracks.size()-1); i++){
    for(int j=i+1; j<int(selectedTracks.size()); j++){
      if(dist_TPs(selectedTracks[i],selectedTracks[j])!=0) continue;
      Double_t x_dv_trk = -9999.0;
      Double_t y_dv_trk = -9999.0;
      Double_t z_dv_trk = -9999.0;
      edm::Ptr<TrackingParticle> tp_i = selectedTPs[i];
      edm::Ptr<TrackingParticle> tp_j = selectedTPs[j];
      bool isReal = false;
      if(!tp_i.isNull() && !tp_j.isNull()){
	bool isHard_i = false;
	bool isHard_j = false;
	if(tp_i->genParticles().size() && tp_j->genParticles().size()){
	  isHard_i = tp_i->genParticles()[0]->isHardProcess() || tp_i->genParticles()[0]->fromHardProcessFinalState();
	  isHard_j = tp_j->genParticles()[0]->isHardProcess() || tp_j->genParticles()[0]->fromHardProcessFinalState();
	}
	
	if(tp_i->eventId().event()==0 && tp_j->eventId().event()==0 && fabs(tp_i->vx()-tp_j->vx())<0.0001 && fabs(tp_i->vy()-tp_j->vy())<0.0001 && fabs(tp_i->vz()-tp_j->vz())<0.0001 && isHard_i && isHard_j){
	  isReal = true;
	}
      }
      
      int inTraj = calcVertex(selectedTracks[i],selectedTracks[j],x_dv_trk,y_dv_trk,z_dv_trk);
      Vertex_Parameters vertex = Vertex_Parameters(x_dv_trk, y_dv_trk, z_dv_trk, selectedTracks[i], selectedTracks[j]);
      l1t::DisplacedTrackVertex outputVertex = l1t::DisplacedTrackVertex(selectedTracks[i].index, selectedTracks[j].index, i, j, inTraj, vertex.d_T, vertex.R_T, vertex.cos_T, vertex.delta_z, vertex.x_dv, vertex.y_dv, vertex.z_dv, vertex.openingAngle, vertex.p_mag, fabs(i-j), isReal);
      std::vector<std::string> ortinput_names;
      std::vector<std::string> ortoutput_names;
      cms::Ort::FloatArrays ortinput;
      cms::Ort::FloatArrays ortoutputs;
      float minD0 = vertex.a.d0;
      if(fabs(vertex.b.d0)<fabs(minD0)) minD0 = vertex.b.d0;
      //std::vector<float> Transformed_features = {vertex.delta_z, vertex.R_T, vertex.cos_T, vertex.d_T, vertex.chi2rzdofSum, float(vertex.numStubsSum), vertex.chi2rphidofSum, minD0, vertex.a.pt+vertex.b.pt};
      std::vector<float> Transformed_features = {selectedTracks[i].pt, selectedTracks[j].pt, selectedTracks[i].eta, selectedTracks[j].eta, selectedTracks[i].phi, selectedTracks[j].phi, selectedTracks[i].d0, selectedTracks[j].d0, selectedTracks[i].z0, selectedTracks[j].z0, selectedTracks[i].chi2rz, selectedTracks[j].chi2rz, selectedTracks[i].bendchi2, selectedTracks[j].bendchi2, selectedTracks[i].MVA1, selectedTracks[j].MVA1, selectedTracks[i].MVA2, selectedTracks[j].MVA2, vertex.d_T, vertex.R_T, vertex.cos_T, vertex.delta_z};
      //cms::Ort::ONNXRuntime Runtime(this->ONNXmodel_);  //Setup ONNX runtime
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

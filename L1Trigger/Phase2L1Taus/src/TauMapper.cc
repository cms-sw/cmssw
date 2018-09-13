#include "L1Trigger/Phase2L1Taus/interface/TauMapper.h"


bool TauMapper::addPFChargedHadron( l1t::PFCandidate in ){
  bool ChargedHadronAdded = false;
  if(!seedHadronSet && contains(in)){
    setSeedChargedHadron(in);
    return true;
  }

  //change deltaR to contains function
  if(seedHadronSet && seedCHConeContains(in)){
    if(in.pt() > prong3.pt() && in.pt() < prong2.pt()){
      prong3 = in;
      ChargedHadronAdded = true;
    }
    else if(in.pt() > prong2.pt()){
      prong2 = in;
      ChargedHadronAdded = true;
    }
  }
  else if(seedHadronSet && isolationConeContains(in)){

    sumChargedIso += in.pt();

    ChargedHadronAdded = true;
  }

  return ChargedHadronAdded;

}

bool TauMapper::seedCHConeContains( l1t::PFCandidate in ){
  if(!seedHadronSet){
    std::cout<<"ERROR TRYING TO CHECK SEED CH CONE CONTAINS SECOND CAND BUT SEED CONE NOT SET"<<std::endl;
    exit(0);
  }

  float pfCharged_pt = in.pt();
  float pfCharged_eta = in.eta();
  float pfCharged_phi = in.phi();

  //float seedCH_pt = seedCH.pt();
  float seedCH_eta = seedCH.eta();
  float seedCH_phi = seedCH.phi();

  float deltaR = 0.2;
  if(pfCharged_pt>3){
    deltaR = 0.18;
  }
  if(pfCharged_pt>5){
    deltaR = 0.17;
  }
  if(pfCharged_pt>10){
    deltaR = 0.1;
  }
  if(pfCharged_pt>18){
    deltaR = 0.06;
  }
  if(pfCharged_pt>25){
    deltaR = 0.03;
  }
  
  if(fabs(pfCharged_eta-seedCH_eta)+fabs(pfCharged_phi-seedCH_phi) < deltaR)
    return true;

  return false;

}


bool TauMapper::contains(l1t::PFCandidate in){

  float pfCharged_eta = in.eta();
  float pfCharged_phi = in.phi();

  //must be within defined Tau Seed HW area
  if((fabs(pfCharged_eta - l1PFTau.hwEta()) < tau_size_eta/2) && (fabs(pfCharged_phi - l1PFTau.hwPhi()) < tau_size_phi/2) )
    return true;

  return false;

}

bool TauMapper::isolationConeContains( l1t::PFCandidate in ){

  if(!seedHadronSet){
    std::cout<<"ERROR TRYING TO CHECK SEED CH CONE CONTAINS SECOND CAND BUT SEED CONE NOT SET"<<std::endl;
    exit(0);
  }

  //float pfCharged_pt = in.pt();
  float pfCharged_eta = in.eta();
  float pfCharged_phi = in.phi();

  //float seedCH_pt = seedCH.pt();
  float seedCH_eta = seedCH.eta();
  float seedCH_phi = seedCH.phi();

  float deltaR = 0.4;


  if(fabs(pfCharged_eta-seedCH_eta)+fabs(pfCharged_phi-seedCH_phi) < deltaR)
    return true;

  return false;
  
}

void TauMapper::buildStripGrid(){
  if(!seedHadronSet){
    std::cout<<"ERROR SEED CONE NOT SET"<<std::endl;
    exit(0);
  }
  for( int i = -2; i<3; i++){
    for( int j = -2; j<3; j++){
      simple_object_t temp;
      temp.et = 0;
      temp.eta = seedCH.eta() + i*tower_size;
      temp.phi = seedCH.phi() + j*tower_size;

      if(temp.phi > 3.14159)
	temp.phi = -1*(2*3.14159-temp.phi);

      if(temp.phi < -3.14159)
	temp.phi =  (2*3.14159-temp.phi);
      
      temp.phi = round_to_tower(temp.phi);
      temp.eta = round_to_tower(temp.eta);

      egGrid[i+2][j+2] = temp;
    }
  }
}

bool TauMapper::addEG( l1t::PFCandidate in ){

  for(int i = 0; i<5; i++){
    for(int j = 0; j<5; j++){
      float temp_in_eta = round_to_tower(in.eta());
      float temp_in_phi = round_to_tower(in.phi());
      if(temp_in_eta == egGrid[i][j].eta &&
	 temp_in_phi == egGrid[i][j].phi ){
	egGrid[i][j].et += in.et();
	return true;
      }
    }
  }
  return false;
}


void TauMapper::process(){

  process_strip();

  //std::cout<<"seedCH pt: "<< seedCH.pt()<<" eta: "<< seedCH.eta() <<" phi: "<<seedCH.phi()<<std::endl;
  //std::cout<<"   prong2 pt: "<< prong2.pt()<<" eta: "<< prong2.eta() <<" phi: "<<prong2.phi()<<std::endl;
  //std::cout<<"   prong3 pt: "<< prong3.pt()<<" eta: "<< prong3.eta() <<" phi: "<<prong3.phi()<<std::endl;
  //std::cout<<"   strip  pt: "<< strip_pt <<" eta: "<< strip_eta <<std::endl;

  // 3 prong
  if(prong2.pt() > 0 && prong3.pt() > 0){
    float pt = prong2.pt()+prong3.pt()+seedCH.pt();
    float eta = (prong2.eta()+prong3.eta()+seedCH.eta())/3;
    float phi = 0;
    if((seedCH.phi()>0 && prong2.phi()>0 && prong3.phi()>0)||
       (seedCH.phi()<0 && prong2.phi()<0 && prong3.phi()<0))
      phi = (prong2.phi()+prong3.phi()+seedCH.phi())/3;
    else 
      phi = seedCH.phi();

    math::PtEtaPhiMLorentzVector tempP4(pt,eta,phi,pt);
    l1PFTau.setP4(tempP4);

    l1PFTau.setChargedIso(sumChargedIso);
    l1PFTau.setTauType(10);    
  }

  //1 prong
  if(prong3.pt() == 0 && strip_pt == 0 ){
    float pt = seedCH.pt();
    float eta = seedCH.eta();
    float phi = seedCH.phi();
    math::PtEtaPhiMLorentzVector tempP4(pt,eta,phi,pt);

    l1PFTau.setP4(tempP4);

    l1PFTau.setChargedIso(sumChargedIso+prong2.pt());

    l1PFTau.setTauType(0);    
  }

  // 1 prong pi0
  if(prong3.pt() == 0 && strip_pt != 0 ){
    float pt = seedCH.pt() + strip_pt;
    float eta = (seedCH.eta() + strip_eta)/2;
    float phi = seedCH.phi();
    math::PtEtaPhiMLorentzVector tempP4(pt,eta,phi,pt);

    l1PFTau.setP4(tempP4);

    l1PFTau.setChargedIso(sumChargedIso+prong2.pt());

    l1PFTau.setTauType(1);    
  }

}

void TauMapper::process_strip(){

  simple_object_t temp_strip[5];    
  simple_object_t final_strip;

  for(int i = 0; i<4; i++){
    for(int j = 0; j<5; j++){
      unsigned int ip = i+1;
      merge_strip(egGrid[i][j], egGrid[ip][j], temp_strip[j]);
    }
  }

  final_strip = temp_strip[0];

  for(unsigned int j = 1; j < 5; j++){
    
    //first check if strip j is greater than final strip
    if(temp_strip[j].et > final_strip.et){
      final_strip = temp_strip[j];
    }

    strip_pt  = final_strip.et;
    strip_eta = final_strip.eta;

  }
}


void TauMapper::merge_strip(simple_object_t cluster_1, simple_object_t cluster_2, simple_object_t &strip){
  if(delta_r_cluster(cluster_1, cluster_2) < 2.5 //dummy value
     && strip.et < (cluster_1.et + cluster_2.et )){

    strip.et  = cluster_1.et + cluster_2.et;
    strip.eta = weighted_avg_eta(cluster_1, cluster_2);
    strip.phi = weighted_avg_phi(cluster_1, cluster_2);
  } //if clusters ddon't merge then take the higher pt cluster if it is greater than current strip pt and it is pi0 like
  else if(strip.et < cluster_1.et && cluster_1.et > 0 && cluster_1.et > cluster_2.et){
    strip.et  = cluster_1.et;
    strip.eta = cluster_1.eta;
    strip.phi = cluster_1.phi;
  }    
  else if(strip.et < cluster_2.et && cluster_2.et>0 && cluster_2.et > cluster_1.et){
    strip.et  = cluster_2.et;
    strip.eta = cluster_2.eta;
    strip.phi = cluster_2.phi;
  }   
}

float TauMapper::delta_r_cluster(simple_object_t cluster_1, simple_object_t cluster_2){
  float delta_r = 20;
  delta_r = fabs(cluster_1.eta - cluster_2.eta) + fabs(cluster_1.phi - cluster_2.phi);

  return delta_r;
}

float TauMapper::weighted_avg_phi(simple_object_t cluster_1, simple_object_t cluster_2){
  float total_pt = (cluster_1.et+cluster_2.et);
  float avg_phi = (cluster_1.phi*cluster_1.et + cluster_2.phi*cluster_2.et)/total_pt;
  return avg_phi;
}

float TauMapper::weighted_avg_eta(simple_object_t cluster_1, simple_object_t cluster_2){
  float total_pt = (cluster_1.et+cluster_2.et);
  float avg_eta = (cluster_1.eta*cluster_1.et + cluster_2.eta*cluster_2.et)/total_pt;
  return avg_eta;
}


TauMapper::TauMapper() {};
TauMapper::~TauMapper() {};

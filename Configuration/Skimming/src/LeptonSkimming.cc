
// -*- C++ -*-
//
// Package:    SkimmingForB/LeptonSkimming
// Class:      LeptonSkimming
// 
/**\class LeptonSkimming LeptonSkimming.cc SkimmingForB/LeptonSkimming/plugins/LeptonSkimming.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Georgios Karathanasis georgios.karathanasis@cern.ch
//         Created:  Thu, 29 Nov 2018 15:23:09 GMT
//
//


#include "Configuration/Skimming/interface/LeptonSkimming.h"


using namespace edm;
using namespace reco;
using namespace std;

LeptonSkimming::LeptonSkimming(const edm::ParameterSet& iConfig):
  electronsToken_(consumes<edm::View<reco::GsfElectron> >(iConfig.getParameter<edm::InputTag>  ("electrons"))),
  muonsToken_(consumes<std::vector<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
  Tracks_(consumes<std::vector<reco::Track> >(iConfig.getParameter<edm::InputTag>("tracks"))),
       vtxToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
  beamSpotToken_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpot"))),
  conversionsToken_(consumes< reco::ConversionCollection > (iConfig.getParameter<edm::InputTag> ("conversions"))),

   trgresultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag> ("triggerresults"))),
  trigobjectsToken_(consumes<trigger::TriggerEvent>(iConfig.getParameter<edm::InputTag> ("triggerobjects"))),
  HLTFilter_(iConfig.getParameter<vector<string> >("HLTFilter")),
  HLTPath_(iConfig.getParameter<vector<string> >("HLTPath"))
  {
  edm::ParameterSet runParameters=iConfig.getParameter<edm::ParameterSet>("RunParameters");     
 PtTrack_Cut=runParameters.getParameter<double>("PtTrack_Cut");
 EtaTrack_Cut=runParameters.getParameter<double>("EtaTrack_Cut");
 //cout<<PtTrack_Cut<<endl;
 MinChi2Track_Cut=runParameters.getParameter<double>("MinChi2Track_Cut");
 MaxChi2Track_Cut=runParameters.getParameter<double>("MaxChi2Track_Cut");
 MuTrkMinDR_Cut=runParameters.getParameter<double>("MuTrkMinDR_Cut");
 MaxMee_Cut=runParameters.getParameter<double>("MaxMee_Cut");
 MinMee_Cut=runParameters.getParameter<double>("MinMee_Cut");
 Probee_Cut=runParameters.getParameter<double>("Probee_Cut");
 Cosee_Cut=runParameters.getParameter<double>("Cosee_Cut");
 
 PtKTrack_Cut=runParameters.getParameter<double>("PtKTrack_Cut");
 MaxMB_Cut=runParameters.getParameter<double>("MaxMB_Cut");
 MinMB_Cut=runParameters.getParameter<double>("MinMB_Cut");
 TrkTrkMinDR_Cut=runParameters.getParameter<double>("TrkTrkMinDR_Cut");

 TrackSdxy_Cut=runParameters.getParameter<double>("TrackSdxy_Cut");
 
 MuTrgMatchCone=runParameters.getParameter<double>("MuTrgMatchCone");
 SkipIfNoMuMatch=runParameters.getParameter<bool>("SkipIfNoMuMatch");
 EpairZvtx_Cut=runParameters.getParameter<double>("EpairZvtx_Cut");
 Ksdxy_Cut=runParameters.getParameter<double>("Ksdxy_Cut");
 ProbeeK_Cut=runParameters.getParameter<double>("ProbeeK_Cut");
 CoseeK_Cut=runParameters.getParameter<double>("CoseeK_Cut");
 TrackMuDz_Cut=runParameters.getParameter<double>("TrackMuDz_Cut");
 MaxMVA_Cut=runParameters.getParameter<double>("MaxMVA_Cut");
 MinMVA_Cut=runParameters.getParameter<double>("MinMVA_Cut");
 TrgExclusionCone=runParameters.getParameter<double>("TrgExclusionCone");
 SLxy_Cut=runParameters.getParameter<double>("SLxy_Cut");
 PtB_Cut=runParameters.getParameter<double>("PtB_Cut");
 PtMu_Cut=runParameters.getParameter<double>("PtMu_Cut");
  PtEl_Cut=runParameters.getParameter<double>("PtEl_Cut");
 QualMu_Cut=runParameters.getParameter<double>("QualMu_Cut");
 MuTrgExclusionCone=runParameters.getParameter<double>("MuTrgExclusionCone");
 ElTrgExclusionCone=runParameters.getParameter<double>("ElTrgExclusionCone");
 TrkObjExclusionCone=runParameters.getParameter<double>("TrkObjExclusionCone");
 MuTrgMuDz_Cut=runParameters.getParameter<double>("MuTrgMuDz_Cut");
 ElTrgMuDz_Cut=runParameters.getParameter<double>("ElTrgMuDz_Cut");
 ObjPtLargerThanTrack=runParameters.getParameter<bool>("ObjPtLargerThanTrack");



}


LeptonSkimming::~LeptonSkimming()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}

std::pair<std::vector<float>,std::vector<std::vector<std::vector<float>>>> LeptonSkimming::HLTAnalyze(const edm::Event& iEvent, const edm::EventSetup& iSetup,std::vector< string> HLTPath,std::vector< string> Seed ){
   using namespace std;  using namespace edm;  using namespace reco;
  using namespace trigger;
 
  edm::Handle<trigger::TriggerEvent> triggerObjectsSummary;
  iEvent.getByToken(trigobjectsToken_ ,triggerObjectsSummary);
  edm::Handle<edm::TriggerResults> trigResults;
  iEvent.getByToken(trgresultsToken_, trigResults);
  trigger::TriggerObjectCollection selectedObjects;
  std::vector<float> fires;
  std::vector<std::vector<std::vector<float>>> trg_event; 

for (unsigned int ipath=0; ipath<Seed.size(); ipath++){ 
  std::vector<std::vector<float>> tot_tr_obj_pt_eta_phi;
  if (triggerObjectsSummary.isValid()) {  
      size_t filterIndex = (*triggerObjectsSummary).filterIndex(InputTag(Seed[ipath],"","HLT"));
      trigger::TriggerObjectCollection allTriggerObjects = triggerObjectsSummary->getObjects();     
      if (filterIndex < (*triggerObjectsSummary).sizeFilters()) { 
      const trigger::Keys &keys = (*triggerObjectsSummary).filterKeys(filterIndex);
      for (size_t j = 0; j < keys.size(); j++) {
	trigger::TriggerObject foundObject = (allTriggerObjects)[keys[j]];
        std::vector<float> tr_obj_pt_eta_phi;
        if (fabs(foundObject.id())!=13) continue;
        tr_obj_pt_eta_phi.push_back(foundObject.pt());
        tr_obj_pt_eta_phi.push_back(foundObject.eta());
        tr_obj_pt_eta_phi.push_back(foundObject.phi());
        tr_obj_pt_eta_phi.push_back(foundObject.id()/fabs(foundObject.id()));
        tot_tr_obj_pt_eta_phi.push_back( tr_obj_pt_eta_phi);
      }
       
       }   
    }    
     trg_event.push_back(tot_tr_obj_pt_eta_phi);
  }
  //paths
  float fire0=0,fire1=0,fire2=0,fire3=0,fire4=0,fire5=0;
    if( !trigResults.failedToGet() ) {
    int N_Triggers = trigResults->size();
    const edm::TriggerNames & trigName = iEvent.triggerNames(*trigResults);    
    //cout << "new" << endl;
    for( int i_Trig = 0; i_Trig < N_Triggers; ++i_Trig ) {
      TString TrigPath =trigName.triggerName(i_Trig);    
      if (TrigPath.Contains(HLTPath[0]) && trigResults->accept(i_Trig)){
          fire0=1;}
      if(TrigPath.Contains(HLTPath[1])  && trigResults->accept(i_Trig) ){
          fire1=1;}
      if (TrigPath.Contains(HLTPath[2]) && trigResults->accept(i_Trig)){ 
          fire2=1;}
      if (TrigPath.Contains(HLTPath[3]) && trigResults->accept(i_Trig)){
	  fire3=1;}
      if (TrigPath.Contains(HLTPath[4]) && trigResults->accept(i_Trig)){
          fire4=1;}
      if (TrigPath.Contains(HLTPath[5]) && trigResults->accept(i_Trig)){ 
	  fire5=1;}
       } 
      }
								   	
    fires.push_back(fire0);  fires.push_back(fire1);   fires.push_back(fire2);
    fires.push_back(fire3); fires.push_back(fire4); fires.push_back(fire5);
    return std::make_pair(fires,trg_event);
}




std::vector<float> LeptonSkimming::SelectTrg_Object(std::vector<std::vector<float>> &tr1,std::vector<std::vector<float>> &tr2, std::vector<std::vector<float>> &tr3,std::vector<std::vector<float>> &tr4,std::vector<std::vector<float>> &tr5,std::vector<std::vector<float>> &tr6){
  //  std::vector<float> kolos; kolos.push_back(1);
  
  std::vector<std::vector<float>> max1;
  for (auto & vec: tr1) max1.push_back(vec);
  for (auto & vec: tr2) max1.push_back(vec);
  for (auto & vec: tr3) max1.push_back(vec);
  for (auto & vec: tr4) max1.push_back(vec);
  for (auto & vec: tr5) max1.push_back(vec);
  for (auto & vec: tr6) max1.push_back(vec);
 
  std::sort(max1.begin(), max1.end(),
          [](const std::vector<float>& a, const std::vector<float>& b) {
  return a[0] > b[0];
	    });
    return max1[0];
}

 
float LeptonSkimming::Dphi(float phi1,float phi2){
    float result = phi1 - phi2;
    while (result > float(M_PI)) result -= float(2*M_PI);
    while (result <= -float(M_PI)) result += float(2*M_PI);
    return result;
}


float LeptonSkimming::DR(float eta1,float phi1,float eta2, float phi2){
  return TMath::Sqrt((eta1-eta2)*(eta1-eta2)+Dphi(phi1,phi2)*Dphi(phi1,phi2));
}


// ------------ method called on each new Event  ------------
bool
LeptonSkimming::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // using namespace PhysicsTools;


  //Get a few collections to apply basic electron ID
  //Get electrons
  edm::Handle<edm::View<reco::GsfElectron> > electrons;
  iEvent.getByToken(electronsToken_, electrons);

 
  edm::Handle<std::vector<reco::Muon>> muons;
   iEvent.getByToken(muonsToken_,muons);

  //Get conversions
  edm::Handle<reco::ConversionCollection> conversions;
  iEvent.getByToken(conversionsToken_, conversions);    
  // Get the beam spot
  edm::Handle<reco::BeamSpot> theBeamSpot;
  iEvent.getByToken(beamSpotToken_,theBeamSpot);  
  //Get vertices 
  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vtxToken_, vertices);
  //continue if there are no vertices
  if (vertices->empty()) return false;
  edm::Handle<vector<reco::Track>> tracks;
  iEvent.getByToken(Tracks_, tracks);
  edm::Handle<edm::TriggerResults> trigResults;
  iEvent.getByToken(trgresultsToken_, trigResults);
  edm::ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  KalmanVertexFitter theKalmanFitter(false);
  TransientVertex LLvertex;
  

  vertex_x=-1000000;  vertex_y=-1000000;  vertex_z=-1000000;  
  beam_x=-9999,beam_y=-9999,beam_z=-99999;
 
  // trigger1=0; trigger2=0; trigger3=0; trigger4=0; trigger5=0; trigger6=0;
  nmuons=0; nel=0; ntracks=0;
    
  TrgObj1_PtEtaPhiCharge.clear(); TrgObj2_PtEtaPhiCharge.clear();
  TrgObj3_PtEtaPhiCharge.clear(); TrgObj4_PtEtaPhiCharge.clear();
  TrgObj5_PtEtaPhiCharge.clear(); TrgObj6_PtEtaPhiCharge.clear();
  trigger1=0,trigger2=0,trigger3=0,trigger4=0,trigger5=0,trigger6=0;
  SelectedTrgObj_PtEtaPhiCharge.clear(); SelectedMu_index=-1;
  SelectedMu_DR=1000; muon_pt.clear(); muon_eta.clear(); muon_phi.clear();
  Result=false;  el_pt.clear(); el_eta.clear(); el_phi.clear();
  Trk_container.clear();  MuTracks.clear();  ElTracks.clear(); 
  object_container.clear(); object_id.clear();  cleanedTracks.clear(); 
  Epair_ObjectId.clear(); muon_soft.clear(); muon_medium.clear(); muon_tight.clear();
  Epair_ObjectIndex.clear(); cleanedObjTracks.clear(); cleanedPairTracks.clear();
    Epair_ObjectIndex.clear(); Epair_ObjectId.clear(); Epair_TrkIndex.clear();
  //internal stuff
   ZvertexTrg=-100000000;

for (VertexCollection::const_iterator vtx = vertices->begin();   vtx != vertices->end(); ++vtx) {
    bool isFake = vtx->isFake();
    if ( isFake) continue;
        vertex_x=vtx->x();  vertex_y=vtx->y();  vertex_z=vtx->z(); 
	if ( vertex_x!=-10000000) break;
    }
  if (vertex_x==-1000000)
         return false;
  reco::TrackBase::Point  vertex_point;
  vertex_point.SetCoordinates(vertex_x,vertex_y,vertex_z);
  beam_x= theBeamSpot->x0(); beam_y= theBeamSpot->y0(); beam_z= theBeamSpot->z0();   

    std::pair<std::vector<float>,std::vector<std::vector<std::vector<float>>>> trgresult=HLTAnalyze(iEvent,iSetup,HLTPath_,HLTFilter_);   
    if(trgresult.first[0]+trgresult.first[1]+trgresult.first[2]+trgresult.first[3]+trgresult.first[4]+trgresult.first[5]==0) return false;    
    SelectedTrgObj_PtEtaPhiCharge=SelectTrg_Object(trgresult.second[0],trgresult.second[1],trgresult.second[2],trgresult.second[3],trgresult.second[4],trgresult.second[5]);
 
  SelectedMu_DR=1000; 
  MuTracks.clear();  object_container.clear(); object_id.clear(); nmuons=0;
  for (std::vector<reco::Muon>::const_iterator mu=muons->begin(); mu!=muons->end(); mu++){
    if (fabs(mu->eta())>EtaTrack_Cut) continue;
    bool tight=false,soft=false;
    if(vertices.isValid()){
      tight=isTightMuonCustom(*mu,(*vertices)[0]);
      soft=muon::isSoftMuon(*mu,(*vertices)[0]);
     }    
    const Track * mutrack= mu->bestTrack();
    muon_medium.push_back(isMediumMuonCustom(*mu));    
    muon_tight.push_back(tight); muon_soft.push_back(soft);
    muon_pt.push_back(mu->pt()); muon_eta.push_back(mu->eta()); muon_phi.push_back(mu->phi()); 
    auto muTrack=std::make_shared<reco::Track>(*mutrack);
    MuTracks.push_back(muTrack);
    object_container.push_back(nmuons);
    object_id.push_back(13);    
    if ( DR(mu->eta(),mu->phi(), SelectedTrgObj_PtEtaPhiCharge[1], SelectedTrgObj_PtEtaPhiCharge[2])<MuTrgMatchCone &&  SelectedMu_DR>DR(mu->eta(),mu->phi(), SelectedTrgObj_PtEtaPhiCharge[1], SelectedTrgObj_PtEtaPhiCharge[2]) ){
        SelectedMu_DR=DR(mu->eta(),mu->phi(), SelectedTrgObj_PtEtaPhiCharge[1], SelectedTrgObj_PtEtaPhiCharge[2]);
        ZvertexTrg=mu->vz();}
    nmuons++;
    //delete mutrack;
  }
  
  if (SelectedMu_DR==1000 && SkipIfNoMuMatch){
      return false;
   }

  ElTracks.clear();
  for(size_t e=0; e<electrons->size(); e++){
      const auto el = electrons->ptrAt(e);  
      bool passConvVeto = !ConversionTools::hasMatchedConversion(*el, conversions, theBeamSpot->position());
      if (!passConvVeto) continue;
      if (fabs(el->eta())>EtaTrack_Cut) continue;
      if (el->pt()<PtEl_Cut) continue;
      const Track * eltrack= el->bestTrack();
      auto ElTrack=std::make_shared<reco::Track>(*eltrack);
      ElTracks.push_back(ElTrack); object_container.push_back(nel);
      el_pt.push_back(el->pt()); el_eta.push_back(el->eta()); el_phi.push_back(el->phi());
      nel++; object_id.push_back(11);
    }
 
  cleanedTracks.clear(); 
  trk_index=0;
 for (typename vector<reco::Track>::const_iterator trk=tracks->begin(); trk!=tracks->end(); trk++){
   if (!trk->quality(Track::highPurity)) continue;
   if (trk->pt()<PtTrack_Cut) continue;
   if (fabs(trk->eta())>EtaTrack_Cut) continue;
   if(trk->charge()==0) continue;
   if(trk->normalizedChi2()>MaxChi2Track_Cut || trk->normalizedChi2()<MinChi2Track_Cut) continue;
   if (fabs(trk->dxy())/trk->dxyError()<TrackSdxy_Cut) continue;
   double minDR=1000;
   for (typename vector<reco::Muon>::const_iterator mu=muons->begin(); mu!=muons->end(); mu++){
      double tempDR=DR(mu->eta(),mu->phi(),trk->eta(),trk->phi());
      if (minDR<tempDR) continue;
      minDR=tempDR;
   }
   if (minDR<MuTrkMinDR_Cut) continue;
   if (SelectedMu_DR<1000 ){
     if (fabs(ZvertexTrg-trk->vz())>TrackMuDz_Cut ) continue;
     if ( DR(trk->eta(),trk->phi(),SelectedTrgObj_PtEtaPhiCharge[1],SelectedTrgObj_PtEtaPhiCharge[2])<TrgExclusionCone) continue;
   }
   //assignments   
   auto cleanTrack=std::make_shared<reco::Track>(*trk);
   cleanedTracks.push_back(cleanTrack);
   Trk_container.push_back(trk_index);
   trk_index++;
 }
 
    //create mother ee combination
 
 // fit track pairs
    cleanedObjTracks.clear(); cleanedPairTracks.clear();     
    TLorentzVector vel1,vel2;
    std::vector<std::shared_ptr<reco::Track>> cleanedObjects; 
    //objects    
    for(auto & vec: MuTracks) cleanedObjects.push_back(vec);
    for(auto & vec: ElTracks) cleanedObjects.push_back(vec);  
    if (cleanedObjects.size()==0)
             return false;

    for(unsigned int iobj=0; iobj<cleanedObjects.size(); iobj++){
      auto obj=cleanedObjects.at(iobj);
      auto tranobj=std::make_shared<reco::TransientTrack>(reco::TransientTrack(*obj,&(*bFieldHandle)));
      unsigned int index=object_container.at(iobj);
      if ( object_id.at(iobj)==13 && QualMu_Cut==1 && !muon_soft.at(index)) continue;
     if ( object_id.at(iobj)==13 && QualMu_Cut==2 && !muon_medium.at(index)) continue;
     if ( object_id.at(iobj)==13 && QualMu_Cut==3 && !muon_tight.at(index)) continue;
      if (object_id.at(iobj)==13) vel1.SetPtEtaPhiM(muon_pt.at(index),muon_eta.at(index),muon_phi.at(index),0.0005);
      else vel1.SetPtEtaPhiM(el_pt.at(index),el_eta.at(index),el_phi.at(index),0.0005);
     
      if (object_id.at(iobj)==13 && vel1.Pt()<PtMu_Cut) continue;
      for(unsigned int itrk2=0; itrk2<cleanedTracks.size(); itrk2++){
         auto trk2=cleanedTracks.at(itrk2);
         if (obj->charge()*trk2->charge()==1) continue;
         if (ObjPtLargerThanTrack && vel1.Pt()<trk2->pt()) continue;
      
         vel2.SetPtEtaPhiM(trk2->pt(),trk2->eta(),trk2->phi(),0.0005);
         if (DR(obj->eta(),obj->phi(),trk2->eta(),trk2->phi())<TrkObjExclusionCone) continue;
         if (object_id.at(iobj)==13 && DR(obj->eta(),obj->phi(),SelectedTrgObj_PtEtaPhiCharge[1],SelectedTrgObj_PtEtaPhiCharge[2])<MuTrgExclusionCone) continue;
         if (object_id.at(iobj)==11 && DR(obj->eta(),obj->phi(),SelectedTrgObj_PtEtaPhiCharge[1],SelectedTrgObj_PtEtaPhiCharge[2])<ElTrgExclusionCone) continue;
         if (SelectedMu_DR<1000 ){
	   if (object_id.at(iobj)==13 && fabs(ZvertexTrg- obj->vz())>MuTrgMuDz_Cut ) continue;
           if (object_id.at(iobj)==11 && fabs(ZvertexTrg-obj->vz())>ElTrgMuDz_Cut ) continue;
         }
         if ((vel1+vel2).M()>MaxMee_Cut || (vel1+vel2).M()<MinMee_Cut ) continue;   
	 auto trantrk2=std::make_shared<reco::TransientTrack>(reco::TransientTrack(*trk2,&(*bFieldHandle)));
	 tempTracks.clear(); 
         tempTracks.push_back(*tranobj); tempTracks.push_back(*trantrk2);
         LLvertex = theKalmanFitter.vertex(tempTracks);
         if (!LLvertex.isValid()) continue;
	 if (ChiSquaredProbability(LLvertex.totalChiSquared(),LLvertex.degreesOfFreedom())<Probee_Cut)  continue;
         if (ZvertexTrg>-1000000 && fabs(ZvertexTrg-LLvertex.position().z())>EpairZvtx_Cut ) continue;
	 GlobalError err =LLvertex.positionError();
	 GlobalPoint Dispbeamspot(-1*((theBeamSpot->x0()-LLvertex.position().x())+(LLvertex.position().z()-theBeamSpot->z0()) * theBeamSpot->dxdz()),-1*((theBeamSpot->y0()-LLvertex.position().y())+ (LLvertex.position().z()-theBeamSpot->z0()) * theBeamSpot->dydz()), 0);
          math::XYZVector pperp((vel1+vel2).Px(),(vel1+vel2).Py(),0);
          math::XYZVector vperp(Dispbeamspot.x(),Dispbeamspot.y(),0.);
          float tempCos=vperp.Dot(pperp)/(vperp.R()*pperp.R());
          if (tempCos<Cosee_Cut) continue;
	  cleanedObjTracks.push_back(obj);
          cleanedPairTracks.push_back(trk2);
	  Epair_ObjectIndex.push_back(object_container.at(iobj));
          Epair_ObjectId.push_back(object_id.at(iobj));
          Epair_TrkIndex.push_back(Trk_container.at(itrk2));     
	}
    }
     
    
    // B recontrsuvtion
     TLorentzVector vK; 
    for(unsigned int iobj=0; iobj<cleanedObjTracks.size(); iobj++){
      auto objtrk=cleanedObjTracks.at(iobj);
      auto pairtrk=cleanedPairTracks.at(iobj);
      auto tranobj=std::make_shared<reco::TransientTrack>(reco::TransientTrack(*objtrk,&(*bFieldHandle)));
      auto tranpair=std::make_shared<reco::TransientTrack>(reco::TransientTrack(*pairtrk,&(*bFieldHandle)));
      unsigned int index=Epair_ObjectIndex.at(iobj);
      if ( Epair_ObjectId.at(iobj)==13) vel1.SetPtEtaPhiM(muon_pt.at(index),muon_eta.at(index),muon_phi.at(index),0.0005);
      else vel1.SetPtEtaPhiM(el_pt.at(index),el_eta.at(index),el_phi.at(index),0.0005);
      for(unsigned int itrk=0; itrk<cleanedTracks.size(); itrk++){
	 auto trk=cleanedTracks.at(itrk);
         if(DR(objtrk->eta(),objtrk->phi(),trk->eta(),trk->phi())<TrkObjExclusionCone) continue;
	 if (trk->pt()<PtKTrack_Cut) continue;
         if (fabs(trk->dxy(vertex_point))/trk->dxyError()<Ksdxy_Cut) continue;
         if (trk->charge()==pairtrk->charge() && DR(pairtrk->eta(),pairtrk->phi(),trk->eta(),trk->phi())<TrkTrkMinDR_Cut) continue;
         vel2.SetPtEtaPhiM(pairtrk->pt(),pairtrk->eta(),pairtrk->phi(),0.0005);
         vK.SetPtEtaPhiM(trk->pt(),trk->eta(),trk->phi(),0.493);
         if ((vel1+vel2+vK).M()> MaxMB_Cut || (vel1+vel2+vK).M()< MinMB_Cut) continue;
         if ((vel1+vel2+vK).Pt()<PtB_Cut) continue;
         auto trantrk=std::make_shared<reco::TransientTrack>(reco::TransientTrack(*trk,&(*bFieldHandle)));
	 tempTracks.clear();
         tempTracks.push_back(*tranobj); tempTracks.push_back(*tranpair);
         tempTracks.push_back(*trantrk);
	 
         LLvertex = theKalmanFitter.vertex(tempTracks);
	 if (!LLvertex.isValid()) continue;
	  if (ChiSquaredProbability(LLvertex.totalChiSquared(),LLvertex.degreesOfFreedom())<ProbeeK_Cut) continue;
	 GlobalError err =LLvertex.positionError();
         GlobalPoint Dispbeamspot(-1*((theBeamSpot->x0()-LLvertex.position().x())+(LLvertex.position().z()-theBeamSpot->z0()) * theBeamSpot->dxdz()),-1*((theBeamSpot->y0()-LLvertex.position().y())+ (LLvertex.position().z()-theBeamSpot->z0()) * theBeamSpot->dydz()), 0);
      
         math::XYZVector pperp((vel1+vel2+vK).Px(),(vel1+vel2+vK).Py(),0);
         math::XYZVector vperp(Dispbeamspot.x(),Dispbeamspot.y(),0.);
         float tempCos=vperp.Dot(pperp)/(vperp.R()*pperp.R());
         if (tempCos<CoseeK_Cut) continue;
         if (SLxy_Cut>Dispbeamspot.perp()/TMath::Sqrt(err.rerr(Dispbeamspot))) continue;
          Result=true;
	  break;
     }
      if (Result) break;
    }
  
     return Result;
   
}

// ------------ method called once each stream before processing any runs, lumis or events  ------------
void
LeptonSkimming::beginStream(edm::StreamID)
{
}

// ------------ method called once each stream after processing all runs, lumis and events  ------------
void
LeptonSkimming::endStream() {
}

// ------------ method called when starting to processes a run  ------------
/*
void
LeptonSkimming::beginRun(edm::Run const&, edm::EventSetup const&)
{ 
}
*/
 
// ------------ method called when ending the processing of a run  ------------
/*
void
LeptonSkimming::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when starting to processes a luminosity block  ------------
/*
void
LeptonSkimming::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
LeptonSkimming::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/
 
// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
LeptonSkimming::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//define this as a plug-in
DEFINE_FWK_MODULE(LeptonSkimming);

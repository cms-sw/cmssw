#include <iostream>
//

#include "DQMOffline/EGamma/plugins/PiZeroAnalyzer.h"


//#define TWOPI 6.283185308
//

/** \class PiZeroAnalyzer
 **
 **
 **  $Id: PiZeroAnalyzer
 **  $Date: 2011/10/13 13:57:27 $
 **  authors:
 **   Nancy Marinelli, U. of Notre Dame, US
 **   Jamie Antonelli, U. of Notre Dame, US
 **
 ***/



using namespace std;


PiZeroAnalyzer::PiZeroAnalyzer( const edm::ParameterSet& pset )
{

    fName_              = pset.getUntrackedParameter<std::string>("Name");
    verbosity_          = pset.getUntrackedParameter<int>("Verbosity");

    prescaleFactor_     = pset.getUntrackedParameter<int>("prescaleFactor",1);



    barrelEcalHits_     = pset.getParameter<edm::InputTag>("barrelEcalHits");
    endcapEcalHits_     = pset.getParameter<edm::InputTag>("endcapEcalHits");


    standAlone_         = pset.getParameter<bool>("standAlone");





    // parameters for Pizero finding
    seleXtalMinEnergy_    = pset.getParameter<double> ("seleXtalMinEnergy");
    clusSeedThr_          = pset.getParameter<double> ("clusSeedThr");
    clusEtaSize_          = pset.getParameter<int> ("clusEtaSize");
    clusPhiSize_          = pset.getParameter<int> ("clusPhiSize");
    ParameterLogWeighted_ = pset.getParameter<bool> ("ParameterLogWeighted");
    ParameterX0_          = pset.getParameter<double> ("ParameterX0");
    ParameterT0_barl_     = pset.getParameter<double> ("ParameterT0_barl");
    ParameterW0_          = pset.getParameter<double> ("ParameterW0");

    selePtGammaOne_       = pset.getParameter<double> ("selePtGammaOne");
    selePtGammaTwo_       = pset.getParameter<double> ("selePtGammaTwo");
    seleS4S9GammaOne_     = pset.getParameter<double> ("seleS4S9GammaOne");
    seleS4S9GammaTwo_     = pset.getParameter<double> ("seleS4S9GammaTwo");
    selePtPi0_            = pset.getParameter<double> ("selePtPi0");
    selePi0Iso_           = pset.getParameter<double> ("selePi0Iso");
    selePi0BeltDR_        = pset.getParameter<double> ("selePi0BeltDR");
    selePi0BeltDeta_      = pset.getParameter<double> ("selePi0BeltDeta");
    seleMinvMaxPi0_       = pset.getParameter<double> ("seleMinvMaxPi0");
    seleMinvMinPi0_       = pset.getParameter<double> ("seleMinvMinPi0");

    parameters_ = pset;


}



PiZeroAnalyzer::~PiZeroAnalyzer() {




}


void PiZeroAnalyzer::beginJob()
{


  nEvt_=0;
  nEntry_=0;

  dbe_ = 0;
  dbe_ = edm::Service<DQMStore>().operator->();



 if (dbe_) {
    if (verbosity_ > 0 ) {
      dbe_->setVerbose(1);
    } else {
      dbe_->setVerbose(0);
    }
  }
  if (dbe_) {
    if (verbosity_ > 0 ) dbe_->showDirStructure();
  }




  //booking all histograms

  if (dbe_) {

    currentFolder_.str("");
    currentFolder_ << "Egamma/PiZeroAnalyzer/";
    dbe_->setCurrentFolder(currentFolder_.str());




    hMinvPi0EB_ = dbe_->book1D("Pi0InvmassEB","Pi0 Invariant Mass in EB",100,0.,0.5);
    hMinvPi0EB_->setAxisTitle("Inv Mass [GeV] ",1);

    hPt1Pi0EB_ = dbe_->book1D("Pt1Pi0EB","Pt 1st most energetic Pi0 photon in EB",100,0.,20.);
    hPt1Pi0EB_->setAxisTitle("1st photon Pt [GeV] ",1);

    hPt2Pi0EB_ = dbe_->book1D("Pt2Pi0EB","Pt 2nd most energetic Pi0 photon in EB",100,0.,20.);
    hPt2Pi0EB_->setAxisTitle("2nd photon Pt [GeV] ",1);

    hPtPi0EB_ = dbe_->book1D("PtPi0EB","Pi0 Pt in EB",100,0.,20.);
    hPtPi0EB_->setAxisTitle("Pi0 Pt [GeV] ",1);

    hIsoPi0EB_ = dbe_->book1D("IsoPi0EB","Pi0 Iso in EB",50,0.,1.);
    hIsoPi0EB_->setAxisTitle("Pi0 Iso",1);


  }

}






void PiZeroAnalyzer::analyze( const edm::Event& e, const edm::EventSetup& esup )
{

  using namespace edm;

  if (nEvt_% prescaleFactor_ ) return;
  nEvt_++;
  LogInfo("PiZeroAnalyzer") << "PiZeroAnalyzer Analyzing event number: " << e.id() << " Global Counter " << nEvt_ <<"\n";


  // Get EcalRecHits
  bool validEcalRecHits=true;
  Handle<EcalRecHitCollection> barrelHitHandle;
  EcalRecHitCollection barrelRecHits;
  e.getByLabel(barrelEcalHits_, barrelHitHandle);
  if (!barrelHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<barrelEcalHits_.label();
    validEcalRecHits=false;
  }

  Handle<EcalRecHitCollection> endcapHitHandle;
  e.getByLabel(endcapEcalHits_, endcapHitHandle);
  EcalRecHitCollection endcapRecHits;
  if (!endcapHitHandle.isValid()) {
    edm::LogError("PhotonProducer") << "Error! Can't get the product "<<endcapEcalHits_.label();
    validEcalRecHits=false;
  }

  if (validEcalRecHits) makePizero(esup,  barrelHitHandle, endcapHitHandle);



}

void PiZeroAnalyzer::makePizero ( const edm::EventSetup& es, const edm::Handle<EcalRecHitCollection> rhEB,  const edm::Handle<EcalRecHitCollection> rhEE ) {

  const EcalRecHitCollection *hitCollection_p = rhEB.product();

  edm::ESHandle<CaloGeometry> geoHandle;
  es.get<CaloGeometryRecord>().get(geoHandle);

  edm::ESHandle<CaloTopology> theCaloTopology;
  es.get<CaloTopologyRecord>().get(theCaloTopology);


  const CaloSubdetectorGeometry *geometry_p;
  const CaloSubdetectorTopology *topology_p;
  const CaloSubdetectorGeometry *geometryES_p;
  geometry_p = geoHandle->getSubdetectorGeometry(DetId::Ecal,EcalBarrel);
  geometryES_p = geoHandle->getSubdetectorGeometry(DetId::Ecal, EcalPreshower);

  // Parameters for the position calculation:
  edm::ParameterSet posCalcParameters =
    parameters_.getParameter<edm::ParameterSet>("posCalcParameters");
  PositionCalc posCalculator_ = PositionCalc(posCalcParameters);
  //
  std::map<DetId, EcalRecHit> recHitsEB_map;
  //
  std::vector<EcalRecHit> seeds;

  seeds.clear();
  //
  vector<EBDetId> usedXtals;
  usedXtals.clear();
  //
  EcalRecHitCollection::const_iterator itb;
  //
  static const int MAXCLUS = 2000;
  int nClus=0;
  vector<float> eClus;
  vector<float> etClus;
  vector<float> etaClus;
  vector<float> phiClus;
  vector<EBDetId> max_hit;
  vector< vector<EcalRecHit> > RecHitsCluster;
  vector<float> s4s9Clus;

  // find cluster seeds in EB
  for(itb=rhEB->begin(); itb!=rhEB->end(); ++itb){
    EBDetId id(itb->id());
    double energy = itb->energy();
    if (energy > seleXtalMinEnergy_) {
      std::pair<DetId, EcalRecHit> map_entry(itb->id(), *itb);
      recHitsEB_map.insert(map_entry);
    }
    if (energy > clusSeedThr_) seeds.push_back(*itb);
  } // Eb rechits

  sort(seeds.begin(), seeds.end(), ecalRecHitLess());
  for (std::vector<EcalRecHit>::iterator itseed=seeds.begin(); itseed!=seeds.end(); itseed++) {
    EBDetId seed_id = itseed->id();
    std::vector<EBDetId>::const_iterator usedIds;

    bool seedAlreadyUsed=false;
    for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
      if(*usedIds==seed_id){
	seedAlreadyUsed=true;
	//cout<< " Seed with energy "<<itseed->energy()<<" was used !"<<endl;
	break;
      }
    }
    if(seedAlreadyUsed)continue;
    topology_p = theCaloTopology->getSubdetectorTopology(DetId::Ecal,EcalBarrel);
    std::vector<DetId> clus_v = topology_p->getWindow(seed_id,clusEtaSize_,clusPhiSize_);
    //std::vector<DetId> clus_used;
    std::vector<std::pair<DetId, float> > clus_used;

    vector<EcalRecHit> RecHitsInWindow;

    double simple_energy = 0;

    for (std::vector<DetId>::iterator det=clus_v.begin(); det!=clus_v.end(); det++) {
      // EBDetId EBdet = *det;
      //      cout<<" det "<< EBdet<<" ieta "<<EBdet.ieta()<<" iphi "<<EBdet.iphi()<<endl;
      bool  HitAlreadyUsed=false;
      for(usedIds=usedXtals.begin(); usedIds!=usedXtals.end(); usedIds++){
	if(*usedIds==*det){
	  HitAlreadyUsed=true;
	  break;
	}
      }
      if(HitAlreadyUsed)continue;
      if (recHitsEB_map.find(*det) != recHitsEB_map.end()){
	//      cout<<" Used det "<< EBdet<<endl;
	std::map<DetId, EcalRecHit>::iterator aHit;
	aHit = recHitsEB_map.find(*det);
	usedXtals.push_back(*det);
	RecHitsInWindow.push_back(aHit->second);
	clus_used.push_back( std::pair<DetId, float>(*det, 1.) );
	simple_energy = simple_energy + aHit->second.energy();
      }
    }

    math::XYZPoint clus_pos = posCalculator_.Calculate_Location(clus_used,hitCollection_p,geometry_p,geometryES_p);
    float theta_s = 2. * atan(exp(-clus_pos.eta()));
    float p0x_s = simple_energy * sin(theta_s) * cos(clus_pos.phi());
    float p0y_s = simple_energy * sin(theta_s) * sin(clus_pos.phi());
    //      float p0z_s = simple_energy * cos(theta_s);
    float et_s = sqrt( p0x_s*p0x_s + p0y_s*p0y_s);

    //cout << "       Simple Clustering: E,Et,px,py,pz: "<<simple_energy<<" "<<et_s<<" "<<p0x_s<<" "<<p0y_s<<" "<<endl;

    eClus.push_back(simple_energy);
    etClus.push_back(et_s);
    etaClus.push_back(clus_pos.eta());
    phiClus.push_back(clus_pos.phi());
    max_hit.push_back(seed_id);
    RecHitsCluster.push_back(RecHitsInWindow);
    //Compute S4/S9 variable
    //We are not sure to have 9 RecHits so need to check eta and phi:
    float s4s9_[4];
    for(int i=0;i<4;i++)s4s9_[i]= itseed->energy();
    for(unsigned int j=0; j<RecHitsInWindow.size();j++){
      //cout << " Simple cluster rh, ieta, iphi : "<<((EBDetId)RecHitsInWindow[j].id()).ieta()<<" "<<((EBDetId)RecHitsInWindow[j].id()).iphi()<<endl;
      if((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()-1 && seed_id.ieta()!=1 ) || ( seed_id.ieta()==1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()-2))){
	if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	  s4s9_[0]+=RecHitsInWindow[j].energy();
	}else{
	  if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()){
	    s4s9_[0]+=RecHitsInWindow[j].energy();
	    s4s9_[1]+=RecHitsInWindow[j].energy();
	  }else{
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
	      s4s9_[1]+=RecHitsInWindow[j].energy();
	    }
	  }
	}
      }else{
	if(((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()){
	  if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	    s4s9_[0]+=RecHitsInWindow[j].energy();
	    s4s9_[3]+=RecHitsInWindow[j].energy();
	  }else{
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
	      s4s9_[1]+=RecHitsInWindow[j].energy();
	      s4s9_[2]+=RecHitsInWindow[j].energy();
	    }
	  }
	}else{
	  if((((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()+1 && seed_id.ieta()!=-1 ) || ( seed_id.ieta()==-1 && (((EBDetId)RecHitsInWindow[j].id()).ieta() == seed_id.ieta()+2))){
	    if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()-1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()-1 ){
	      s4s9_[3]+=RecHitsInWindow[j].energy();
	    }else{
	      if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()){
		s4s9_[2]+=RecHitsInWindow[j].energy();
		s4s9_[3]+=RecHitsInWindow[j].energy();
	      }else{
		if(((EBDetId)RecHitsInWindow[j].id()).iphi() == seed_id.iphi()+1 ||((EBDetId)RecHitsInWindow[j].id()).iphi()-360 == seed_id.iphi()+1 ){
		  s4s9_[2]+=RecHitsInWindow[j].energy();
		}
	      }
	    }
	  }else{
	    cout<<" (EBDetId)RecHitsInWindow[j].id()).ieta() "<<((EBDetId)RecHitsInWindow[j].id()).ieta()<<" seed_id.ieta() "<<seed_id.ieta()<<endl;
	    cout<<" Problem with S4 calculation "<<endl;return;
	  }
	}
      }
    }
    s4s9Clus.push_back(*max_element( s4s9_,s4s9_+4)/simple_energy);
    //    cout<<" s4s9Clus[0] "<<s4s9_[0]/simple_energy<<" s4s9Clus[1] "<<s4s9_[1]/simple_energy<<" s4s9Clus[2] "<<s4s9_[2]/simple_energy<<" s4s9Clus[3] "<<s4s9_[3]/simple_energy<<endl;
    //    cout<<" Max "<<*max_element( s4s9_,s4s9_+4)/simple_energy<<endl;
    nClus++;
    if (nClus == MAXCLUS) return;
  }  //  End loop over seed clusters

  // cout<< " Pi0 clusters: "<<nClus<<endl;

  // Selection, based on Simple clustering
  //pi0 candidates
  static const int MAXPI0S = 200;
  int npi0_s=0;

  vector<EBDetId> scXtals;
  scXtals.clear();

  if (nClus <= 1) return;
  for(Int_t i=0 ; i<nClus ; i++){
    for(Int_t j=i+1 ; j<nClus ; j++){
      //      cout<<" i "<<i<<"  etClus[i] "<<etClus[i]<<" j "<<j<<"  etClus[j] "<<etClus[j]<<endl;
      if( etClus[i]>selePtGammaOne_ && etClus[j]>selePtGammaTwo_ && s4s9Clus[i]>seleS4S9GammaOne_ && s4s9Clus[j]>seleS4S9GammaTwo_){
	float theta_0 = 2. * atan(exp(-etaClus[i]));
	float theta_1 = 2. * atan(exp(-etaClus[j]));

	float p0x = eClus[i] * sin(theta_0) * cos(phiClus[i]);
	float p1x = eClus[j] * sin(theta_1) * cos(phiClus[j]);
	float p0y = eClus[i] * sin(theta_0) * sin(phiClus[i]);
	float p1y = eClus[j] * sin(theta_1) * sin(phiClus[j]);
	float p0z = eClus[i] * cos(theta_0);
	float p1z = eClus[j] * cos(theta_1);

	float pt_pi0 = sqrt( (p0x+p1x)*(p0x+p1x) + (p0y+p1y)*(p0y+p1y));
	//      cout<<" pt_pi0 "<<pt_pi0<<endl;
	if (pt_pi0 < selePtPi0_)continue;
	float m_inv = sqrt ( (eClus[i] + eClus[j])*(eClus[i] + eClus[j]) - (p0x+p1x)*(p0x+p1x) - (p0y+p1y)*(p0y+p1y) - (p0z+p1z)*(p0z+p1z) );
	if ( (m_inv<seleMinvMaxPi0_) && (m_inv>seleMinvMinPi0_) ){

	  //New Loop on cluster to measure isolation:
	  vector<int> IsoClus;
	  IsoClus.clear();
	  float Iso = 0;
	  TVector3 pi0vect = TVector3((p0x+p1x), (p0y+p1y), (p0z+p1z));
	  for(Int_t k=0 ; k<nClus ; k++){
	    if(k==i || k==j)continue;
	    TVector3 Clusvect = TVector3(eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * cos(phiClus[k]), eClus[k] * sin(2. * atan(exp(-etaClus[k]))) * sin(phiClus[k]) , eClus[k] * cos(2. * atan(exp(-etaClus[k]))));
	    float dretaclpi0 = fabs(etaClus[k] - pi0vect.Eta());
	    float drclpi0 = Clusvect.DeltaR(pi0vect);

	    if((drclpi0<selePi0BeltDR_) && (dretaclpi0<selePi0BeltDeta_) ){

	      Iso = Iso + etClus[k];
	      IsoClus.push_back(k);
	    }
	  }


	  if(Iso/pt_pi0<selePi0Iso_){

	    hMinvPi0EB_->Fill(m_inv);
	    hPt1Pi0EB_->Fill(etClus[i]);
	    hPt2Pi0EB_->Fill(etClus[j]);
	    hPtPi0EB_->Fill(pt_pi0);
	    hIsoPi0EB_->Fill(Iso/pt_pi0);


	    npi0_s++;
	  }

	  if(npi0_s == MAXPI0S) return;
	}
      }
    }
  }

}



void PiZeroAnalyzer::endJob()
{



  bool outputMEsInRootFile = parameters_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters_.getParameter<std::string>("OutputFileName");
  if(outputMEsInRootFile){
    dbe_->save(outputFileName);
  }

  edm::LogInfo("PiZeroAnalyzer") << "Analyzed " << nEvt_  << "\n";
  return ;
}



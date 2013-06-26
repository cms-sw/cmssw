//
// $Id: PATMHTProducer.cc,v 1.8 2013/02/27 23:26:56 wmtan Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATMHTProducer.h"

pat::PATMHTProducer::PATMHTProducer(const edm::ParameterSet & iConfig){

  // Initialize the configurables
  verbose_ = iConfig.getParameter<double>("verbose");

  jetLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("jetTag");
  eleLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("electronTag");
  muoLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("muonTag");
  tauLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("tauTag");
  phoLabel_ = iConfig.getUntrackedParameter<edm::InputTag>("photonTag");
  
  uncertaintyScaleFactor_ = iConfig.getParameter<double>( "uncertaintyScaleFactor") ;
  controlledUncertainty_  = iConfig.getParameter<bool>( "controlledUncertainty") ;

  jetPtMin_     = iConfig.getParameter<double>("jetPtMin");
  jetEtaMax_    = iConfig.getParameter<double>("jetEtaMax");
  jetEMfracMax_ = iConfig.getParameter<double>("jetEMfracMax");
  elePtMin_     = iConfig.getParameter<double>("elePtMin");
  eleEtaMax_    = iConfig.getParameter<double>("eleEtaMax");
  muonPtMin_    = iConfig.getParameter<double>("muonPtMin");
  muonEtaMax_   = iConfig.getParameter<double>("muonEtaMax");

  jetEtUncertaintyParameter0_ =  iConfig.getParameter<double>( "jetEtUncertaintyParameter0") ; 
  jetEtUncertaintyParameter1_ =  iConfig.getParameter<double>( "jetEtUncertaintyParameter1") ; 
  jetEtUncertaintyParameter2_ =  iConfig.getParameter<double>( "jetEtUncertaintyParameter2") ; 
  jetPhiUncertaintyParameter0_=  iConfig.getParameter<double>( "jetPhiUncertaintyParameter0"); 
  jetPhiUncertaintyParameter1_=  iConfig.getParameter<double>( "jetPhiUncertaintyParameter1"); 
  jetPhiUncertaintyParameter2_=  iConfig.getParameter<double>( "jetPhiUncertaintyParameter2"); 
    
  eleEtUncertaintyParameter0_  =  iConfig.getParameter<double>( "eleEtUncertaintyParameter0") ; 
  elePhiUncertaintyParameter0_ =  iConfig.getParameter<double>( "elePhiUncertaintyParameter0") ; 

  muonEtUncertaintyParameter0_  =  iConfig.getParameter<double>( "muonEtUncertaintyParameter0") ; 
  muonPhiUncertaintyParameter0_ =  iConfig.getParameter<double>( "muonPhiUncertaintyParameter0") ; 

  CaloTowerTag_  = iConfig.getParameter<edm::InputTag>("CaloTowerTag");
  noHF_ = iConfig.getParameter<bool>( "noHF"); 
  
  //  muonCalo_ = iConfig.getParameter<bool>("muonCalo");
  towerEtThreshold_ = iConfig.getParameter<double>( "towerEtThreshold") ; 
  useHO_ = iConfig.getParameter<bool>("useHO");

  produces<pat::MHTCollection>();

}


pat::PATMHTProducer::~PATMHTProducer() {
}

void pat::PATMHTProducer::beginJob() {
  setUncertaintyParameters();
}
void pat::PATMHTProducer::endJob() {
}


void 
pat::PATMHTProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) 
{
  // make sure the SigInputObj container is empty
  while(physobjvector_.size()>0){
    physobjvector_.erase(physobjvector_.begin(),physobjvector_.end());
  }

  // Clean the clustered towers 
  s_clusteredTowers.clear();

  double number_of_jets = getJets(iEvent, iSetup);

  double number_of_electrons = getElectrons(iEvent, iSetup);

  double number_of_muons = getMuons(iEvent, iSetup);

  if (verbose_ == 1.) {
    std::cout << ">>>---> Number of jets: "  <<  number_of_jets << std::endl;
    std::cout << ">>>---> Number of electrons: "  <<  number_of_jets << std::endl;
    std::cout << ">>>---> Number of muons: " <<  number_of_muons << std::endl;
  }

  double met_x=0;
  double met_y=0;
  double met_et=0;
  double met_phi=0;
  double met_set=0;
  
  
  std::auto_ptr<pat::MHTCollection>  themetsigcoll (new pat::MHTCollection);

  if(physobjvector_.size() >= 1) { // Only when the vector is not empty 

    // calculate the MHT significance

    metsig::significanceAlgo signifAlgo;
    signifAlgo.addObjects(physobjvector_);
    double significance = signifAlgo.significance(met_et,met_phi,met_set);
    
    met_x=met_et*cos(met_phi);
    met_y=met_et*sin(met_phi);
  
    if (verbose_ == 1.) {
      std::cout << ">>>----> MHT Sgificance = " << significance << std::endl;
    }
 
    pat::MHT themetsigobj(reco::Particle::LorentzVector(met_x,met_y,0,met_et),met_set,significance);


    // Store the number of jets, electrons, muons
    themetsigobj.setNumberOfJets(number_of_jets);
    themetsigobj.setNumberOfElectrons(number_of_electrons);
    themetsigobj.setNumberOfMuons(number_of_muons);

    themetsigcoll->push_back(themetsigobj);

  } // If the vector is empty, just put empty product. 


  iEvent.put( themetsigcoll);

  
}  

// --------------------------------------------------
//  Fill Input Vector with Jets
// --------------------------------------------------
double 
pat::PATMHTProducer::getJets(edm::Event& iEvent, const edm::EventSetup & iSetup){

  std::string objectname="jet";

  double number_of_jets_ = 0.0;

  edm::Handle<edm::View<pat::Jet> > jetHandle;
  iEvent.getByLabel(jetLabel_,jetHandle);
  edm::View<pat::Jet> jets = *jetHandle;

  // Fill Input Vector with Jets 
  for(edm::View<pat::Jet>::const_iterator jet_iter = jets.begin(); jet_iter!=jets.end(); ++jet_iter){
    
    if( (jet_iter->pt()  < jetPtMin_) ||
	(TMath::Abs(jet_iter->eta()) > jetEtaMax_) || 
        (jet_iter->emEnergyFraction() > jetEMfracMax_ ) )
      continue; 
    
    double jet_et = jet_iter->et();
    double jet_phi = jet_iter->phi();
    
    if (verbose_ == 3.) {
      std::cout << "jet pt : " << jet_iter->pt() << " eta : " << jet_iter->eta() 
		<< " EMF: "  << jet_iter->emEnergyFraction() <<  std::endl;
    }
    
    double sigma_et, sigma_phi ;

    if (controlledUncertainty_) {
      sigma_et  = jetUncertainty.etUncertainty->Eval(jet_et);
      sigma_phi = jetUncertainty.phiUncertainty->Eval(jet_et);
    } 
    else {
      sigma_et = 0.0 ; // jet_iter->resolutionEt();
      sigma_phi =  0.0 ; //jet_iter->resolutionPhi();
    }

    if (verbose_ == 3.) {
      std::cout << "jet sigma_et : " << sigma_et << ", jet sigma_phi : " << sigma_phi <<  std::endl;
    }

    if(sigma_et<=0 || sigma_phi<=0)
      edm::LogWarning("PATMHTProducer") << 
	" uncertainties for "  << objectname <<
	" are (et, phi): " << sigma_et << "," << sigma_phi << " (et,phi): " << jet_et << "," << jet_phi;
    // try to read out the jet resolution from the root file at PatUtils
    //-- Store jet for Significance Calculation --//
    
    if (uncertaintyScaleFactor_ != 1.0){
      sigma_et  = sigma_et  * uncertaintyScaleFactor_;
      sigma_phi = sigma_phi * uncertaintyScaleFactor_;
      // edm::LogWarning("PATMHTProducer") << " using uncertainty scale factor: " << uncertaintyScaleFactor_ <<
      //" , uncertainties for " << objectname <<" changed to (et, phi): " << sigma_et << "," << sigma_phi; 
    }


    if (verbose_ == 101.) { // Study the Jets behavior

      std::cout << "v101> " <<  number_of_jets_ << "  " 
		<< jet_et   << "  "  <<  sigma_et << "  "
		<< jet_phi  << "  "  <<  sigma_phi << std::endl;
    }
  

    metsig::SigInputObj tmp_jet(objectname,jet_et,jet_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_jet);
    number_of_jets_ ++;

    //-- Store tower DetId's to be removed from Calo Tower sum later --//
    std::vector<CaloTowerPtr> v_towers = jet_iter->getCaloConstituents();
    //std::cout << "tower size = " << v_towers.size() << std::endl;

    for (unsigned int ii=0; ii < v_towers.size(); ii++) {
      s_clusteredTowers.insert( (*v_towers.at(ii)).id() );
      //std::cout << "tower id = " << (*v_towers.at(ii)).id() << std::endl;
    }

  }
  
  if (verbose_ == 101.) { // Study the Jets behavior - seperate events
    std::cout << "v101> --------------------------------------------" << std::endl;
  }
 
  return number_of_jets_;

}


// --------------------------------------------------
//  Fill Input Vector with Electrons
// --------------------------------------------------
double
pat::PATMHTProducer::getElectrons(edm::Event& iEvent, const edm::EventSetup & iSetup){

  std::string objectname="electron";

  double number_of_electrons_ = 0.0;

  // edm::ESHandle<CaloTowerConstituentsMap> cttopo;
  // iSetup.get<IdealGeometryRecord>().get(cttopo);
  // const CaloTowerConstituentsMap* caloTowerMap = cttopo.product();

  edm::Handle<edm::View<pat::Electron> > electronHandle;
  iEvent.getByLabel(eleLabel_,electronHandle);
  edm::View<pat::Electron> electrons = *electronHandle;
  DetId nullDetId;

  // Fill Input Vector with Electrons 
  for(edm::View<pat::Electron>::const_iterator electron_iter = electrons.begin(); electron_iter!=electrons.end(); ++electron_iter){

    // Select electrons
    if (electron_iter->et() < elePtMin_ || 
	TMath::Abs(electron_iter->eta()) > eleEtaMax_  ) continue; 

    if (verbose_ == 3.) {
      std::cout << "electron pt = " << electron_iter->pt()  << " eta : " << electron_iter->eta() 
		<<  std::endl;
    }

    double electron_et  = electron_iter->et();
    double electron_phi = electron_iter->phi();

    double sigma_et, sigma_phi ;

    if (controlledUncertainty_) {
      sigma_et  = eleUncertainty.etUncertainty->Eval(electron_et);
      sigma_phi = eleUncertainty.phiUncertainty->Eval(electron_et);
    } 
    else {
      sigma_et = 0.0; //electron_iter->resolutionEt();
      sigma_phi = 0.0; // electron_iter->resolutionPhi();
    }

    if (verbose_ == 3.) {
      std::cout << "electron sigma_et : " << sigma_et << ", electron sigma_phi : " << sigma_phi
		<<  std::endl;}

    if(sigma_et< 0 || sigma_phi< 0)
      edm::LogWarning("PATMHTProducer") << " uncertainties for "  << objectname 
					<<" are (et, phi): " << sigma_et 
					<< "," << sigma_phi <<  " (et,phi): " 
					<< electron_et << "," << electron_phi;
    
    if (uncertaintyScaleFactor_ != 1.0){
      sigma_et  = sigma_et  * uncertaintyScaleFactor_;
      sigma_phi = sigma_phi * uncertaintyScaleFactor_;
    }

    metsig::SigInputObj tmp_electron(objectname,electron_et,electron_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_electron);
    number_of_electrons_ ++; 
    
    //-- Store tower DetId's to be removed from Calo Tower sum later --//
    /*
    const reco::SuperCluster& eleSC = *( electron_iter->superCluster() );
    
    std::vector<DetId> v_eleDetIds = eleSC.getHitsByDetId();
    
    //-- Convert cells to calo towers and add to set --//
    for( std::vector<DetId>::iterator cellId = v_eleDetIds.begin();
         cellId != v_eleDetIds.end();
         cellId++) {

      CaloTowerDetId towerId = caloTowerMap->towerOf(*cellId);
      if (towerId != nullDetId) {
	//std::cout << ">>> electron towerId: " << towerId << std::endl;
	std::pair<std::_Rb_tree_const_iterator<CaloTowerDetId>,bool> p1 = s_clusteredTowers.insert(towerId);
      }
      else
	std::cerr<<"No matching tower found for electron cell!\n";
    }

    */

  }

  return number_of_electrons_; 
}



// --------------------------------------------------
//  Fill Input Vector with Muons
// --------------------------------------------------

double pat::PATMHTProducer::getMuons(edm::Event& iEvent, const edm::EventSetup & iSetup){

  std::string objectname="muon";
  edm::Handle<edm::View<pat::Muon> > muonHandle;
  iEvent.getByLabel(muoLabel_,muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;

  if ( !muonHandle.isValid() ) {
    std::cout << ">>> PATMHTSelector not valid muon Handle!" << std::endl;
    return 0.0;
  }


  double number_of_muons_ = 0.0;

  for(edm::View<pat::Muon>::const_iterator muon_iter = muons.begin(); muon_iter!=muons.end(); ++muon_iter){

    if (muon_iter->pt() < muonPtMin_ || TMath::Abs(muon_iter->eta()) > muonEtaMax_  ) continue; 

    if (verbose_ == 3.) {
      std::cout << "muon pt = " << muon_iter->pt() << " eta : " << muon_iter->eta() <<  std::endl;
    }

    double muon_pt  = muon_iter->pt();
    double muon_phi = muon_iter->phi();

    double sigma_et, sigma_phi ;

    if (controlledUncertainty_) {
       sigma_et  = muonUncertainty.etUncertainty->Eval(muon_pt);
       sigma_phi = muonUncertainty.phiUncertainty->Eval(muon_pt);
    } 
    else {
      sigma_et  = 0.0; //muon_iter->resolutionEt();
      sigma_phi = 0.0; // muon_iter->resolutionPhi();
    }

    if (verbose_ == 3.) {
      std::cout << "muon sigma_et : " << sigma_et
		<< ", muon sigma_phi : " << sigma_phi
		<<  std::endl;}

   if(sigma_et< 0 || sigma_phi< 0)
      edm::LogWarning("PATMHTProducer") << 
	" uncertainties for "  << objectname << " are (et, phi): " << sigma_et << "," <<
	sigma_phi << " (pt,phi): " << muon_pt << "," << muon_phi;

    if (uncertaintyScaleFactor_ != 1.0){
      sigma_et  = sigma_et  * uncertaintyScaleFactor_;
      sigma_phi = sigma_phi * uncertaintyScaleFactor_;
    }

    metsig::SigInputObj tmp_muon(objectname,muon_pt,muon_phi,sigma_et,sigma_phi);
    physobjvector_.push_back(tmp_muon);
    number_of_muons_ ++;

  }// end Muon loop

  return number_of_muons_;
}


//=== Uncertainty Functions ===============================================
void pat::PATMHTProducer::setUncertaintyParameters(){

  // set the various functions here:

  //-- For Et functions, [0]= par_n, [1]=par_s, [2]= par_c ---//
  //-- Ecal Uncertainty Functions ------------------------------------//
  //-- From: FastSimulation/Calorimetry/data/HcalResponse.cfi --//
  //-- Ecal Barrel --//
  ecalEBUncertainty.etUncertainty = new TF1("ecalEBEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  ecalEBUncertainty.etUncertainty->SetParameter(0,0.2);
  ecalEBUncertainty.etUncertainty->SetParameter(1,0.03);
  ecalEBUncertainty.etUncertainty->SetParameter(2,0.005);

  ecalEBUncertainty.phiUncertainty = new TF1("ecalEBphiFunc","[0]*x",1);
  ecalEBUncertainty.phiUncertainty->SetParameter(0,0.0174);

  //-- Ecal Endcap --//
  ecalEEUncertainty.etUncertainty = new TF1("ecalEEEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  ecalEEUncertainty.etUncertainty->SetParameter(0,0.2);
  ecalEEUncertainty.etUncertainty->SetParameter(1,0.03);
  ecalEEUncertainty.etUncertainty->SetParameter(2,0.005);

  ecalEEUncertainty.phiUncertainty = new TF1("ecalEEphiFunc","[0]*x",1);
  ecalEEUncertainty.phiUncertainty->SetParameter(0,0.087);

  //-- Hcal Uncertainty Functions --------------------------------------//
  //-- From: FastSimulation/Calorimetry/data/HcalResponse.cfi --//
  //-- Hcal Barrel --//
  hcalHBUncertainty.etUncertainty = new TF1("hcalHBEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  hcalHBUncertainty.etUncertainty->SetParameter(0,0.);
  hcalHBUncertainty.etUncertainty->SetParameter(1,1.22);
  hcalHBUncertainty.etUncertainty->SetParameter(2,0.05);

  hcalHBUncertainty.phiUncertainty = new TF1("ecalHBphiFunc","[0]*x",1);
  hcalHBUncertainty.phiUncertainty->SetParameter(0,0.087);

  //-- Hcal Endcap --//
  hcalHEUncertainty.etUncertainty = new TF1("hcalHEEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  hcalHEUncertainty.etUncertainty->SetParameter(0,0.);
  hcalHEUncertainty.etUncertainty->SetParameter(1,1.3);
  hcalHEUncertainty.etUncertainty->SetParameter(2,0.05);

  hcalHEUncertainty.phiUncertainty = new TF1("ecalHEphiFunc","[0]*x",1);
  hcalHEUncertainty.phiUncertainty->SetParameter(0,0.087);

  //-- Hcal Outer --//
  hcalHOUncertainty.etUncertainty = new TF1("hcalHOEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  hcalHOUncertainty.etUncertainty->SetParameter(0,0.);
  hcalHOUncertainty.etUncertainty->SetParameter(1,1.82);
  hcalHOUncertainty.etUncertainty->SetParameter(2,0.09);

  hcalHOUncertainty.phiUncertainty = new TF1("ecalHOphiFunc","[0]*x",1);
  hcalHOUncertainty.phiUncertainty->SetParameter(0,0.087);

  //-- Hcal Forward --//
  hcalHFUncertainty.etUncertainty = new TF1("hcalHFEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  hcalHFUncertainty.etUncertainty->SetParameter(0,0.);
  hcalHFUncertainty.etUncertainty->SetParameter(1,1.82);
  hcalHFUncertainty.etUncertainty->SetParameter(2,0.09);

  hcalHFUncertainty.phiUncertainty = new TF1("ecalHFphiFunc","[0]*x",1);
  hcalHFUncertainty.phiUncertainty->SetParameter(0,0.174);

  //--- Jet Uncertainty Functions --------------------------------------//
  jetUncertainty.etUncertainty = new TF1("jetEtFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  //-- values from PTDR 1, ch 11.4 --//
  jetUncertainty.etUncertainty->SetParameter(0, jetEtUncertaintyParameter0_);
  jetUncertainty.etUncertainty->SetParameter(1, jetEtUncertaintyParameter1_);
  jetUncertainty.etUncertainty->SetParameter(2, jetEtUncertaintyParameter2_);


  //-- phi value from our own fits --//
  //jetUncertainty.phiUncertainty = new TF1("jetPhiFunc","[0]*x",1);
  //jetUncertainty.phiUncertainty->SetParameter(0, jetPhiUncertaintyParameter0_);

  //-- phi Functions and values from 
  // http://indico.cern.ch/getFile.py/access?contribId=9&sessionId=0&resId=0&materialId=slides&confId=46394
  jetUncertainty.phiUncertainty = new TF1("jetPhiFunc","x*sqrt(([0]*[0]/(x*x))+([1]*[1]/x)+([2]*[2]))",3);
  jetUncertainty.phiUncertainty->SetParameter(0, jetPhiUncertaintyParameter0_);
  jetUncertainty.phiUncertainty->SetParameter(1, jetPhiUncertaintyParameter1_);
  jetUncertainty.phiUncertainty->SetParameter(2, jetPhiUncertaintyParameter2_);
  


  //-- Jet corrections are assumed not to have an error --//
  /*jetCorrUncertainty.etUncertainty = new TF1("jetCorrEtFunc","[0]*x",1);
  jetCorrUncertainty.etUncertainty->SetParameter(0,0.0);
  jetCorrUncertainty.phiUncertainty = new TF1("jetCorrPhiFunc","[0]*x",1);
  jetCorrUncertainty.phiUncertainty->SetParameter(0,0.0*(3.14159/180.));*/


  //--- Electron Uncertainty Functions ---------------------------------//
  // completely ambiguious values for electron-like jets...
  // the egamma group keeps track of these here:
  // https://twiki.cern.ch/twiki/bin/view/CMS/EgammaCMSSWVal
  // electron resolution in energy is around 3.4%, measured for 10 < pT < 50 at realistic events with pile-up.
  
  eleUncertainty.etUncertainty = new TF1("eleEtFunc","[0] * x",1);
  //  eleUncertainty.etUncertainty->SetParameter(0,0.034); 
  eleUncertainty.etUncertainty->SetParameter(0, eleEtUncertaintyParameter0_); 


  eleUncertainty.phiUncertainty = new TF1("elePhiFunc","[0] * x",1);
  //  eleUncertainty.phiUncertainty->SetParameter(0,1*(3.14159/180.));
  eleUncertainty.phiUncertainty->SetParameter(0, elePhiUncertaintyParameter0_);

  //--- Muon Uncertainty Functions ------------------------------------//
  // and ambiguious values for the muons...
  
  muonUncertainty.etUncertainty = new TF1("muonEtFunc","[0] * x",1);
  //  muonUncertainty.etUncertainty->SetParameter(0,0.01);
  muonUncertainty.etUncertainty->SetParameter(0, muonEtUncertaintyParameter0_);
  muonUncertainty.phiUncertainty = new TF1("muonPhiFunc","[0] * x",1);
  //  muonUncertainty.phiUncertainty->SetParameter(0,1*(3.14159/180.));
  muonUncertainty.phiUncertainty->SetParameter(0, muonPhiUncertaintyParameter0_);

  //-- Muon calo deposites are assumed not to have an error --//
  /*muonCorrUncertainty.etUncertainty = new TF1("muonCorrEtFunc","[0] * x",1);
  muonCorrUncertainty.etUncertainty->SetParameter(0,0.0);
  muonCorrUncertainty.phiUncertainty = new TF1("muonCorrPhiFunc","[0] * x",1);
  muonCorrUncertainty.phiUncertainty->SetParameter(0,0.0*(3.14159/180.)); */
 
}


using namespace pat; 
DEFINE_FWK_MODULE(PATMHTProducer);


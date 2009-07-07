#include "DQMOffline/JetMET/interface/ECALRecHitAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

// author: Bobby Scurlock, University of Florida
// first version 11/20/2006

#define DEBUG(X) { if (debug_) { cout << X << endl; } }

ECALRecHitAnalyzer::ECALRecHitAnalyzer(const edm::ParameterSet& iConfig)
{
  
  // Retrieve Information from the Configuration File
  EBRecHitsLabel_  = iConfig.getParameter<edm::InputTag>("EBRecHitsLabel");
  EERecHitsLabel_    = iConfig.getParameter<edm::InputTag>("EERecHitsLabel");
  FolderName_           = iConfig.getUntrackedParameter<std::string>("FolderName");
  debug_             = iConfig.getParameter<bool>("Debug");


}

void ECALRecHitAnalyzer::endJob() {

} 

//void ECALRecHitAnalyzer::beginJob(const edm::EventSetup& iSetup){
void ECALRecHitAnalyzer::beginRun(const edm::Run& iRun,const edm::EventSetup& iSetup){
  CurrentEvent = -1;
  // Book the Histograms
  // Fill the geometry histograms
  BookHistos();
  FillGeometry(iSetup);
}

void ECALRecHitAnalyzer::BookHistos()
{
  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  
  if (dbe_) {
    
    // Book Geometry Histograms
    dbe_->setCurrentFolder(FolderName_+"/geometry");
    

    // ECAL barrel
  me["hEB_ieta_iphi_etaMap"] = dbe_->book2D("hEB_ieta_iphi_etaMap","", 171, -85, 86, 360, 1, 361);
  me["hEB_ieta_iphi_phiMap"] = dbe_->book2D("hEB_ieta_iphi_phiMap","", 171, -85, 86, 360, 1, 361);
  me["hEB_ieta_detaMap"] = dbe_->book1D("hEB_ieta_detaMap","", 171, -85, 86);
  me["hEB_ieta_dphiMap"] = dbe_->book1D("hEB_ieta_dphiMap","", 171, -85, 86);
  // ECAL +endcap
  me["hEEpZ_ix_iy_irMap"] = dbe_->book2D("hEEpZ_ix_iy_irMap","", 100,1,101, 100,1,101);
  me["hEEpZ_ix_iy_xMap"] = dbe_->book2D("hEEpZ_ix_iy_xMap","", 100,1,101, 100,1,101);
  me["hEEpZ_ix_iy_yMap"] = dbe_->book2D("hEEpZ_ix_iy_yMap","", 100,1,101, 100,1,101);
  me["hEEpZ_ix_iy_zMap"] = dbe_->book2D("hEEpZ_ix_iy_zMap","", 100,1,101, 100,1,101);
  me["hEEpZ_ix_iy_dxMap"] = dbe_->book2D("hEEpZ_ix_iy_dxMap","", 100,1,101, 100,1,101);  
  me["hEEpZ_ix_iy_dyMap"] = dbe_->book2D("hEEpZ_ix_iy_dyMap","", 100,1,101, 100,1,101);
  // ECAL -endcap
  me["hEEmZ_ix_iy_irMap"] = dbe_->book2D("hEEmZ_ix_iy_irMap","", 100,1,101, 100,1,101);
  me["hEEmZ_ix_iy_xMap"] = dbe_->book2D("hEEmZ_ix_iy_xMap","", 100,1,101, 100,1,101);
  me["hEEmZ_ix_iy_yMap"] = dbe_->book2D("hEEmZ_ix_iy_yMap","", 100,1,101, 100,1,101);
  me["hEEmZ_ix_iy_zMap"] = dbe_->book2D("hEEmZ_ix_iy_zMap","", 100,1,101, 100,1,101);
  me["hEEmZ_ix_iy_dxMap"] = dbe_->book2D("hEEmZ_ix_iy_dxMap","", 100,1,101, 100,1,101);  
  me["hEEmZ_ix_iy_dyMap"] = dbe_->book2D("hEEmZ_ix_iy_dyMap","", 100,1,101, 100,1,101);

  // Initialize bins for geometry to -999 because z = 0 is a valid entry 
  for (int i=1; i<=100; i++)
    for (int j=1; j<=100; j++)
      {
	me["hEEpZ_ix_iy_irMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_xMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_yMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_zMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_dxMap"]->setBinContent(i,j,-999);
	me["hEEpZ_ix_iy_dyMap"]->setBinContent(i,j,-999);

	me["hEEmZ_ix_iy_irMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_xMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_yMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_zMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_dxMap"]->setBinContent(i,j,-999);
	me["hEEmZ_ix_iy_dyMap"]->setBinContent(i,j,-999);
      }

  for (int i=1; i<=171; i++)
    {
      me["hEB_ieta_detaMap"]->setBinContent(i,-999);
      me["hEB_ieta_dphiMap"]->setBinContent(i,-999);
      for (int j=1; j<=360; j++)
	{
	  me["hEB_ieta_iphi_etaMap"]->setBinContent(i,j,-999);
	  me["hEB_ieta_iphi_phiMap"]->setBinContent(i,j,-999);
	}
    }

  // Book Data Histograms
  dbe_->setCurrentFolder(FolderName_);

  me["hECAL_Nevents"]          = dbe_->book1D("hECAL_Nevents","",1,0,1); 
  

  // Energy Histograms by logical index
  me["hEEpZ_energy_ix_iy"] = dbe_->book2D("hEEpZ_energy_ix_iy","", 100,1,101, 100,1,101);
  me["hEEmZ_energy_ix_iy"] = dbe_->book2D("hEEmZ_energy_ix_iy","", 100,1,101, 100,1,101);
  me["hEB_energy_ieta_iphi"] = dbe_->book2D("hEB_energy_ieta_iphi","", 171, -85, 86, 360, 1, 361);   

  me["hEEpZ_Minenergy_ix_iy"] = dbe_->book2D("hEEpZ_Minenergy_ix_iy","", 100,1,101, 100,1,101);
  me["hEEmZ_Minenergy_ix_iy"] = dbe_->book2D("hEEmZ_Minenergy_ix_iy","", 100,1,101, 100,1,101);
  me["hEB_Minenergy_ieta_iphi"] = dbe_->book2D("hEB_Minenergy_ieta_iphi","", 171, -85, 86, 360, 1, 361);   

  me["hEEpZ_Maxenergy_ix_iy"] = dbe_->book2D("hEEpZ_Maxenergy_ix_iy","", 100,1,101, 100,1,101);
  me["hEEmZ_Maxenergy_ix_iy"] = dbe_->book2D("hEEmZ_Maxenergy_ix_iy","", 100,1,101, 100,1,101);
  me["hEB_Maxenergy_ieta_iphi"] = dbe_->book2D("hEB_Maxenergy_ieta_iphi","", 171, -85, 86, 360, 1, 361);   

  // need to initialize those
  for (int i=1; i<=171; i++)
    for (int j=1; j<=360; j++)
      {
	me["hEB_Maxenergy_ieta_iphi"]->setBinContent(i,j,-999);
	me["hEB_Minenergy_ieta_iphi"]->setBinContent(i,j,14000);
      }
  for (int i=1; i<=100; i++)
    for (int j=1; j<=100; j++)
      {
	me["hEEpZ_Maxenergy_ix_iy"]->setBinContent(i,j,-999);
	me["hEEpZ_Minenergy_ix_iy"]->setBinContent(i,j,14000);
	me["hEEmZ_Maxenergy_ix_iy"]->setBinContent(i,j,-999);
	me["hEEmZ_Minenergy_ix_iy"]->setBinContent(i,j,14000);
      }
  

  // Occupancy Histograms by logical index
  me["hEEpZ_Occ_ix_iy"] = dbe_->book2D("hEEpZ_Occ_ix_iy","", 100,1,101, 100,1,101);  
  me["hEEmZ_Occ_ix_iy"] = dbe_->book2D("hEEmZ_Occ_ix_iy","", 100,1,101, 100,1,101);  
  me["hEB_Occ_ieta_iphi"] = dbe_->book2D("hEB_Occ_ieta_iphi","",171, -85, 86, 360, 1, 361);   

  // Integrated Histograms
  if(finebinning_)
    {
      me["hEEpZ_energyvsir"] = dbe_->book2D("hEEpZ_energyvsir","", 100,1,101, 20110,-10,201);
      me["hEEmZ_energyvsir"] = dbe_->book2D("hEEmZ_energyvsir","", 100,1,101, 20110,-10,201);
      me["hEB_energyvsieta"] = dbe_->book2D("hEB_energyvsieta","", 171, -85, 86, 20110, -10, 201);   
      
      me["hEEpZ_Maxenergyvsir"] = dbe_->book2D("hEEpZ_Maxenergyvsir","", 100,1,101, 20110,-10,201);
      me["hEEmZ_Maxenergyvsir"] = dbe_->book2D("hEEmZ_Maxenergyvsir","", 100,1,101, 20110,-10,201);
      me["hEB_Maxenergyvsieta"] = dbe_->book2D("hEB_Maxenergyvsieta","", 171, -85, 86, 20110, -10, 201);   
      
      me["hEEpZ_Minenergyvsir"] = dbe_->book2D("hEEpZ_Minenergyvsir","", 100,1,101, 20110,-10,201);
      me["hEEmZ_Minenergyvsir"] = dbe_->book2D("hEEmZ_Minenergyvsir","", 100,1,101, 20110,-10,201);
      me["hEB_Minenergyvsieta"] = dbe_->book2D("hEB_Minenergyvsieta","", 171, -85, 86, 20110, -10, 201);   
      
      me["hEEpZ_SETvsir"] = dbe_->book2D("hEEpZ_SETvsir","", 50,1,51, 20010,0,201);
      me["hEEmZ_SETvsir"] = dbe_->book2D("hEEmZ_SETvsir","", 50,1,51, 20010,0,201);
      me["hEB_SETvsieta"] = dbe_->book2D("hEB_SETvsieta","", 171, -85, 86, 20010, 0, 201);   
      
      me["hEEpZ_METvsir"] = dbe_->book2D("hEEpZ_METvsir","", 50,1,51, 20010,0,201);
      me["hEEmZ_METvsir"] = dbe_->book2D("hEEmZ_METvsir","", 50,1,51, 20010,0,201);
      me["hEB_METvsieta"] = dbe_->book2D("hEB_METvsieta","", 171, -85, 86, 20010, 0, 201);   
      
      me["hEEpZ_METPhivsir"] = dbe_->book2D("hEEpZ_METPhivsir","", 50,1,51, 80,-4,4);
      me["hEEmZ_METPhivsir"] = dbe_->book2D("hEEmZ_METPhivsir","", 50,1,51, 80,-4,4);
      me["hEB_METPhivsieta"] = dbe_->book2D("hEB_METPhivsieta","", 171, -85, 86, 80,-4,4);   
      
      me["hEEpZ_MExvsir"] = dbe_->book2D("hEEpZ_MExvsir","", 50,1,51, 10010,-50,51);
      me["hEEmZ_MExvsir"] = dbe_->book2D("hEEmZ_MExvsir","", 50,1,51, 10010,-50,51);
      me["hEB_MExvsieta"] = dbe_->book2D("hEB_MExvsieta","", 171, -85, 86, 10010,-50,51);   
      
      me["hEEpZ_MEyvsir"] = dbe_->book2D("hEEpZ_MEyvsir","", 50,1,51, 10010,-50,51);
      me["hEEmZ_MEyvsir"] = dbe_->book2D("hEEmZ_MEyvsir","", 50,1,51, 10010,-50,51);
      me["hEB_MEyvsieta"] = dbe_->book2D("hEB_MEyvsieta","", 171, -85, 86, 10010,-50,51);   
      
      me["hEEpZ_Occvsir"] = dbe_->book2D("hEEpZ_Occvsir","", 50,1,51, 1000,0,1000);
      me["hEEmZ_Occvsir"] = dbe_->book2D("hEEmZ_Occvsir","", 50,1,51, 1000,0,1000);
      me["hEB_Occvsieta"] = dbe_->book2D("hEB_Occvsieta","", 171, -85, 86, 400,0,400);   
    }
  else 
    {
      me["hEEpZ_energyvsir"] = dbe_->book2D("hEEpZ_energyvsir","", 100,1,101, 510,-10,100);
      me["hEEmZ_energyvsir"] = dbe_->book2D("hEEmZ_energyvsir","", 100,1,101, 510,-10,100);
      me["hEB_energyvsieta"] = dbe_->book2D("hEB_energyvsieta","", 171, -85, 86, 510, -10, 100);
      
      me["hEEpZ_Maxenergyvsir"] = dbe_->book2D("hEEpZ_Maxenergyvsir","", 100,1,101, 510,-10,100);
      me["hEEmZ_Maxenergyvsir"] = dbe_->book2D("hEEmZ_Maxenergyvsir","", 100,1,101, 510,-10,100);
      me["hEB_Maxenergyvsieta"] = dbe_->book2D("hEB_Maxenergyvsieta","", 171, -85, 86, 510, -10, 100);

      me["hEEpZ_Minenergyvsir"] = dbe_->book2D("hEEpZ_Minenergyvsir","", 100,1,101, 510,-10,100);
      me["hEEmZ_Minenergyvsir"] = dbe_->book2D("hEEmZ_Minenergyvsir","", 100,1,101, 510,-10,100);
      me["hEB_Minenergyvsieta"] = dbe_->book2D("hEB_Minenergyvsieta","", 171, -85, 86, 510, -10, 100);

      me["hEEpZ_SETvsir"] = dbe_->book2D("hEEpZ_SETvsir","", 50,1,51, 510,0,100);
      me["hEEmZ_SETvsir"] = dbe_->book2D("hEEmZ_SETvsir","", 50,1,51, 510,0,100);
      me["hEB_SETvsieta"] = dbe_->book2D("hEB_SETvsieta","", 171, -85, 86, 510, 0, 100);

      me["hEEpZ_METvsir"] = dbe_->book2D("hEEpZ_METvsir","", 50,1,51, 510,0,100);
      me["hEEmZ_METvsir"] = dbe_->book2D("hEEmZ_METvsir","", 50,1,51, 510,0,100);
      me["hEB_METvsieta"] = dbe_->book2D("hEB_METvsieta","", 171, -85, 86, 510, 0, 100);

      me["hEEpZ_METPhivsir"] = dbe_->book2D("hEEpZ_METPhivsir","", 50,1,51, 80,-4,4);
      me["hEEmZ_METPhivsir"] = dbe_->book2D("hEEmZ_METPhivsir","", 50,1,51, 80,-4,4);
      me["hEB_METPhivsieta"] = dbe_->book2D("hEB_METPhivsieta","", 171, -85, 86, 80,-4,4);

      me["hEEpZ_MExvsir"] = dbe_->book2D("hEEpZ_MExvsir","", 50,1,51, 510,-50,51);
      me["hEEmZ_MExvsir"] = dbe_->book2D("hEEmZ_MExvsir","", 50,1,51, 510,-50,51);
      me["hEB_MExvsieta"] = dbe_->book2D("hEB_MExvsieta","", 171, -85, 86, 510,-50,51);

      me["hEEpZ_MEyvsir"] = dbe_->book2D("hEEpZ_MEyvsir","", 50,1,51, 510,-50,51);
      me["hEEmZ_MEyvsir"] = dbe_->book2D("hEEmZ_MEyvsir","", 50,1,51, 510,-50,51);
      me["hEB_MEyvsieta"] = dbe_->book2D("hEB_MEyvsieta","", 171, -85, 86, 510,-50,51);

      me["hEEpZ_Occvsir"] = dbe_->book2D("hEEpZ_Occvsir","", 50,1,51, 1000,0,1000);
      me["hEEmZ_Occvsir"] = dbe_->book2D("hEEmZ_Occvsir","", 50,1,51, 1000,0,1000);
      me["hEB_Occvsieta"] = dbe_->book2D("hEB_Occvsieta","", 171, -85, 86, 400,0,400);

    }



}
}

void ECALRecHitAnalyzer::FillGeometry(const edm::EventSetup& iSetup)
{
  // Fill geometry histograms
  using namespace edm;
  //int b=0;
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry cG = *pG;
  
  //----Fill Ecal Barrel----//
  const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  int n=0;
  std::vector<DetId> EBids=EBgeom->getValidDetIds(DetId::Ecal, 1);
  for (std::vector<DetId>::iterator i=EBids.begin(); i!=EBids.end(); i++) {
    n++;
    const CaloCellGeometry* cell=EBgeom->getGeometry(*i);
    //GlobalPoint p = cell->getPosition();
    
    EBDetId EcalID(i->rawId());
    
    int Crystal_ieta = EcalID.ieta();
    int Crystal_iphi = EcalID.iphi();
    double Crystal_eta = cell->getPosition().eta();
    double Crystal_phi = cell->getPosition().phi();
    me["hEB_ieta_iphi_etaMap"]->setBinContent(Crystal_ieta+86, Crystal_iphi, Crystal_eta);
    me["hEB_ieta_iphi_phiMap"]->setBinContent(Crystal_ieta+86, Crystal_iphi, (Crystal_phi*180/M_PI) );
    
    DEBUG( " Crystal " << n );
    DEBUG( "  ieta, iphi = " << Crystal_ieta << ", " << Crystal_iphi);
    DEBUG( "   eta,  phi = " << cell->getPosition().eta() << ", " << cell->getPosition().phi());
    DEBUG( " " );
    
  }
  //----Fill Ecal Endcap----------//
  const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  n=0;
  std::vector<DetId> EEids=EEgeom->getValidDetIds(DetId::Ecal, 2);
  for (std::vector<DetId>::iterator i=EEids.begin(); i!=EEids.end(); i++) {
    n++;
    const CaloCellGeometry* cell=EEgeom->getGeometry(*i);
    //GlobalPoint p = cell->getPosition();
    EEDetId EcalID(i->rawId());
    int Crystal_zside = EcalID.zside();
    int Crystal_ix = EcalID.ix();
    int Crystal_iy = EcalID.iy();
    Float_t ix_ = Crystal_ix-50.5;
    Float_t iy_ = Crystal_iy-50.5;
    Int_t ir = (Int_t)sqrt(ix_*ix_ + iy_*iy_);

    //double Crystal_eta = cell->getPosition().eta();
    //double Crystal_phi = cell->getPosition().phi();
    double Crystal_x = cell->getPosition().x();
    double Crystal_y = cell->getPosition().y();
    double Crystal_z = cell->getPosition().z();
    // ECAL -endcap
    if (Crystal_zside == -1)
      {
	me["hEEmZ_ix_iy_irMap"]->setBinContent(Crystal_ix, Crystal_iy, ir);
	me["hEEmZ_ix_iy_xMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_x);
	me["hEEmZ_ix_iy_yMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_y);
	me["hEEmZ_ix_iy_zMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_z);
      }
    // ECAL +endcap
    if (Crystal_zside == 1)
      {
	me["hEEpZ_ix_iy_irMap"]->setBinContent(Crystal_ix, Crystal_iy, ir);
	me["hEEpZ_ix_iy_xMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_x);
	me["hEEpZ_ix_iy_yMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_y);
	me["hEEpZ_ix_iy_zMap"]->setBinContent(Crystal_ix, Crystal_iy, Crystal_z);
      }

      DEBUG( " Crystal " << n );
      DEBUG( "  side = " << Crystal_zside );
      DEBUG("   ix, iy = " << Crystal_ix << ", " << Crystal_iy);
      DEBUG("    x,  y = " << Crystal_x << ", " << Crystal_y);;
      DEBUG( " " );

  }
 
  //-------Set the cell size for each (ieta, iphi) bin-------//
  double currentLowEdge_eta = 0;
  //double currentHighEdge_eta = 0;
  for (int ieta=1; ieta<=85 ; ieta++)
    {
      int ieta_ = 86 + ieta;
      
      double eta = me["hEB_ieta_iphi_etaMap"]->getBinContent(ieta_, 1);
      double etam1 = -999;
      
      if (ieta==1) 
	etam1 = me["hEB_ieta_iphi_etaMap"]->getBinContent(85, 1);
      else 
	etam1 = me["hEB_ieta_iphi_etaMap"]->getBinContent(ieta_ - 1, 1);

      //double phi = me["hEB_ieta_iphi_phiMap"]->getBinContent(ieta_, 1);
      double deta = fabs( eta - etam1 );
      double dphi = fabs( me["hEB_ieta_iphi_phiMap"]->getBinContent(ieta_, 1) - me["hEB_ieta_iphi_phiMap"]->getBinContent(ieta_, 2) );
          
      currentLowEdge_eta += deta;
      me["hEB_ieta_detaMap"]->setBinContent(ieta_, deta); // positive rings
      me["hEB_ieta_dphiMap"]->setBinContent(ieta_, dphi); // positive rings
      me["hEB_ieta_detaMap"]->setBinContent(86-ieta, deta); // negative rings
      me["hEB_ieta_dphiMap"]->setBinContent(86-ieta, dphi); // negative rings
    }
}



void ECALRecHitAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  CurrentEvent++;
  DEBUG( "Event: " << CurrentEvent);
  WriteECALRecHits( iEvent, iSetup );
  me["hECAL_Nevents"]->Fill(0.5);
}

void ECALRecHitAnalyzer::WriteECALRecHits(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<EBRecHitCollection> EBRecHits;
  edm::Handle<EERecHitCollection> EERecHits;
  iEvent.getByLabel( EBRecHitsLabel_, EBRecHits );
  iEvent.getByLabel( EERecHitsLabel_, EERecHits );
  DEBUG( "Got ECALRecHits");

  /*
  edm::Handle<reco::CandidateCollection> to;
  iEvent.getByLabel( "caloTowers", to );
  const CandidateCollection *towers = (CandidateCollection *)to.product();
  reco::CandidateCollection::const_iterator tower = towers->begin();
  edm::Ref<CaloTowerCollection> towerRef = tower->get<CaloTowerRef>();
  const CaloTowerCollection *towerCollection = towerRef.product();
  CaloTowerCollection::const_iterator calotower = towerCollection->begin();
  DEBUG( "Got Towers");    
  DEBUG( "tower size = " << towerCollection->size());
  */

  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry cG = *pG;
  //const CaloSubdetectorGeometry* EBgeom=cG.getSubdetectorGeometry(DetId::Ecal,1);
  //const CaloSubdetectorGeometry* EEgeom=cG.getSubdetectorGeometry(DetId::Ecal,2);
  DEBUG( "Got Geometry");

  TLorentzVector vEBMET_EtaRing[171];
  int EBActiveRing[171];
  int EBNActiveCells[171];
  double EBSET_EtaRing[171];
  double EBMaxEnergy_EtaRing[171];
  double EBMinEnergy_EtaRing[171];
  double EBenergy_EtaRing[171];

  for (int i=0; i<171; i++)
    {
      EBActiveRing[i] = 0;
      EBNActiveCells[i] = 0;
      EBSET_EtaRing[i] = 0.0;
      EBMaxEnergy_EtaRing[i] = -999; 
      EBMinEnergy_EtaRing[i] = 14E3; 
      EBenergy_EtaRing[i] = 0.0;
    }

  edm::LogInfo("OutputInfo") << "Looping over EB" << endl;

  EBRecHitCollection::const_iterator ebrechit;
  //int nEBrechit = 0;

  for (ebrechit = EBRecHits->begin(); ebrechit != EBRecHits->end(); ebrechit++) {
    
    EBDetId det = ebrechit->id();
    double Energy = ebrechit->energy();
    Int_t ieta = det.ieta();
    Int_t iphi = det.iphi();
    int EtaRing = 85 + ieta; // this counts from 0
    double eta = me["hEB_ieta_iphi_etaMap"]->getBinContent(EtaRing+1,iphi);
    double phi = me["hEB_ieta_iphi_phiMap"]->getBinContent(EtaRing+1,iphi);
    double theta = 2*TMath::ATan(exp(-1*eta));
    double ET = Energy*TMath::Sin(theta);
    TLorentzVector v_;

    if (Energy>EBMaxEnergy_EtaRing[EtaRing]) EBMaxEnergy_EtaRing[EtaRing] = Energy;
    if (Energy<EBMinEnergy_EtaRing[EtaRing]) EBMinEnergy_EtaRing[EtaRing] = Energy;
    
    if (Energy>0)
      {
	EBActiveRing[EtaRing] = 1;
	EBNActiveCells[EtaRing]++;
	EBSET_EtaRing[EtaRing]+=ET;
	v_.SetPtEtaPhiE(ET, 0, phi, ET);
	vEBMET_EtaRing[EtaRing]-=v_;
	EBenergy_EtaRing[EtaRing]+=Energy;
	me["hEB_Occ_ieta_iphi"]->Fill(ieta, iphi);

      }
    
    me["hEB_energy_ieta_iphi"]->Fill(ieta, iphi, Energy);
    if (Energy>me["hEB_Maxenergy_ieta_iphi"]->getBinContent(EtaRing+1, iphi))
      me["hEB_Maxenergy_ieta_iphi"]->setBinContent(EtaRing+1, iphi, Energy);
    if (Energy<me["hEB_Minenergy_ieta_iphi"]->getBinContent(EtaRing+1, iphi))
      me["hEB_Minenergy_ieta_iphi"]->setBinContent(EtaRing+1, iphi, Energy);


  } // loop over EB

  for (int iEtaRing = 0; iEtaRing < 171; iEtaRing++)
    {
      me["hEB_Minenergyvsieta"]->Fill(iEtaRing-85, EBMinEnergy_EtaRing[iEtaRing]);
      me["hEB_Maxenergyvsieta"]->Fill(iEtaRing-85, EBMaxEnergy_EtaRing[iEtaRing]);

      if (EBActiveRing[iEtaRing])
	{
	  me["hEB_METvsieta"]->Fill(iEtaRing-85, vEBMET_EtaRing[iEtaRing].Pt());
	  me["hEB_METPhivsieta"]->Fill(iEtaRing-85, vEBMET_EtaRing[iEtaRing].Phi());
	  me["hEB_MExvsieta"]->Fill(iEtaRing-85, vEBMET_EtaRing[iEtaRing].Px());
	  me["hEB_MEyvsieta"]->Fill(iEtaRing-85, vEBMET_EtaRing[iEtaRing].Py());
	  me["hEB_SETvsieta"]->Fill(iEtaRing-85, EBSET_EtaRing[iEtaRing]);
	  me["hEB_Occvsieta"]->Fill(iEtaRing-85, EBNActiveCells[iEtaRing]);
	  me["hEB_energyvsieta"]->Fill(iEtaRing-85, EBenergy_EtaRing[iEtaRing]);
	}
    }


  TLorentzVector vEEpZMET_EtaRing[101];
  int EEpZActiveRing[101];
  int EEpZNActiveCells[101];
  double EEpZSET_EtaRing[101];
  double EEpZMaxEnergy_EtaRing[101];
  double EEpZMinEnergy_EtaRing[101];

  TLorentzVector vEEmZMET_EtaRing[101];
  int EEmZActiveRing[101];
  int EEmZNActiveCells[101];
  double EEmZSET_EtaRing[101];
  double EEmZMaxEnergy_EtaRing[101];
  double EEmZMinEnergy_EtaRing[101];

  for (int i=0;i<101; i++)
    {
      EEpZActiveRing[i] = 0;
      EEpZNActiveCells[i] = 0;
      EEpZSET_EtaRing[i] = 0.0;
      EEpZMaxEnergy_EtaRing[i] = -999;
      EEpZMinEnergy_EtaRing[i] = 14E3;

      EEmZActiveRing[i] = 0;
      EEmZNActiveCells[i] = 0;
      EEmZSET_EtaRing[i] = 0.0;
      EEmZMaxEnergy_EtaRing[i] = -999;
      EEmZMinEnergy_EtaRing[i] = 14E3;
    }

  edm::LogInfo("OutputInfo") << "Looping over EE" << endl;
  EERecHitCollection::const_iterator eerechit;
  //int nEErechit = 0;
  for (eerechit = EERecHits->begin(); eerechit != EERecHits->end(); eerechit++) {
    
    EEDetId det = eerechit->id();
    double Energy = eerechit->energy();
    Int_t ix = det.ix();
    Int_t iy = det.iy();
    //Float_t ix_ = (Float_t)-999;
    //Float_t iy_ = (Float_t)-999;
    Int_t ir = -999;
    //    edm::LogInfo("OutputInfo") << ix << " " << iy << " " << ix_ << " " << iy_ << " " << ir << endl;

    double x = -999;
    double y = -999;
    double z = -999;
    double eta = -999;
    double theta = -999;
    double phi = -999;

    int Crystal_zside = det.zside();

    if (Crystal_zside == -1)
      {
	ir = (Int_t)me["hEEmZ_ix_iy_irMap"]->getBinContent(ix,iy);
	x = me["hEEmZ_ix_iy_xMap"]->getBinContent(ix,iy);
	y = me["hEEmZ_ix_iy_yMap"]->getBinContent(ix,iy);
	z = me["hEEmZ_ix_iy_zMap"]->getBinContent(ix,iy);
      }
    if (Crystal_zside == 1)
      {
	ir = (Int_t)me["hEEpZ_ix_iy_irMap"]->getBinContent(ix,iy);
	x = me["hEEpZ_ix_iy_xMap"]->getBinContent(ix,iy);
	y = me["hEEpZ_ix_iy_yMap"]->getBinContent(ix,iy);
	z = me["hEEpZ_ix_iy_zMap"]->getBinContent(ix,iy);
      }
    TVector3 pos_vector(x,y,z);
    phi = pos_vector.Phi();
    theta = pos_vector.Theta();
    eta = pos_vector.Eta();
    double ET = Energy*TMath::Sin(theta);
    TLorentzVector v_;


    if (Crystal_zside == -1)
      {
	if (Energy>0)
	  {
	    EEmZActiveRing[ir] = 1;
	    EEmZNActiveCells[ir]++;
	    EEmZSET_EtaRing[ir]+=ET;
	    v_.SetPtEtaPhiE(ET,0,phi,ET);
	    vEEmZMET_EtaRing[ir]-=v_;
	    me["hEEmZ_Occ_ix_iy"]->Fill(ix, iy);
	  }
	me["hEEmZ_energyvsir"]->Fill(ir, Energy);
	me["hEEmZ_energy_ix_iy"]->Fill(ix, iy, Energy);

	if (Energy>EEmZMaxEnergy_EtaRing[ir]) EEmZMaxEnergy_EtaRing[ir] = Energy;
	if (Energy<EEmZMinEnergy_EtaRing[ir]) EEmZMinEnergy_EtaRing[ir] = Energy;

	if (Energy>me["hEEmZ_Maxenergy_ix_iy"]->getBinContent(ix,iy))
	  me["hEEmZ_Maxenergy_ix_iy"]->setBinContent(ix,iy, Energy);
	if (Energy<me["hEEmZ_Minenergy_ix_iy"]->getBinContent(ix,iy))
	  me["hEEmZ_Minenergy_ix_iy"]->setBinContent(ix,iy, Energy);
      }
    if (Crystal_zside == 1)
      {
	if (Energy>0)
	  {
	    EEpZActiveRing[ir] = 1;
	    EEpZNActiveCells[ir]++;
	    EEpZSET_EtaRing[ir]+=ET;
	    v_.SetPtEtaPhiE(ET,0,phi,ET);
	    vEEpZMET_EtaRing[ir]-=v_;
	    me["hEEpZ_Occ_ix_iy"]->Fill(ix, iy);
	  }
	me["hEEpZ_energyvsir"]->Fill(ir, Energy);
	me["hEEpZ_energy_ix_iy"]->Fill(ix, iy, Energy);

	if (Energy>EEpZMaxEnergy_EtaRing[ir]) EEpZMaxEnergy_EtaRing[ir] = Energy;
	if (Energy<EEpZMinEnergy_EtaRing[ir]) EEpZMinEnergy_EtaRing[ir] = Energy;
	if (Energy>me["hEEpZ_Maxenergy_ix_iy"]->getBinContent(ix,iy))
	  me["hEEpZ_Maxenergy_ix_iy"]->setBinContent(ix,iy, Energy);
	if (Energy<me["hEEpZ_Minenergy_ix_iy"]->getBinContent(ix,iy))
	  me["hEEpZ_Minenergy_ix_iy"]->setBinContent(ix,iy, Energy);
      }
  } // loop over EE
  edm::LogInfo("OutputInfo") << "Done Looping over EE" << endl;
  for (int iEtaRing = 0; iEtaRing<101; iEtaRing++)
    {
      me["hEEpZ_Maxenergyvsir"]->Fill(iEtaRing, EEpZMaxEnergy_EtaRing[iEtaRing]);
      me["hEEpZ_Minenergyvsir"]->Fill(iEtaRing, EEpZMinEnergy_EtaRing[iEtaRing]);
      me["hEEmZ_Maxenergyvsir"]->Fill(iEtaRing, EEmZMaxEnergy_EtaRing[iEtaRing]);
      me["hEEmZ_Minenergyvsir"]->Fill(iEtaRing, EEmZMinEnergy_EtaRing[iEtaRing]);

      if (EEpZActiveRing[iEtaRing])
	{
	  me["hEEpZ_METvsir"]->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Pt());
	  me["hEEpZ_METPhivsir"]->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Phi());
	  me["hEEpZ_MExvsir"]->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Px());
	  me["hEEpZ_MEyvsir"]->Fill(iEtaRing, vEEpZMET_EtaRing[iEtaRing].Py());
	  me["hEEpZ_SETvsir"]->Fill(iEtaRing, EEpZSET_EtaRing[iEtaRing]);
	  me["hEEpZ_Occvsir"]->Fill(iEtaRing, EEpZNActiveCells[iEtaRing]);
	}

      if (EEmZActiveRing[iEtaRing])
	{
	  me["hEEmZ_METvsir"]->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Pt());
	  me["hEEmZ_METPhivsir"]->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Phi());
	  me["hEEmZ_MExvsir"]->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Px());
	  me["hEEmZ_MEyvsir"]->Fill(iEtaRing, vEEmZMET_EtaRing[iEtaRing].Py());
	  me["hEEmZ_SETvsir"]->Fill(iEtaRing, EEmZSET_EtaRing[iEtaRing]);
	  me["hEEmZ_Occvsir"]->Fill(iEtaRing, EEmZNActiveCells[iEtaRing]);
	}
    }
  edm::LogInfo("OutputInfo") << "Done ..." << endl;
} // loop over RecHits




#include "DQMCPPF.h"

DQM_CPPF::DQM_CPPF(const edm::ParameterSet& iConfig) :
  cppfDigiToken_(consumes<l1t::CPPFDigiCollection>(iConfig.getParameter<edm::InputTag>("cppfdigiLabel"))),
  EMTF_sector(0),    
  EMTF_subsector(0),
  EMTF_bx(0) {
}

DQM_CPPF::~DQM_CPPF(){
}

void DQM_CPPF::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){  
  
  //Get the CPPFDigi 
  edm::Handle<l1t::CPPFDigiCollection> CppfDigis;
  iEvent.getByToken(cppfDigiToken_, CppfDigis);
  
  //Fill the specific bin for each EMTF sector 
  for(int i = 1; i < 7; i++ ){
    EMTFsector1bins.push_back(i);
    EMTFsector2bins.push_back(i+6);
    EMTFsector3bins.push_back(i+12);
    EMTFsector4bins.push_back(i+18);
    EMTFsector5bins.push_back(i+24);
    EMTFsector6bins.push_back(i+30);
  }
  //FIll the map for each EMTF sector 
  fill_info[1] = EMTFsector1bins;
  fill_info[2] = EMTFsector2bins;
  fill_info[3] = EMTFsector3bins;
  fill_info[4] = EMTFsector4bins;
  fill_info[5] = EMTFsector5bins;
  fill_info[6] = EMTFsector6bins;
  
  
  for(auto& cppf_digis : *CppfDigis){
    
    RPCDetId rpcId = cppf_digis.rpcId();
    int ring = rpcId.ring();
    int station = rpcId.station();
    int region = rpcId.region();
    int subsector = rpcId.subsector();
    
    //Region -	
    if(region == -1){
      
      //for Occupancy
      EMTF_sector = cppf_digis.emtf_sector();
      EMTF_subsector = fill_info[EMTF_sector][subsector-1];
      
      if((station == 4) && (ring == 3))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 1);
      else if((station == 4) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 2);
      else if((station == 3) && (ring == 3))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 3);
      else if((station == 3) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 4);
      else if((station == 2) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 5);
      else if((station == 1) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 6);
      
      //for Track_Bx
      EMTF_bx = cppf_digis.bx();  
      if(EMTF_sector==1)
	Track_Bx->Fill(6,EMTF_bx);
      else if(EMTF_sector==2)
	Track_Bx->Fill(5,EMTF_bx);
      else if(EMTF_sector==3)
	Track_Bx->Fill(4,EMTF_bx);
      else if(EMTF_sector==4)
	Track_Bx->Fill(3,EMTF_bx);
      else if(EMTF_sector==5)
	Track_Bx->Fill(2,EMTF_bx);
      else if(EMTF_sector==6)
	Track_Bx->Fill(1,EMTF_bx);
    }

    //Region +	
    else if(region == 1){
      
      //for Occupancy
      EMTF_sector = cppf_digis.emtf_sector();
      EMTF_subsector = fill_info[EMTF_sector][subsector-1]; 
      
      if((station == 1) && (ring == 2)) 
	Occupancy_EMTFSector->Fill(EMTF_subsector, 7);          
      else if((station == 2) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 8);          
      else if((station == 3) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 9);          
      else if((station == 3) && (ring == 3))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 10);          
      else if((station == 4) && (ring == 2))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 11);          
      else if((station == 4) && (ring == 3))
	Occupancy_EMTFSector->Fill(EMTF_subsector, 12);          

      //for Track_Bx
      EMTF_bx = cppf_digis.bx();
      if(EMTF_sector==1)
	Track_Bx->Fill(7,EMTF_bx);
      else if(EMTF_sector==2)
	Track_Bx->Fill(8,EMTF_bx);
      else if(EMTF_sector==3)
	Track_Bx->Fill(9,EMTF_bx);
      else if(EMTF_sector==4)
	Track_Bx->Fill(10,EMTF_bx);
      else if(EMTF_sector==5)
	Track_Bx->Fill(11,EMTF_bx);
      else if(EMTF_sector==6)
	Track_Bx->Fill(12,EMTF_bx);
    }
    
    
    //General hists    
    
    Phi_Integer->Fill(cppf_digis.phi_int());
    Theta_Integer->Fill(cppf_digis.theta_int());
    Phi_Global->Fill(cppf_digis.phi_glob()*TMath::Pi()/180.);
    Theta_Global->Fill(cppf_digis.theta_glob()*TMath::Pi()/180.);
    Phi_Global_Integer->Fill(cppf_digis.phi_glob(), cppf_digis.phi_int());
    Theta_Global_Integer->Fill(cppf_digis.theta_glob(), cppf_digis.theta_int());
    
  } // loop over CPPFDigis
  
} //End class

void DQM_CPPF::beginRun(const edm::Run& run, const edm::EventSetup& iSetup){
  iSetup.get<MuonGeometryRecord>().get(rpcGeom);
}

void DQM_CPPF::beginJob(){
  edm::Service<TFileService> fs;
  Phi_Integer = fs->make<TH1D>("Phi_Integer", "Phi_Integer", 1240, 0., 1240.);
  Theta_Integer = fs->make<TH1D>("Theta_Integer", "Theta_Integer", 32, 0., 32.);
  Phi_Global = fs->make<TH1D>("Phi_Global", "Phi_Global", 72, -3.15, 3.15);
  Theta_Global = fs->make<TH1D>("Theta_Global", "Theta_Global", 32, 0., 3.15);
  Phi_Global_Integer = fs->make<TH2D>("Phi_Global_Integer", "Phi_Global_Integer", 360, -180, 180, 1240, 0.,1240.);
  Theta_Global_Integer = fs->make<TH2D>("Theta_Global_Integer", "Theta_Global_Integer", 45, 0, 45, 32, 0.,32.);
  Occupancy_EMTFSector = fs->make<TH2D>("Occupancy_EMTFSector", "Occupancy_EMTFSector", 36, 1., 37., 12, 1.,13.); 
  Track_Bx = fs->make<TH2D>("Track_Bx","Track_Bx", 12, 1., 13., 7,-3.,4.);
  return;
}
//define this as a plug-in
DEFINE_FWK_MODULE(DQM_CPPF);

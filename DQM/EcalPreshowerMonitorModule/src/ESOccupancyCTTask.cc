#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyCTTask.h"

#include <iostream>
#include "TMinuit.h"
#include "TRandom.h"

using namespace cms;
using namespace edm;
using namespace std;

ESOccupancyCTTask::ESOccupancyCTTask(const ParameterSet& ps) {

  digilabel_    = ps.getUntrackedParameter<string>("DigiLabel");
  rechitlabel_  = ps.getUntrackedParameter<string>("RecHitLabel");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  gain_         = ps.getUntrackedParameter<int>("ESGain", 1);
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);
  zs_N_sigmas_     = ps.getUntrackedParameter<double>("zs_N_sigmas", 5.0);

  init_ = false;

  for (int i=0; i<2; ++i) {
    for (int j=0; j<6; ++j) {
      meEnergy_[i][j] = 0;
      meOccupancy1D_[i][j] = 0;  
      meOccupancy2D_[i][j] = 0;  
    }
  }
  hitStrips1B_=0;
  hitStrips2B_=0;
  hitSensors1B_=0;
  hitSensors2B_=0;

  meTrack_Npoints_=0;
  me_Nhits_lad0_=0;
  me_Nhits_lad1_=0;
  me_Nhits_lad2_=0;
  me_Nhits_lad3_=0;
  me_Nhits_lad4_=0;
  me_Nhits_lad5_=0;
  me_hit_x_=0;
  me_hit_y_=0;
  meTrack_hit0_=0;
  meTrack_hit1_=0;
  meTrack_hit2_=0;
  meTrack_hit3_=0;
  meTrack_hit4_=0;
  meTrack_hit5_=0;
  meTrack_Px0_=0;
  meTrack_Px1_=0;
  meTrack_Px2_=0;
  meTrack_Px3_=0;
  meTrack_Px4_=0;
  meTrack_Px5_=0;
  meTrack_Py0_=0;
  meTrack_Py1_=0;
  meTrack_Py2_=0;
  meTrack_Py3_=0;
  meTrack_Py4_=0;
  meTrack_Py5_=0;
  meTrack_Pz0_=0;
  meTrack_Pz1_=0;
  meTrack_Pz2_=0;
  meTrack_Pz3_=0;
  meTrack_Pz4_=0;
  meTrack_Pz5_=0;
  meTrack_par0_=0;
  meTrack_par1_=0;
  meTrack_par2_=0;
  meTrack_par3_=0;
  meTrack_par4_=0;
  meTrack_par5_=0;

}

ESOccupancyCTTask::~ESOccupancyCTTask(){
}

void ESOccupancyCTTask::beginJob(const EventSetup& c) {

  ievt_ = 0;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESOccupancyCTTask");
    dbe_->rmdir("ES/ESOccupancyCTTask");
  }

}

void ESOccupancyCTTask::setup(void){

  init_ = true;

  Char_t hist[200];

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {   
    dbe_->setCurrentFolder("ES/ESOccupancyCTTask");
    for (int i=0; i<2; ++i) {       
      for (int j=0; j<6; ++j) {

	sprintf(hist, "ES Energy Box %d P %d", i+1, j+1);      
	meEnergy_[i][j] = dbe_->book1D(hist, hist, 500, 0, 500);

	sprintf(hist, "ES Occupancy 1D Box %d P %d", i+1, j+1);      
	meOccupancy1D_[i][j] = dbe_->book1D(hist, hist, 30, 0, 30);

	sprintf(hist, "ES Occupancy 2D Box %d P %d", i+1, j+1);      
	meOccupancy2D_[i][j] = dbe_->book2D(hist, hist, 64, 0, 64, 5, 0, 5);
      }
    }

    sprintf(hist, "Box1 Plane vs Strip, Current event");
    hitStrips1B_ = dbe_->book2D(hist, hist, 64, 0, 64, 6, 0, 6);
    sprintf(hist, "Box1 Plane vs Sensor, Current event");
    hitSensors1B_= dbe_->book2D(hist, hist, 5, 0, 5, 6, 0, 6);
    sprintf(hist, "Box2 Plane vs Strip, Current event");
    hitStrips2B_ = dbe_->book2D(hist, hist, 64, 0, 64, 6, 0, 6);
    sprintf(hist, "Box2 Plane vs Sensor, Current event");
    hitSensors2B_= dbe_->book2D(hist, hist, 5, 0, 5, 6, 0, 6);

    meTrack_Npoints_=dbe_->bookInt("meTrack_Npoints");
    me_Nhits_lad0_=dbe_->bookInt("me_Nhits_lad0");
    me_Nhits_lad1_=dbe_->bookInt("me_Nhits_lad1");
    me_Nhits_lad2_=dbe_->bookInt("me_Nhits_lad2");
    me_Nhits_lad3_=dbe_->bookInt("me_Nhits_lad3");
    me_Nhits_lad4_=dbe_->bookInt("me_Nhits_lad4");
    me_Nhits_lad5_=dbe_->bookInt("me_Nhits_lad5");
    me_hit_x_=dbe_->book2D("me_hit_x", "me_hit_x", 6, 0, 6, 200, 0, 200);
    me_hit_y_=dbe_->book2D("me_hit_y", "me_hit_y", 6, 0, 6, 200, 0, 200);
    meTrack_hit0_=dbe_->bookInt("meTrack_hit0");
    meTrack_hit1_=dbe_->bookInt("meTrack_hit1");
    meTrack_hit2_=dbe_->bookInt("meTrack_hit2");
    meTrack_hit3_=dbe_->bookInt("meTrack_hit3");
    meTrack_hit4_=dbe_->bookInt("meTrack_hit4");
    meTrack_hit5_=dbe_->bookInt("meTrack_hit5");
    meTrack_Px0_=dbe_->bookFloat("meTrack_Px0");
    meTrack_Px1_=dbe_->bookFloat("meTrack_Px1");
    meTrack_Px2_=dbe_->bookFloat("meTrack_Px2");
    meTrack_Px3_=dbe_->bookFloat("meTrack_Px3");
    meTrack_Px4_=dbe_->bookFloat("meTrack_Px4");
    meTrack_Px5_=dbe_->bookFloat("meTrack_Px5");
    meTrack_Py0_=dbe_->bookFloat("meTrack_Py0");
    meTrack_Py1_=dbe_->bookFloat("meTrack_Py1");
    meTrack_Py2_=dbe_->bookFloat("meTrack_Py2");
    meTrack_Py3_=dbe_->bookFloat("meTrack_Py3");
    meTrack_Py4_=dbe_->bookFloat("meTrack_Py4");
    meTrack_Py5_=dbe_->bookFloat("meTrack_Py5");
    meTrack_Pz0_=dbe_->bookFloat("meTrack_Pz0");
    meTrack_Pz1_=dbe_->bookFloat("meTrack_Pz1");
    meTrack_Pz2_=dbe_->bookFloat("meTrack_Pz2");
    meTrack_Pz3_=dbe_->bookFloat("meTrack_Pz3");
    meTrack_Pz4_=dbe_->bookFloat("meTrack_Pz4");
    meTrack_Pz5_=dbe_->bookFloat("meTrack_Pz5");
    meTrack_par0_=dbe_->bookFloat("meTrack_par0");
    meTrack_par1_=dbe_->bookFloat("meTrack_par1");
    meTrack_par2_=dbe_->bookFloat("meTrack_par2");
    meTrack_par3_=dbe_->bookFloat("meTrack_par3");
    meTrack_par4_=dbe_->bookFloat("meTrack_par4");
    meTrack_par5_=dbe_->bookFloat("meTrack_par5");

  }

}

void ESOccupancyCTTask::cleanup(void){

  if (sta_) return;

  dbe_ = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe_ ) {
    dbe_->setCurrentFolder("ES/ESOccupancyCTTask");

    for (int i=0; i<2; ++i) {
      for (int j=0; j<6; ++j) {
	if ( meEnergy_[i][j] ) dbe_->removeElement( meEnergy_[i][j]->getName() );
	meEnergy_[i][j] = 0;

	if ( meOccupancy1D_[i][j] ) dbe_->removeElement( meOccupancy1D_[i][j]->getName() );
	meOccupancy1D_[i][j] = 0;

	if ( meOccupancy2D_[i][j] ) dbe_->removeElement( meOccupancy2D_[i][j]->getName() );
	meOccupancy2D_[i][j] = 0;
      }
    }

    if ( hitStrips1B_ ) dbe_->removeElement( hitStrips1B_->getName() );
    hitStrips1B_ = 0;

    if ( hitSensors1B_ ) dbe_->removeElement( hitSensors1B_->getName() );
    hitSensors1B_ = 0;

    if ( hitStrips2B_ ) dbe_->removeElement( hitStrips2B_->getName() );
    hitStrips2B_ = 0;

    if ( hitSensors2B_ ) dbe_->removeElement( hitSensors2B_->getName() );
    hitSensors2B_ = 0;

    if ( meTrack_Npoints_ ) dbe_->removeElement("meTrack_Npoints");
    if ( me_Nhits_lad0_ ) dbe_->removeElement("me_Nhits_lad0");
    if ( me_Nhits_lad1_ ) dbe_->removeElement("me_Nhits_lad1");
    if ( me_Nhits_lad2_ ) dbe_->removeElement("me_Nhits_lad2");
    if ( me_Nhits_lad3_ ) dbe_->removeElement("me_Nhits_lad3");
    if ( me_Nhits_lad4_ ) dbe_->removeElement("me_Nhits_lad4");
    if ( me_Nhits_lad5_ ) dbe_->removeElement("me_Nhits_lad5");
    if ( me_hit_x_ ) dbe_->removeElement("me_hit_x");
    if ( me_hit_y_ ) dbe_->removeElement("me_hit_y");
    if ( meTrack_hit0_ ) dbe_->removeElement("meTrack_hit0");
    if ( meTrack_hit1_ ) dbe_->removeElement("meTrack_hit1");
    if ( meTrack_hit2_ ) dbe_->removeElement("meTrack_hit2");
    if ( meTrack_hit3_ ) dbe_->removeElement("meTrack_hit3");
    if ( meTrack_hit4_ ) dbe_->removeElement("meTrack_hit4");
    if ( meTrack_hit5_ ) dbe_->removeElement("meTrack_hit5");
    if ( meTrack_Px0_ ) dbe_->removeElement("meTrack_Px0");
    if ( meTrack_Px1_ ) dbe_->removeElement("meTrack_Px1");
    if ( meTrack_Px2_ ) dbe_->removeElement("meTrack_Px2");
    if ( meTrack_Px3_ ) dbe_->removeElement("meTrack_Px3");
    if ( meTrack_Px4_ ) dbe_->removeElement("meTrack_Px4");
    if ( meTrack_Px5_ ) dbe_->removeElement("meTrack_Px5");
    if ( meTrack_Py0_ ) dbe_->removeElement("meTrack_Py0");
    if ( meTrack_Py1_ ) dbe_->removeElement("meTrack_Py1");
    if ( meTrack_Py2_ ) dbe_->removeElement("meTrack_Py2");
    if ( meTrack_Py3_ ) dbe_->removeElement("meTrack_Py3");
    if ( meTrack_Py4_ ) dbe_->removeElement("meTrack_Py4");
    if ( meTrack_Py5_ ) dbe_->removeElement("meTrack_Py5");
    if ( meTrack_Pz0_ ) dbe_->removeElement("meTrack_Pz0");
    if ( meTrack_Pz1_ ) dbe_->removeElement("meTrack_Pz1");
    if ( meTrack_Pz2_ ) dbe_->removeElement("meTrack_Pz2");
    if ( meTrack_Pz3_ ) dbe_->removeElement("meTrack_Pz3");
    if ( meTrack_Pz4_ ) dbe_->removeElement("meTrack_Pz4");
    if ( meTrack_Pz5_ ) dbe_->removeElement("meTrack_Pz5");
    if ( meTrack_par0_ ) dbe_->removeElement("meTrack_par0");
    if ( meTrack_par1_ ) dbe_->removeElement("meTrack_par1");
    if ( meTrack_par2_ ) dbe_->removeElement("meTrack_par2");
    if ( meTrack_par3_ ) dbe_->removeElement("meTrack_par3");
    if ( meTrack_par4_ ) dbe_->removeElement("meTrack_par4");
    if ( meTrack_par5_ ) dbe_->removeElement("meTrack_par5");
    meTrack_Npoints_=0;
    me_Nhits_lad0_=0;
    me_Nhits_lad1_=0;
    me_Nhits_lad2_=0;
    me_Nhits_lad3_=0;
    me_Nhits_lad4_=0;
    me_Nhits_lad5_=0;
    me_hit_x_=0;
    me_hit_y_=0;
    meTrack_hit0_=0;
    meTrack_hit1_=0;
    meTrack_hit2_=0;
    meTrack_hit3_=0;
    meTrack_hit4_=0;
    meTrack_hit5_=0;
    meTrack_Px0_=0;
    meTrack_Px1_=0;
    meTrack_Px2_=0;
    meTrack_Px3_=0;
    meTrack_Px4_=0;
    meTrack_Px5_=0;
    meTrack_Py0_=0;
    meTrack_Py1_=0;
    meTrack_Py2_=0;
    meTrack_Py3_=0;
    meTrack_Py4_=0;
    meTrack_Py5_=0;
    meTrack_Pz0_=0;
    meTrack_Pz1_=0;
    meTrack_Pz2_=0;
    meTrack_Pz3_=0;
    meTrack_Pz4_=0;
    meTrack_Pz5_=0;
    meTrack_par0_=0;
    meTrack_par1_=0;
    meTrack_par2_=0;
    meTrack_par3_=0;
    meTrack_par4_=0;
    meTrack_par5_=0;

  }

  init_ = false;
}

void ESOccupancyCTTask::endJob(void) {

  LogInfo("ESOccupancyCTTask") << "analyzed " << ievt_ << " events";

  if ( init_ ) this->cleanup();
}

void ESOccupancyCTTask::analyze(const Event& e, const EventSetup& c){

  if ( ! init_ ) this->setup();
  ievt_++;

  Handle<ESRawDataCollection> dccs;
  try {
    e.getByLabel(digilabel_, dccs);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESOccupancy : Error! can't get ES raw data collection !" << std::endl;
  }

  //Handle<ESDigiCollection> digis;
  //try {
  //e.getByLabel(label_, instanceName_, digis);
  //} catch ( cms::Exception &e ) {
  //LogDebug("") << "ESOccupancy : Error! can't get collection !" << std::endl;
  //}

  Handle<ESRecHitCollection> hits;
  try {
    e.getByLabel(rechitlabel_, instanceName_, hits);
  } catch ( cms::Exception &e ) {
    LogDebug("") << "ESOccupancy : Error! can't get ES rec hit collection !" << std::endl;
  }


  // Strips with signal StripSignal=1 else 0
  double StripSignal[2][6][2][5][32];
  for (int i=0; i<2; ++i) 
    for (int j=0; j<6; ++j)
      for (int k=0; k<2; ++k)        
	for (int m=0; m<5; ++m)
	  for (int n=0; n<32; ++n)
	    StripSignal[i][j][k][m][n] = 0.;  


  // DCC
  vector<int> tdc;
  for ( ESRawDataCollection::const_iterator dccItr = dccs->begin(); dccItr != dccs->end(); ++dccItr ) {

    ESDCCHeaderBlock dcc = (*dccItr);

    if (dcc.fedId()==4) tdc = dcc.getTDCChannel();   
  }

  double fts = ((double)tdc[7]-800.)*24.951/194.;

  double w[3];
  if (gain_==1) {
    w[0] = -0.35944  + 0.03199 * fts;
    w[1] =  0.58562  + 0.03724 * fts;
    w[2] =  0.77204  - 0.06913 * fts;
  } else {
    w[0] = -0.26888  + 0.01807 * fts;
    w[1] =  0.54452  + 0.03204 * fts;
    w[2] =  0.72597  - 0.05062 * fts;
  }

  double zsThreshold = zs_N_sigmas_*sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])*7.*81.08/50./1000000.;
  //cout<<"ZS : "<<zsThreshold<<" "<<tdc[7]<<" "<<fts<<" "<<sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])<<" "<<w[0]<<" "<<w[1]<<" "<<w[2]<<endl;

  //for (ESDigiCollection::const_iterator digiItr = digis->begin(); digiItr != digis->end(); ++digiItr ) {
  //ESDataFrame dataframe = (*digiItr);
  //ESDetId id = dataframe.id();
  //int plane = id.plane();
  //int ix    = id.six();
  //int iy    = id.siy();
  //}


  int occ[2][6];
  for (int m=0; m<2; ++m) 
    for (int n=0; n<6; ++n) 
      occ[m][n] = 0;  

  for ( ESRecHitCollection::const_iterator recHit = hits->begin(); recHit != hits->end(); ++recHit )  {

    if (recHit->energy()<zsThreshold) continue;


    ESDetId ESid = ESDetId(recHit->id());

    //int zside = ESid.zside();
    //int plane = ESid.plane();
    int ix    = ESid.six();
    int iy    = ESid.siy();
    int strip = ESid.strip();

    int j = (ix-1)/2; 
    int i;
    if (j<=5) i = 0;
    else i = 1;
    if (j>5) j = j-6;    
    int k = (ix-1)%2;

    meEnergy_[i][j]->Fill(recHit->energy()*1000000.);
    meOccupancy2D_[i][j]->Fill(abs(strip-32-k*32), iy-1, 1);

    occ[i][j]++;

    //Store StripSignal for strips with real signal 
    StripSignal[i][j][k][iy-1][strip-1]=recHit->energy();
  }

  for (int m=0; m<2; ++m) 
    for (int n=0; n<6; ++n) 
      meOccupancy1D_[m][n]->Fill(occ[m][n]);



  /////////   
  // Event at old DQM format for tracking
  float local_event[2][6][10][32]; //Z, Plane, detector_id, strip_id
  int id_det=0;

  for (int i=0; i<2; ++i){ 
    for (int j=0; j<6; ++j){
      for (int k=0; k<2; ++k){        
	for (int m=0; m<5; ++m){
	  if(k==0 && m==0) id_det=8;
	  if(k==0 && m==1) id_det=6;
	  if(k==0 && m==2) id_det=4;
	  if(k==0 && m==3) id_det=2;
	  if(k==0 && m==4) id_det=0;
	  if(k==1 && m==0) id_det=9;
	  if(k==1 && m==1) id_det=7;
	  if(k==1 && m==2) id_det=5;
	  if(k==1 && m==3) id_det=3;
	  if(k==1 && m==4) id_det=1;
	  for (int n=0; n<32; ++n){
	    local_event[i][j][id_det][n]=StripSignal[i][j][k][m][n];
	  }
	}
      }
    }
  }



  // Test Pattern: uncomment only for tests!!
  //      for (int i=0; i<2; ++i) 
  //        for (int j=0; j<6; ++j)
  //          for (int k=0; k<10; ++k)        
  //  	  for (int m=0; m<32; ++m)
  //              local_event[i][j][k][m]=0;

  //        local_event[0][0][0][0]=1;
  //        local_event[0][1][6][5]=1;
  //        local_event[0][2][2][0]=1;
  //        local_event[0][3][4][10]=1;
  //        local_event[0][4][2][0]=1;
  //        local_event[0][5][8][15]=1;
  //        local_event[1][0][1][25]=1;
  //        local_event[1][1][6][5]=1;
  //        local_event[1][2][3][10]=1;
  //        local_event[1][3][5][12]=1;
  //        local_event[1][4][9][15]=1;
  //        local_event[1][5][8][19]=1;


  int hit1_strips[6][64], hit1_sensors[6][5]; //Z=1 (1st box)
  int hit2_strips[6][64], hit2_sensors[6][5]; //Z=1 (2nd box)

  for(int i=0; i<6; i++){
    for(int j=0; j<64; ++j){
      hit1_strips[i][j]=0;
      hit2_strips[i][j]=0;
    }
    for(int j=0; j<5; ++j){
      hit1_sensors[i][j]=0;
      hit2_sensors[i][j]=0;
    }   
  }

  // Evaluate hits on strip level - Sensor level 1st box
  for(int j=0; j<6; j++){
    for(int k=0; k<10; k=k+2){
      for(int m=0; m<32; ++m){
	if(local_event[0][j][k][m]>0 ){
	  hit1_strips[j][m]=1;
	  if (k==0) {hit1_sensors[j][0]=1;}
	  if (k==2) {hit1_sensors[j][1]=1;}
	  if (k==4) {hit1_sensors[j][2]=1;}
	  if (k==6) {hit1_sensors[j][3]=1;}
	  if (k==8) {hit1_sensors[j][4]=1;}
	}
      }
    }
  }
  for(int j=0; j<6; j++){
    for(int k=1; k<10; k=k+2){
      for(int m=0; m<32; ++m){ 
	if(local_event[0][j][k][m]>0 ){  
	  hit1_strips[j][m+32]=1; 
	  if (k==1) {hit1_sensors[j][0]=1;}
	  if (k==3) {hit1_sensors[j][1]=1;}
	  if (k==5) {hit1_sensors[j][2]=1;}
	  if (k==7) {hit1_sensors[j][3]=1;}
	  if (k==9) {hit1_sensors[j][4]=1;}
	}
      }
    }
  }

  // Evaluate hits on strip level - Sensor level 2nd box
  for(int j=0; j<6; j++){
    for(int k=0; k<10; k=k+2){
      for(int m=0; m<32; ++m){
	if(local_event[1][j][k][m]>0 ){
	  hit2_strips[j][m]=1;
	  if (k==0) {hit2_sensors[j][0]=1;}
	  if (k==2) {hit2_sensors[j][1]=1;}
	  if (k==4) {hit2_sensors[j][2]=1;}
	  if (k==6) {hit2_sensors[j][3]=1;}
	  if (k==8) {hit2_sensors[j][4]=1;}
	}
      }
    }
  }
  for(int j=0; j<6; j++){
    for(int k=1; k<10; k=k+2){
      for(int m=0; m<32; ++m){ 
	if(local_event[1][j][k][m]>0 ){  
	  hit2_strips[j][m+32]=1; 
	  if (k==1) {hit2_sensors[j][0]=1;}
	  if (k==3) {hit2_sensors[j][1]=1;}
	  if (k==5) {hit2_sensors[j][2]=1;}
	  if (k==7) {hit2_sensors[j][3]=1;}
	  if (k==9) {hit2_sensors[j][4]=1;}
	}
      }
    }
  }


  //Reset MonitoringElement first - All bins to zero    
  for(int i=0; i<=6; ++i){
    for(int j=1; j<=64; ++j){
      hitStrips1B_->setBinContent(j,i,0);
      hitStrips2B_->setBinContent(j,i,0);
    }
    for(int j=0; j<=5; ++j){
      hitSensors1B_->setBinContent(j,i,0);
      hitSensors2B_->setBinContent(j,i,0);
    }
  }


  // Fill histos for single event
  for(int i=0;i<6;i++){
    for(int j=0;j<64;j++){
      if(hit1_strips[i][j]==1){hitStrips1B_->Fill(j,i);}
      if(hit2_strips[i][j]==1){hitStrips2B_->Fill(j,i);}
    }
    for(int j=0;j<5;j++){
      if(hit1_sensors[i][j]==1){hitSensors1B_->Fill(j,i);}
      if(hit2_sensors[i][j]==1){hitSensors2B_->Fill(j,i);}
    }
  }


  // Here should be put the Tracking algorithm
  // with local_event[2][6][10][32]


  DoTracking(local_event, 0);   // do tracking for z=0


}



//==========================================================================
//==========================================================================
//==========================================================================

int Npoints=0;
double Px[6], Py[6], Pz[6];
double best_Px[6], best_Py[6], best_Pz[6], best_fcn, best_par[6], best_epar[6];
int best_hit[6];

//-------------------------------------------------------------------------------------------------

void fcn(Int_t &npar, Double_t *gin, Double_t &f, Double_t *par, Int_t iflag)
{
  // par[0] = P1x   P1: a point on the track
  // par[1] = P1y
  // par[2] = P1z
  // par[3] = ax     a: a track direction vector
  // par[4] = ay
  // par[5] = az

  //       | (r0-r1) x a |
  //   d = ---------------
  //             |a|

  double deg=acos(-1.0)/180.0;
  double sx=0.19/2.0;             // cm
  double sy=6.3*cos(3.8*deg)/2.0; // cm
  double sz=6.3*sin(3.8*deg)/2.0; // cm

  // a0 = normalized direction vector
  Double_t a0[3];
  a0[0]=par[3];
  a0[1]=par[4];
  a0[2]=par[5];
  TMath::Normalize(a0);

  //   d = | (r0-r1) x a0 |

  Double_t Dr[3], dum[3];
  Double_t d=0, sum2=0, sd2=0;
  for (int i=0; i<Npoints; i++) {
    Dr[0]=Px[i]-par[0];
    Dr[1]=Py[i]-par[1];
    Dr[2]=Pz[i]-par[2];
    d=TMath::NormCross(Dr,a0,dum);
    if (d==0) d=1e-10;
    sd2 = ( // sigma d squared
	pow(( a0[1]*a0[1]*Dr[0] - a0[0]*a0[1]*Dr[1] + a0[2]*( a0[2]*Dr[0] - a0[0]*Dr[2]))*sx,2) + 
	pow((-a0[0]*a0[1]*Dr[0] + a0[0]*a0[0]*Dr[1] + a0[2]*( a0[2]*Dr[1] - a0[1]*Dr[2]))*sy,2) + 
	pow((-a0[0]*a0[2]*Dr[0] + a0[0]*a0[0]*Dr[2] + a0[1]*(-a0[2]*Dr[1] + a0[1]*Dr[2]))*sz,2)
	) / (d*d) ;
    sum2+=d*d/sd2;
  }
  f = sum2;
}

//-------------------------------------------------------------------------------------------------

Double_t Ycenter(Double_t x) {
  // translate to positive values...
  Double_t Ylen=6.3*cos(3.8*acos(-1.0)/180);
  Double_t xx=x+Ylen*5.0/2.0;
  int i=TMath::FloorNint(xx/Ylen)-2;
  return i*Ylen;
}

//-------------------------------------------------------------------------------------------------

void MakeTheFit(TMinuit *mini, int Npoints, Double_t *Px, Double_t *Py, Double_t *Pz,
    int *NonEmpty, int *ind) {

  // printf("========> Npoints = %d\n", Npoints);
  // for (int i=0; i<Npoints; i++) {
  //   printf("lad-%d hit-%d   x=%6.2f y=%6.2f z=%6.2f\n", NonEmpty[i], ind[i], Px[i], Py[i], Pz[i]);
  // }

  Int_t ierflg = 0;
  Double_t arglist[10];

  // do fit...
  mini->mnparm(0, "P1x",            Px[0], 1,    -8,    8,   ierflg);  // P1: a point on the track
  mini->mnparm(1, "P1y",            Py[0], 1,   -40,   40,   ierflg);
  mini->mnparm(2, "P1z",            Pz[0], 1,   -40,   40,   ierflg);
  mini->mnparm(3, "ax", Px[Npoints]-Px[0], 1,  -100,  100,   ierflg);  // a: a track direction vector
  mini->mnparm(4, "ay", Py[Npoints]-Py[0], 1,  -100,  100,   ierflg);  // initialized using the top and bottom hits...
  mini->mnparm(5, "az", Pz[Npoints]-Pz[0], 1,  -100,  100,   ierflg);
  arglist[0] = 5000;  // max number of calls to fcn
  arglist[1] = 1e-20; // tolerance
  mini->mnexcm("MIGRAD", arglist, 2, ierflg);

  // printf("-----------------------------------------\n");
  Double_t par[6], epar[6], fval;
  for (int i=0; i<6; i++) {
    mini->GetParameter(i,par[i],epar[i]);
    // printf("par %d = %8.3f +- %8.3f\n",i,par[i],epar[i]);
  }
  mini->Eval(6, 0, fval, par, 1);
  // printf("ierflg=%d (%s)   FCN=%f\n", ierflg, ierflg?"FAILED":"OK", fval);
  // printf("-----------------------------------------\n");

  if (fval<best_fcn) {
    best_fcn=fval;
    for (int i=0; i<Npoints; i++) {
      best_Px[i]=Px[i];
      best_Py[i]=Py[i];
      best_Pz[i]=Pz[i];
      best_hit[NonEmpty[i]]=ind[i];
    }
    for (int i=0; i<6; i++) {
      mini->GetParameter(i,best_par[i],best_epar[i]);
    }
  }

}

//-------------------------------------------------------------------------------------------------

void ESOccupancyCTTask::DoTracking(float local_event[2][6][10][32], int zbox) {

  // scan all sensors to find all hits as cluster centroids
  int    Nhits_lad[6];       // Nhits_lad[i]  : gives the number of hits in ladder i
  int      hit_sen[6][200];  // hit_sen[i][j] : gives the sensor that fired by the j-th hit in ladder i
  Double_t hit_str[6][200];  // hit_str[i][j] : gives the strip  that fired by the j-th hit in ladder i

  for(int i=0; i<6; ++i){
    Nhits_lad[i]=0;
    for(int j=0; j<10; ++j){
      for(int k=0; k<32; ++k) {

	if (local_event[zbox][i][j][k]>0) {
	  Double_t q=0, qx=0;
	  int kk;
	  for (kk=k; kk<32; kk++) {
	    if (local_event[zbox][i][j][kk]==0) break;
	    q+=local_event[zbox][i][j][kk];
	    qx+=local_event[zbox][i][j][kk]*kk;
	  }
	  hit_sen[i][Nhits_lad[i]]=j;
	  hit_str[i][Nhits_lad[i]]=qx/q;
	  Nhits_lad[i]++;
	  k=kk;
	}

      }
    }
  }

  // print hits in Ladder-Sensor-Strip representation
  /*
     for (int i=0; i<6; i++) {
     printf("Ladder %d has %d hits %s   ", i, Nhits_lad[i], Nhits_lad[i]?":":"");
     for (int j=0; j<Nhits_lad[i]; j++) {
     printf("(sen=%d str=%.2f)   ",hit_sen[i][j],hit_str[i][j]);
     }
     printf("\n");
     }
     printf("\n");
   */

  Double_t deg=acos(-1.0)/180.0;
  Double_t phi=3.8;              //      sensor tilt in degrees
  Double_t DX=6.3;               //      sensor X projection length (cm)
  Double_t DY=6.3*cos(phi*deg);  //      sensor Y projection length (cm)
  Double_t DZ=39.0;              // all sensors Z projection length (cm)

  // Translate Ladder-Sensor-Strip hits to X-Y-Z hits.
  Double_t hit_x[6][200];
  Double_t hit_y[6][200];
  Double_t hit_z[6];
  for (int i=0; i<6; i++) {
    hit_z[i]=i*DZ/5-DZ/2;
    for (int j=0; j<Nhits_lad[i]; j++) {
      hit_x[i][j]=( (hit_sen[i][j]%2)*32 + hit_str[i][j] + 0.5)*DX/32 - DX;
      hit_y[i][j]=DY*(hit_sen[i][j]/2)-DY*2;
    }
  }

  //--------------------------------------------------------------------------
  //--------- create simulated hits for debugging... -------------------------
  //--------------------------------------------------------------------------
  if ( 0 ) {
    Double_t xslope=2*(2*gRandom->Rndm()-1);
    Double_t yslope=4*(2*gRandom->Rndm()-1);
    for (int i=0; i<6; i++) {
      hit_z[i]=i*DZ/5-DZ/2;
      //Nhits_lad[i]=1+gRandom->Integer(6);
      Nhits_lad[i]=gRandom->Integer(3);
      for (int j=0; j<Nhits_lad[i]; j++) {
	if (j==0) {
	  hit_x[i][j]=xslope*(i-2)+gRandom->Gaus(0,0.1);
	  hit_y[i][j]=Ycenter(yslope*(i-2)+gRandom->Gaus(0,0.1));
	}
	else {
	  hit_x[i][j]=DX*(2*gRandom->Rndm()-1);
	  hit_y[i][j]=Ycenter(DY*(2*gRandom->Rndm()-1));
	}
      }
    }
  }
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------

  // print hits in X-Y-Z representation
  /*
     for (int i=0; i<6; i++) {
     printf("Ladder %d (z=%6.2f) has %d hits%s ", i, hit_z[i], Nhits_lad[i], Nhits_lad[i]?":":"");
     for (int j=0; j<Nhits_lad[i]; j++) {
     printf("(x=%6.2f y=%6.2f) ",hit_x[i][j],hit_y[i][j]);
     }
     printf("\n");
     }
   */

  // find Npoints : the number of points per fit (max=6), i.e. the number of ladders that fired
  Npoints=0;
  int Ncomb=1;  // number of possible tracks
  int NonEmpty[6];
  for (int i=0; i<6; i++) {
    if (Nhits_lad[i]) {
      // printf("---> %d lad=%d %2d\n",Npoints,i,Nhits_lad[i]);
      Ncomb*=Nhits_lad[i];
      NonEmpty[Npoints]=i;
      Npoints++;
    }
  }
  //printf("===> Npoints=%d Ncomb=%d\n\n",Npoints,Ncomb);

  /*
     if (Npoints<3) {
     printf("===> Npoints=%d. Too few or no hits in event!...\n",Npoints);
     return;
     }
   */


  if (Ncomb<100) { // Avoid making fits if there are too many hits, as it would run forever...

    // MINUIT initialization with a maximum of 6 parameters
    TMinuit *mini = new TMinuit(6);
    Int_t ierflg = 0;
    Double_t arglist[10];
    mini->SetFCN(fcn);       // bind fcn to MINUIT
    mini->SetPrintLevel(-1); // -1=quiet 0=normal 1=verbose
    arglist[0] = 1;          // Normally, for chisquared fits = 1, and for negative log likelihood = 0.5
    mini->mnexcm("SET ERR", arglist, 1, ierflg);

    // Make all possible track fits and select the best one (best FCN)
#define FILL_POINT_ARRAYS_AND_MAKE_THE_FIT \
    { for (int i=0; i<Npoints; i++) { Px[i]=hit_x[NonEmpty[i]][ind[i]]; Py[i]=hit_y[NonEmpty[i]][ind[i]]; Pz[i]=hit_z[NonEmpty[i]]; } \
      MakeTheFit(mini,Npoints,Px,Py,Pz,NonEmpty,ind); }
    best_fcn=1e38; // set FCN to something big initially...
    int ind[6];
    for (ind[0]=0; ind[0]<Nhits_lad[NonEmpty[0]]; ind[0]++) {
      if (Npoints==1) FILL_POINT_ARRAYS_AND_MAKE_THE_FIT
      else
	for (ind[1]=0; ind[1]<Nhits_lad[NonEmpty[1]]; ind[1]++) {
	  if (Npoints==2) FILL_POINT_ARRAYS_AND_MAKE_THE_FIT
	  else
	    for (ind[2]=0; ind[2]<Nhits_lad[NonEmpty[2]]; ind[2]++) {
	      if (Npoints==3) FILL_POINT_ARRAYS_AND_MAKE_THE_FIT
	      else
		for (ind[3]=0; ind[3]<Nhits_lad[NonEmpty[3]]; ind[3]++) {
		  if (Npoints==4) FILL_POINT_ARRAYS_AND_MAKE_THE_FIT
		  else
		    for (ind[4]=0; ind[4]<Nhits_lad[NonEmpty[4]]; ind[4]++) {
		      if (Npoints==5) FILL_POINT_ARRAYS_AND_MAKE_THE_FIT
		      else
			for (ind[5]=0; ind[5]<Nhits_lad[NonEmpty[5]]; ind[5]++) {
			  if (Npoints==6) FILL_POINT_ARRAYS_AND_MAKE_THE_FIT
			}
		    }
		}
	    }
	}
    }

    // free memory...
    mini->Delete();

    // Here we have the best fit...
    meTrack_hit0_->Fill(best_hit[0]);
    meTrack_hit1_->Fill(best_hit[1]);
    meTrack_hit2_->Fill(best_hit[2]);
    meTrack_hit3_->Fill(best_hit[3]);
    meTrack_hit4_->Fill(best_hit[4]);
    meTrack_hit5_->Fill(best_hit[5]);
    meTrack_Px0_->Fill(best_Px[0]);
    meTrack_Px1_->Fill(best_Px[1]);
    meTrack_Px2_->Fill(best_Px[2]);
    meTrack_Px3_->Fill(best_Px[3]);
    meTrack_Px4_->Fill(best_Px[4]);
    meTrack_Px5_->Fill(best_Px[5]);
    meTrack_Py0_->Fill(best_Py[0]);
    meTrack_Py1_->Fill(best_Py[1]);
    meTrack_Py2_->Fill(best_Py[2]);
    meTrack_Py3_->Fill(best_Py[3]);
    meTrack_Py4_->Fill(best_Py[4]);
    meTrack_Py5_->Fill(best_Py[5]);
    meTrack_Pz0_->Fill(best_Pz[0]);
    meTrack_Pz1_->Fill(best_Pz[1]);
    meTrack_Pz2_->Fill(best_Pz[2]);
    meTrack_Pz3_->Fill(best_Pz[3]);
    meTrack_Pz4_->Fill(best_Pz[4]);
    meTrack_Pz5_->Fill(best_Pz[5]);
    meTrack_par0_->Fill(best_par[0]);
    meTrack_par1_->Fill(best_par[1]);
    meTrack_par2_->Fill(best_par[2]);
    meTrack_par3_->Fill(best_par[3]);
    meTrack_par4_->Fill(best_par[4]);
    meTrack_par5_->Fill(best_par[5]);
  }
  else {
    // -1 means no track to plot
    meTrack_hit0_->Fill(-1);
    meTrack_hit1_->Fill(-1);
    meTrack_hit2_->Fill(-1);
    meTrack_hit3_->Fill(-1);
    meTrack_hit4_->Fill(-1);
    meTrack_hit5_->Fill(-1);
  }
  meTrack_Npoints_->Fill(Npoints);
  me_Nhits_lad0_->Fill(Nhits_lad[0]);
  me_Nhits_lad1_->Fill(Nhits_lad[1]);
  me_Nhits_lad2_->Fill(Nhits_lad[2]);
  me_Nhits_lad3_->Fill(Nhits_lad[3]);
  me_Nhits_lad4_->Fill(Nhits_lad[4]);
  me_Nhits_lad5_->Fill(Nhits_lad[5]);

  MonitorElementT<TNamed>* meT;
  TH2F *me_hit_x;
  meT = dynamic_cast<MonitorElementT<TNamed>*>(me_hit_x_);
  me_hit_x = dynamic_cast<TH2F*> (meT->operator->());
  TH2F *me_hit_y;
  meT = dynamic_cast<MonitorElementT<TNamed>*>(me_hit_y_);
  me_hit_y = dynamic_cast<TH2F*> (meT->operator->());

  me_hit_x->Reset();
  me_hit_y->Reset();
  for (int i=0; i<6; i++) {
    for (int j=0; j<Nhits_lad[i]; j++) {
      me_hit_x->SetBinContent(i,j,hit_x[i][j]);
      me_hit_y->SetBinContent(i,j,hit_y[i][j]);
    }
  }

}

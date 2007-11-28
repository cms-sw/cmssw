#include "DQM/EcalPreshowerMonitorModule/interface/ESOccupancyCTTask.h"

#include <iostream>

using namespace cms;
using namespace edm;
using namespace std;

ESOccupancyCTTask::ESOccupancyCTTask(const ParameterSet& ps) {

  digilabel_    = ps.getUntrackedParameter<string>("DigiLabel");
  rechitlabel_  = ps.getUntrackedParameter<string>("RecHitLabel");
  instanceName_ = ps.getUntrackedParameter<string>("InstanceES");
  gain_         = ps.getUntrackedParameter<int>("ESGain", 1);
  sta_          = ps.getUntrackedParameter<bool>("RunStandalone", false);

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

  double zsThreshold = 5.*sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])*7.*81.08/50./1000000.;
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
//    for (int i=0; i<2; ++i) 
//      for (int j=0; j<6; ++j)
//        for (int k=0; k<10; ++k)        
//	  for (int m=0; m<32; ++m)
//            local_event[i][j][k][m]=0;

//      local_event[0][0][0][0]=1;
//      local_event[0][1][6][5]=1;
//      local_event[0][2][2][0]=1;
//      local_event[0][3][4][10]=1;
//      local_event[0][4][2][0]=1;
//      local_event[0][5][8][15]=1;
//      local_event[1][0][1][25]=1;
//      local_event[1][1][6][5]=1;
//      local_event[1][2][3][10]=1;
//      local_event[1][3][5][12]=1;
//      local_event[1][4][9][15]=1;
//      local_event[1][5][8][19]=1;


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


 
}

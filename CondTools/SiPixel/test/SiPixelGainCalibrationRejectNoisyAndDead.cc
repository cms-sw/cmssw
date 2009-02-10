// -*- C++ -*-
//
// Package:    InsertNoisyPixelsInDB
// Class:      InsertNoisyPixelsInDB
// 
/**\class InsertNoisyPixelsInDB InsertNoisyPixelsInDB.cc CondTools/InsertNoisyPixelsInDB/src/InsertNoisyPixelsInDB.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Romain Rougny
//         Created:  Tue Feb  3 15:18:02 CET 2009
// $Id: SiPixelGainCalibrationRejectNoisyAndDead.cc,v 1.1 2009/02/03 16:00:57 rougny Exp $
//
//

#include <fstream>

#include "SiPixelGainCalibrationRejectNoisyAndDead.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

using namespace edm;
using namespace std;

void SiPixelGainCalibrationRejectNoisyAndDead::fillDatabase(const edm::EventSetup& iSetup){
  
  //Get the Calibration Data
  SiPixelGainCalibrationService_.setESObjects(iSetup);
  edm::LogInfo("SiPixelCondObjOfflineReader") << "[SiPixelCondObjOfflineReader::beginJob] End Reading CondObjOfflineects" << std::endl;

  //Get list of ideal detids
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );     
  edm::LogInfo("SiPixelCondObjOfflineBuilder") <<" There are "<<pDD->dets().size() <<" detectors"<<std::endl;
  
  pedlow_=-100;pedhi_=300;gainlow_=0.9;gainhi_=20;
  if(gainlow_<0) gainlow_=0;
  if(gainhi_>20) gainhi_=20;
  if(pedlow_<-100) pedlow_=-100;
  if(pedhi_>300) pedhi_=300;
  theGainCalibrationDbInputOffline_ = new SiPixelGainCalibrationOffline(pedlow_*0.999,pedhi_*1.001,gainlow_*0.999,gainhi_*1.001);

  int nnoisy = 0;
  int ndead = 0;

  uint32_t detid=0;
  int NDetid = 0;
  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
    detid=0;
    if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0)
      detid=((*it)->geographicalId()).rawId();
    if(detid==0)
      continue;
      
    NDetid++;
    //cout<<NDetid<<"  "<<detid<<endl;
    //if(NDetid==164) continue;
 
    // Get the module sizes
    const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>((*it));
    const PixelTopology & topol = pixDet->specificTopology();
    size_t nrows = topol.nrows();      // rows in x
    size_t ncols = topol.ncolumns();   // cols in y
    
    float ped;
    float gainforthiscol[2];
    int nusedrows[2];
    size_t nrowsrocsplit = theGainCalibrationDbInputOffline_->getNumberOfRowsToAverageOver();
    std::vector<char> theSiPixelGainCalibrationGainPerColPedPerPixel;

    
    for(size_t icol=0; icol<=ncols-1; icol++){
      nusedrows[0]=nusedrows[1]=0;
      gainforthiscol[0]=gainforthiscol[1]=0;
      
      for(size_t jrow=0; jrow<=nrows-1; jrow++){
        //std::cout<<"col : "<<icol<<", row : "<<jrow<<std::endl;
	ped = 0;
	
	size_t iglobalrow=0;
	if(jrow>nrowsrocsplit)
	  iglobalrow=1;
	
	bool isPixelDeadOld = false;
	bool isColumnDeadOld = false;
	bool isPixelDeadNew = false;
	bool isPixelNoisyOld = false;
	bool isColumnNoisyOld = false;
	bool isPixelNoisyNew = false;
	
	bool isColumnDead = false;
	bool isColumnNoisy = false;
	bool isPixelDead = false;
	bool isPixelNoisy = false;
	
	
        try {
	  isPixelDeadOld = SiPixelGainCalibrationService_.isDead(detid, icol, jrow);
	  isPixelNoisyOld = SiPixelGainCalibrationService_.isNoisy( detid, icol, jrow);
	  isColumnDeadOld = SiPixelGainCalibrationService_.isNoisyColumn(detid, icol, jrow);
	  isColumnNoisyOld = SiPixelGainCalibrationService_.isDeadColumn( detid, icol, jrow);
	  if(!isColumnDeadOld && !isColumnNoisyOld)
	    gainforthiscol[iglobalrow] = SiPixelGainCalibrationService_.getGain(detid, icol, jrow);
	  if(!isPixelDeadOld && !isPixelNoisyOld)
	    ped = SiPixelGainCalibrationService_.getPedestal(detid, icol, jrow);
	}
	catch(const std::exception& er){}
	//std::cout<<"For DetId "<<detid<<" we found gain : "<<gain<<", pedestal : "<<ped<<std::endl;
        
	
	//Check if pixel is in new noisy list
	for(std::map <int,std::vector<std::pair<int,int> > >::const_iterator it=noisypixelkeeper.begin();it!=noisypixelkeeper.end();it++) 
          for(int i=0;i<(it->second).size();i++)
	    if(it->first==detid && (it->second.at(i)).first==icol && (it->second.at(i)).second==jrow)
	      isPixelNoisyNew = true;
	
	isColumnDead = isColumnDeadOld;
	isPixelDead = isPixelDeadOld;
	
	if(insertnoisypixelsindb_==0){
	  isColumnNoisy = isColumnNoisyOld;
	  isPixelNoisy = isPixelNoisyOld;
	}
	else if(insertnoisypixelsindb_==1)
	  isPixelNoisy = isPixelNoisyNew;
	else if(insertnoisypixelsindb_==2)
	  isPixelNoisy = isPixelNoisyNew || isPixelNoisyOld;
	
	//**********  Fill the new DB !!
	
	
	if(isPixelNoisy) cout<<"Found a noisy pixel in "<<detid<<" at col,row "<<icol<<","<<jrow<<endl;
	if(isPixelNoisy) nnoisy++;
	if(isPixelDead) ndead++;
	
	
	//Set Pedestal
	if(isPixelDead)
	  theGainCalibrationDbInputOffline_->setDeadPixel(theSiPixelGainCalibrationGainPerColPedPerPixel);
	else if(isPixelNoisy)
	  theGainCalibrationDbInputOffline_->setNoisyPixel(theSiPixelGainCalibrationGainPerColPedPerPixel);
	else
	  theGainCalibrationDbInputOffline_->setDataPedestal(ped, theSiPixelGainCalibrationGainPerColPedPerPixel);
	  
	  
	//Set Gain
        if((jrow+1)%nrowsrocsplit==0){
	  if(isColumnDead)
	    theGainCalibrationDbInputOffline_->setDeadColumn(nrowsrocsplit,theSiPixelGainCalibrationGainPerColPedPerPixel);
	  else if(isColumnNoisy)
	    theGainCalibrationDbInputOffline_->setNoisyColumn(nrowsrocsplit,theSiPixelGainCalibrationGainPerColPedPerPixel);
	  else
	    theGainCalibrationDbInputOffline_->setDataGain(gainforthiscol[iglobalrow],nrowsrocsplit,theSiPixelGainCalibrationGainPerColPedPerPixel);
	}

      }//end of loop over rows
       
    }//end of loop over col
        
    SiPixelGainCalibrationOffline::Range offlinerange(theSiPixelGainCalibrationGainPerColPedPerPixel.begin(),theSiPixelGainCalibrationGainPerColPedPerPixel.end());
    if( !theGainCalibrationDbInputOffline_->put(detid,offlinerange,ncols) )
      edm::LogError("SiPixelGainCalibrationAnalysis")<<"warning: detid already exists for Offline (gain per col, ped per pixel) calibration database"<<std::endl;
    
  }//end of loop over Detids
  
  
  std::cout << " --- writing to DB!" << std::endl;
  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if(!mydbservice.isAvailable() ){
    edm::LogError("db service unavailable");
    return;
  }
  else{
    std::cout << "now doing SiPixelGainCalibrationOfflineRcd payload..." << std::endl; 
    if( mydbservice->isNewTagRequest("SiPixelGainCalibrationOfflineRcd") ){
      mydbservice->createNewIOV<SiPixelGainCalibrationOffline>(  theGainCalibrationDbInputOffline_,
								 mydbservice->beginOfTime(),
								 mydbservice->endOfTime(),
								 "SiPixelGainCalibrationOfflineRcd");
    }
    else{
      mydbservice->appendSinceTime<SiPixelGainCalibrationOffline>( theGainCalibrationDbInputOffline_, 
								   mydbservice->currentTime(),
								   "SiPixelGainCalibrationOfflineRcd");
    }
  }
  
  
  std::cout<<" ---> SUMMARY :"<<std::endl;
  std::cout<<" DB has now "<<nnoisy<<" noisy pixels"<<std::endl;
  std::cout<<" DB has now "<<ndead<<" dead pixels"<<std::endl;
   
}




void SiPixelGainCalibrationRejectNoisyAndDead::getNoisyPixels(){
  ifstream in;
  struct stat Stat;
  if(stat(noisypixellist_.c_str(),&Stat)) {
    std::cout<<"No file named "<<noisypixellist_<<std::endl;
    std::cout<<"If you don't want to insert noisy pixel flag, disable it using tag insertNoisyPixelsInDB "<<std::endl;
    return;
  }
  
  in.open(noisypixellist_.c_str());
  if(in.is_open()) {
    TString line;
    cout<<"opened"<<endl;
    char linetmp[201];
    while( in.getline( linetmp,200 ) ){
      line=linetmp;
      if(line.Contains("OFFLINE")){
	line.Remove(0,line.First(",")+9);
	TString detidstring = line;
	detidstring.Remove(line.First(" "),line.Sizeof());
	
	line.Remove(0,line.First(",")+20);
	TString row = line;
	row.Remove(line.First(","),line.Sizeof());
	line.Remove(0,line.First(",")+1);
	TString col = line;
	col.Remove(line.First(" "),line.Sizeof());
	
	std::cout<<"Found noisy pixel in DETID "<<detidstring<< " col,row "<<col<<","<<row<<std::endl;
	
	std::vector < std::pair <int,int> > tempvec;
	if(noisypixelkeeper.find(detidstring.Atoi()) != noisypixelkeeper.end())
	  tempvec = (noisypixelkeeper.find(detidstring.Atoi()))->second;
	
	std::pair <int,int> temppair(col.Atoi(),row.Atoi());
	tempvec.push_back(temppair);
	noisypixelkeeper[detidstring.Atoi()]=tempvec;
	  
	
	
      }
    }
  }
  /*
  for(std::map <int,std::vector<std::pair<int,int> > >::const_iterator it=noisypixelkeeper.begin();it!=noisypixelkeeper.end();it++) 
    for(int i=0;i<(it->second).size();i++)
      std::cout<<it->first<<"  "<<(it->second.at(i)).first<<"  "<<(it->second.at(i)).second<<std::endl;
  */	
}






void SiPixelGainCalibrationRejectNoisyAndDead::getDeadPixels(){

}

SiPixelGainCalibrationRejectNoisyAndDead::SiPixelGainCalibrationRejectNoisyAndDead(const edm::ParameterSet& iConfig):
  conf_(iConfig),
  SiPixelGainCalibrationService_(iConfig),
  insertnoisypixelsindb_(iConfig.getUntrackedParameter<int>("insertNoisyPixelsInDB",1)),
  noisypixellist_(iConfig.getUntrackedParameter<std::string>("noisyPixelList","noisypixel.txt"))
 

{
   //now do what ever initialization is needed

}


SiPixelGainCalibrationRejectNoisyAndDead::~SiPixelGainCalibrationRejectNoisyAndDead()
{

}

// ------------ method called to for each event  ------------
void
SiPixelGainCalibrationRejectNoisyAndDead::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  if(insertnoisypixelsindb_!=0) getNoisyPixels();
  fillDatabase(iSetup);

}


// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelGainCalibrationRejectNoisyAndDead::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelGainCalibrationRejectNoisyAndDead::endJob() {
}

//define this as a plug-in

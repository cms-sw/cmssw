#include "CalibTracker/SiStripAPVAnalysis/interface/DBPedestal.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
//#include "CalibTracker/SiStripAPVAnalysis/interface/ApvAnalysisFactory.h"
#include "CalibTracker/SiStripAPVAnalysis/interface/PedFactoryService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include <cmath>
#include <numeric>
#include <algorithm>
using namespace std;


DBPedestal::DBPedestal(int detId, int thisApv) :   
  alreadyRetrievedFromDB(false),
  numberOfEvents(0),
  detId_(0),
  thisApv_(0)

{
  //  if (0) cout << "Constructing TT6PedestalCalculator " << endl;
  //  eventsRequiredToCalibrate = evnt_ini; 
  //  eventsRequiredToUpdate    = evnt_iter;
  //  cutToAvoidSignal          = sig_cut;
  init();

  detId_=detId;
  thisApv_=thisApv;
}
//
// Initialization.
//
void DBPedestal::init() { 
  theRawNoise.clear();
  thePedestal.clear();
  //  thePedSum.clear();
  //  thePedSqSum.clear();
  //  theEventPerStrip.clear();
  theStatus.setCalibrating();
}
//
//  -- Destructor  
//
DBPedestal::~DBPedestal() {
  if (0) cout << "Destructing DBPedestal " << endl;
}
//
// -- Set Pedestal Update Status
//
void DBPedestal::updateStatus(){

/*  
  if (theStatus.isCalibrating() && numberOfEvents >= eventsRequiredToCalibrate) {
    theStatus.setUpdating();
  }
*/

}
//
// -- Initialize or Update (when needed) Pedestal Values
//
void DBPedestal::updatePedestal(ApvAnalysis::RawSignalType& in) {

   //  std::cout << " ++++++> ev 2 this should be the first event to get pedestal (due to post module feature)" << std::endl; 

   if ( numberOfEvents==2) {

   // if ( numberOfEvents==2 && !alreadyRetrievedFromDB) {
    initializePedestal(in);
    alreadyRetrievedFromDB = true;
  }

}

//
// -- Initialize Pedestal Values using a set of events (eventsRequiredToCalibrate)
//

void DBPedestal::initializePedestal(ApvAnalysis::RawSignalType& in) {
  
  PedService_ = edm::Service<PedFactoryService>().operator->();
  pedDB_ = PedService_->getPedDB(); 
 
  
  SiStripPedestals::Range range=pedDB_->getRange(detId_);
  
  edm::LogInfo(" DBPedestal");
  std::cout << std::dec; 
  std::cout  << "PED detid " << detId_ << " this apv " << thisApv_
	     << " \t for strip # 10 :::::\t"
	     << pedDB_->getPed(10,range)   << std::endl;
  
  
  double avVal=0.0;
  double rmsVal=0.0;
  
  
  std::cout << std::dec;
  for (int istrip=0; istrip<128; istrip++) {
    
    //   std::cout << "istrip " << istrip << "thisApv " << thisApv_ << std::endl;
    avVal=pedDB_->getPed(istrip+thisApv_*128,range);
    thePedestal.push_back(static_cast<float>(avVal));

    
  }
  
  
   
  //  thePedestal.push_back(static_cast<float>(avVal));
 theRawNoise.push_back(static_cast<float>(rmsVal));
  
  
}



//
// -- Update Pedestal Values when needed.
//
void DBPedestal::refinePedestal(ApvAnalysis::RawSignalType& in) {

   cout << "refinePedestal ::: nothing to do " << endl;

}
//
// Define New Event
// 
void DBPedestal::newEvent(){

   numberOfEvents++;
  // retrieve ped form DB only one time 
 

}

/*
 * \file EBBeamTask.cc
 *
 * $Date: 2006/05/05 20:12:03 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include <DQM/EcalBarrelMonitorTasks/interface/EBBeamTask.h>

EBBeamTask::EBBeamTask(const ParameterSet& ps){

  init_ = false;

}

EBBeamTask::~EBBeamTask(){

}

void EBBeamTask::beginJob(const EventSetup& c){

  ievt_ = 0;

}

void EBBeamTask::setup(void){

  init_ = true;

  DaqMonitorBEInterface* dbe = 0;

  // get hold of back-end interface
  dbe = Service<DaqMonitorBEInterface>().operator->();

  if ( dbe ) {
    dbe->setCurrentFolder("EcalBarrel/EBBeamTask");

  }

}

void EBBeamTask::endJob(){

  LogInfo("EBBeamTask") << "analyzed " << ievt_ << " events";

}

void EBBeamTask::analyze(const Event& e, const EventSetup& c){
  
  bool enable = false;
  
  Handle<EcalRawDataCollection> dcchs;
  
  Handle<EcalTBEventHeader> pEvH;

  try{
    e.getByLabel("ecalEBunpacker", dcchs);
    
    int nebc = dcchs->size();
    LogDebug("EBBeamTask") << "event: " << ievt_ << " DCC headers collection size: " << nebc;
    
    for ( EcalRawDataCollection::const_iterator dcchItr = dcchs->begin(); dcchItr != dcchs->end(); ++dcchItr ) {
      
      EcalDCCHeaderBlock dcch = (*dcchItr);
      
      if ( dcch.getRunType() == EcalDCCHeaderBlock::BEAMH4
	   || dcch.getRunType() == EcalDCCHeaderBlock::BEAMH2  ) enable = true;
    }
    
  }
  catch ( std::exception& ex) {
    LogDebug("EcalBeamTask") << " EcalRawDataCollection not in event. Trying EcalTBEventHeader." << std::endl;
    
    try {
      e.getByType(pEvH);
      enable = true;
    LogDebug("EcalBeamTask") << " EcalTBEventHeader found, instead." << std::endl;
    } 
    catch ( std::exception& ex ) {
      LogError("EBBeamTask") << "EcalTBEventHeader not present in event TOO!." << std::endl;
    }
    
  }
  
  
  if ( ! enable ) return;
  
  if ( ! init_ ) this->setup();

  
  
  Handle<EcalTBTDCRawInfo> pTDCRaw;
  const EcalTBTDCRawInfo* rawTDC=0;
  try {
    e.getByType( pTDCRaw );
    rawTDC = pTDCRaw.product(); // get a ptr to the product
  } catch ( std::exception& ex ) {
    LogError("EcalBeamTask") << "Error! can't get the product EcalTBTDCRawInfo" << std::endl;
  }


   Handle<EcalTBTDCRecInfo> pTDC;
   const EcalTBTDCRecInfo* recTDC=0;
   try {
     e.getByLabel( "tdcReco", "EcalTBTDCRecInfo", pTDC);
     recTDC = pTDC.product(); // get a ptr to the product
     LogInfo("EBBeamTask") << " TDC offset is: " <<      recTDC->offset() << endl;
  } catch ( std::exception& ex ) {
     LogError("EcalBeamTask") << "Error! can't get the product EcalTBTDCRecInfo" << std::endl;
   }


   Handle<EcalTBHodoscopeRawInfo> pHodoRaw;
   const EcalTBHodoscopeRawInfo* rawHodo=0;
   try {
     e.getByType( pHodoRaw );
     rawHodo = pHodoRaw.product(); // get a ptr to the product
     LogInfo("EcalBeamTask") << "hodoscopeRaw  Num planes: " <<  rawHodo->planes() 
			     << " channels in plane 1: "  <<  rawHodo->channels(1) << endl;
   } catch ( std::exception& ex ) {
     LogInfo("EcalBeamTask") << "Error! can't get the product EcalTBHodoscopeRawInfo" << std::endl;
   }

  
   Handle<EcalTBHodoscopeRecInfo> pHodo;
   const EcalTBHodoscopeRecInfo* recHodo=0;
   try {
     e.getByLabel( "hodoscopeReco", "EcalTBHodoscopeRecInfo", pHodo);
     recHodo = pHodo.product(); // get a ptr to the product
     LogInfo("EcalBeamTask") << "hodoscopeReco:    x: " << recHodo->posX()// to go away
              << "\ty: " << recHodo->posY()
	  << endl;
   } catch ( std::exception& ex ) {
     LogError("EcalBeamTask") << "Error! can't get the product EcalTBHodoscopeRecInfo" << std::endl;
   }

   if ( !rawTDC || !recTDC ||!rawHodo || !recHodo) return;

   const  std::vector<bool>& plane1 = rawHodo->hits(1);
   std::vector<bool>::const_iterator  fiber;
   for(fiber= plane1.begin(); fiber != plane1.end(); fiber++)
     {;}

   //   std::vector<bool>& hits(unsigned int plane) const { return planeHits_[plane].hits(); }


  ievt_++;

}


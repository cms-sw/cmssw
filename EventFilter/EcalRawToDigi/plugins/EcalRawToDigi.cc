#include "EventFilter/EcalRawToDigi/plugins/EcalRawToDigi.h"
#include "EventFilter/EcalRawToDigi/interface/EcalElectronicsMapper.h"
#include "EventFilter/EcalRawToDigi/interface/DCCDataUnpacker.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

EcalRawToDigi::EcalRawToDigi(edm::ParameterSet const& conf):
  
  //define the list of FED to be unpacked
  fedUnpackList_(conf.getParameter<std::vector<int> >("FEDs")),

  //define the ordered FED list
  orderedFedUnpackList_(conf.getParameter<std::vector<int> >("orderedFedList")),

  //define the ordered DCCId list
  orderedDCCIdList_(conf.getParameter<std::vector<int> >("orderedDCCIdList")),

  //get number of Xtal Time Samples
  numbXtalTSamples_(conf.getParameter<int>("numbXtalTSamples")),

  //Get number of Trigger Time Samples
  numbTriggerTSamples_(conf.getParameter<int>("numbTriggerTSamples")),
  
  //See if header unpacking is enabled
  headerUnpacking_(conf.getParameter<bool>("headerUnpacking")),
 
  //See if srp unpacking is enabled
  srpUnpacking_(conf.getParameter<bool>("srpUnpacking")),
  
  //See if tcc unpacking is enabled
  tccUnpacking_(conf.getParameter<bool>("tccUnpacking")),
  
  //See if fe unpacking is enabled
  feUnpacking_(conf.getParameter<bool>("feUnpacking")),
  
  //See if fe unpacking is enabled for mem box
  memUnpacking_(conf.getParameter<bool>("memUnpacking")), 

  //See if syncCheck is enabled
  syncCheck_(conf.getParameter<bool>("syncCheck")), 

  //See if feIdCheck is enabled
  feIdCheck_(conf.getParameter<bool>("feIdCheck")),

  // See if we want to keep data even if we have a mismatch between SR decision and block length
  forceToKeepFRdata_(conf.getParameter<bool>("forceToKeepFRData")),

  
  put_(conf.getParameter<bool>("eventPut")),
  


  REGIONAL_(conf.getParameter<bool>("DoRegional")),



  myMap_(0),
  
  theUnpacker_(0)

{
  
  first_ = true;
  DCCDataUnpacker::silentMode_ = conf.getUntrackedParameter<bool>("silentMode",false) ;
  
  if( numbXtalTSamples_ <6 || numbXtalTSamples_>64 || (numbXtalTSamples_-2)%4 ){
    std::ostringstream output;
    output      <<"\n Unsuported number of xtal time samples : "<<numbXtalTSamples_
                <<"\n Valid Number of xtal time samples are : 6,10,14,18,...,62"; 
    edm::LogError("IncorrectConfiguration")<< output.str();
    // todo : throw an execption
  }
  
  if( numbTriggerTSamples_ !=1 && numbTriggerTSamples_ !=4 && numbTriggerTSamples_ !=8  ){
    std::ostringstream output;
    output      <<"\n Unsuported number of trigger time samples : "<<numbTriggerTSamples_
                <<"\n Valid number of trigger time samples are :  1, 4 or 8"; 
    edm::LogError("IncorrectConfiguration")<< output.str();
    // todo : throw an execption
  }
  
  //NA : testing
  //nevts_=0;
  //RUNNING_TIME_=0;

  // if there are FEDs specified to unpack fill the vector of the fedUnpackList_
  // else fill with the entire ECAL fed range (600-670)
  if (fedUnpackList_.empty()) 
    for (int i=FEDNumbering::MINECALFEDID; i<=FEDNumbering::MAXECALFEDID; i++)
      fedUnpackList_.push_back(i);

  //print the FEDs to unpack to the logger
  std::ostringstream loggerOutput_;
  if(fedUnpackList_.size()!=0){
    for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
      loggerOutput_ << fedUnpackList_[i] << " ";
    edm::LogInfo("EcalRawToDigi") << "EcalRawToDigi will unpack FEDs ( " << loggerOutput_.str() << ")";
    LogDebug("EcalRawToDigi") << "EcalRawToDigi will unpack FEDs ( " << loggerOutput_.str() << ")";
  }
  
  edm::LogInfo("EcalRawToDigi")
    <<"\n ECAL RawToDigi configuration:"
    <<"\n Header  unpacking is "<<headerUnpacking_
    <<"\n SRP Bl. unpacking is "<<srpUnpacking_
    <<"\n TCC Bl. unpacking is "<<tccUnpacking_  
    <<"\n FE  Bl. unpacking is "<<feUnpacking_
    <<"\n MEM Bl. unpacking is "<<memUnpacking_
    <<"\n sync check is "<<syncCheck_
    <<"\n feID check is "<<feIdCheck_
    <<"\n force keep FR data is "<<forceToKeepFRdata_
    <<"\n";

  edm::InputTag dataLabel = conf.getParameter<edm::InputTag>("InputLabel");
  edm::InputTag fedsLabel = conf.getParameter<edm::InputTag>("FedLabel");

  // Producer products :
  produces<EBDigiCollection>("ebDigis"); 
  produces<EEDigiCollection>("eeDigis");
  produces<EBSrFlagCollection>();
  produces<EESrFlagCollection>();
  produces<EcalRawDataCollection>();
  produces<EcalPnDiodeDigiCollection>();
  produces<EcalTrigPrimDigiCollection>("EcalTriggerPrimitives");
  produces<EcalPSInputDigiCollection>("EcalPseudoStripInputs");
  
  // Integrity for xtal data
  produces<EBDetIdCollection>("EcalIntegrityGainErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors");
  produces<EBDetIdCollection>("EcalIntegrityChIdErrors");

  // Integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
  produces<EEDetIdCollection>("EcalIntegrityGainErrors");
  produces<EEDetIdCollection>("EcalIntegrityGainSwitchErrors");
  produces<EEDetIdCollection>("EcalIntegrityChIdErrors");

  // Integrity Errors
  produces<EcalElectronicsIdCollection>("EcalIntegrityTTIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityZSXtalIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityBlockSizeErrors");
 
  // Mem channels' integrity
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemTtIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemBlockSizeErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemChIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemGainErrors");

  dataToken_=consumes<FEDRawDataCollection>(dataLabel);
  if (REGIONAL_){
      fedsToken_=consumes<EcalListOfFEDS>(fedsLabel);
  }

  // Build a new Electronics mapper and parse default map file
  myMap_ = new EcalElectronicsMapper(numbXtalTSamples_,numbTriggerTSamples_);

  // in case of external  tsext file (deprecated by HLT environment) 
  //  bool readResult = myMap_->readDCCMapFile(conf.getParameter<std::string>("DCCMapFile",""));

  // use two arrays from cfg to establish DCCId:FedId. If they are empy, than use hard coded correspondence 
  bool readResult = myMap_->makeMapFromVectors(orderedFedUnpackList_, orderedDCCIdList_);
  // myMap::makeMapFromVectors() returns "true" always
  // need to be fixed?

  if(!readResult){
    edm::LogWarning("IncorrectConfiguration")
      << "Arrays orderedFedList and orderedDCCIdList are emply. "
         "Hard coded correspondence for DCCId:FedId will be used.";
    // edm::LogError("EcalRawToDigi")<<"\n unable to read file : "
    //   <<conf.getParameter<std::string>("DCCMapFile");
  }
  
  // Build a new ECAL DCC data unpacker
  theUnpacker_ = new DCCDataUnpacker(myMap_,headerUnpacking_,srpUnpacking_,tccUnpacking_,feUnpacking_,memUnpacking_,syncCheck_,feIdCheck_,forceToKeepFRdata_);
   
}


// print list of crystals with non-zero statuses
// this functions is only for debug purposes
void printStatusRecords(const DCCDataUnpacker* unpacker,
                        const EcalElectronicsMapping* mapping)
{
  // Endcap
  std::cout << "===> ENDCAP" << std::endl;
  for (int i = 0; i < EEDetId::kSizeForDenseIndexing; ++i) {
    const EEDetId id = EEDetId::unhashIndex(i);
    if (!id.null()) {
      // channel status
      const uint16_t code = unpacker->getChannelValue(id);
      
      if (code) {
        const EcalElectronicsId ei = mapping->getElectronicsId(id);
        
        // convert DCC ID (1 - 54) to FED ID (601 - 654)
        const int fed_id = unpacker->electronicsMapper()->getDCCId(ei.dccId());
        
        std::cout
          << " id " << id.rawId()
          << " -> (" << id.ix() << ", " << id.iy() << ", " << id.zside() << ") "
          << "(" << ei.dccId() << " : " << fed_id << ", " << ei.towerId() << ", " << ei.stripId() << ", " << ei.xtalId() << ") "
          << "status = " << code
          << std::endl;
      }
    }
  }
  std::cout << "<=== ENDCAP" << std::endl;
  
  std::cout << "===> BARREL" << std::endl;
  for (int i = 0; i < EBDetId::kSizeForDenseIndexing; ++i) {
    const EBDetId id = EBDetId::unhashIndex(i);
    if (!id.null()) {
      // channel status
      const uint16_t code = unpacker->getChannelValue(id);
      
      if (code) {
        const EcalElectronicsId ei = mapping->getElectronicsId(id);
        
        // convert DCC ID (1 - 54) to FED ID (601 - 654)
        const int fed_id = unpacker->electronicsMapper()->getDCCId(ei.dccId());
        
        std::cout
          << " id " << id.rawId()
          << " -> (" << id.ieta() << ", " << id.iphi() << ", " << id.zside() << ") "
          << "(" << ei.dccId() << " : " << fed_id << ", " << ei.towerId() << ", " << ei.stripId() << ", " << ei.xtalId() << ") "
          << "status = " << code
          << std::endl;
      }
    }
  }
  std::cout << "<=== BARREL" << std::endl;
}

void EcalRawToDigi::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("tccUnpacking",true);
  desc.add<edm::InputTag>("FedLabel",edm::InputTag("listfeds"));
  desc.add<bool>("srpUnpacking",true);
  desc.add<bool>("syncCheck",true);
  desc.add<bool>("feIdCheck",true);
  desc.addUntracked<bool>("silentMode",true);
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
  {
    std::vector<int> temp1;
    unsigned int nvec = 54;
    temp1.reserve(nvec);
    for (unsigned int i=0; i<nvec; i++) temp1.push_back(601+i);
    desc.add<std::vector<int> >("orderedFedList",temp1);
  }
  desc.add<bool>("eventPut",true);
  desc.add<int>("numbTriggerTSamples",1);
  desc.add<int>("numbXtalTSamples",10);
  {
    std::vector<int> temp1;
    unsigned int nvec = 54;
    temp1.reserve(nvec);
    for (unsigned int i=0; i<nvec; i++) temp1.push_back(1+i);
    desc.add<std::vector<int> >("orderedDCCIdList",temp1);
  }
  {
    std::vector<int> temp1;
    unsigned int nvec = 54;
    temp1.reserve(nvec);
    for (unsigned int i=0; i<nvec; i++) temp1.push_back(601+i);
    desc.add<std::vector<int> >("FEDs",temp1);
  }
  desc.add<bool>("DoRegional",false);
  desc.add<bool>("feUnpacking",true);
  desc.add<bool>("forceToKeepFRData",false);
  desc.add<bool>("headerUnpacking",true);
  desc.add<bool>("memUnpacking",true);
  descriptions.add("ecalRawToDigi",desc);
}


void EcalRawToDigi::beginRun(const edm::Run&, const edm::EventSetup& es)
{
  // channel status database
  edm::ESHandle<EcalChannelStatusMap> pChStatus;
  es.get<EcalChannelStatusRcd>().get(pChStatus);
  theUnpacker_->setChannelStatusDB(pChStatus.product());
  
  // uncomment following line to print list of crystals with bad status
  //edm::ESHandle<EcalElectronicsMapping> pEcalMapping;
  //es.get<EcalMappingRcd>().get(pEcalMapping);
  //const EcalElectronicsMapping* mapping = pEcalMapping.product();
  //printStatusRecords(theUnpacker_, mapping);
}


void EcalRawToDigi::produce(edm::Event& e, const edm::EventSetup& es)
{
  
  //double TIME_START = clock();
  //nevts_++; //NUNO


  if (first_) {
   watcher_.check(es);
   edm::ESHandle< EcalElectronicsMapping > ecalmapping;
   es.get< EcalMappingRcd >().get(ecalmapping);
   myMap_ -> setEcalElectronicsMapping(ecalmapping.product());
   
   first_ = false;

  }else{

    if ( watcher_.check(es) ) {    
      edm::ESHandle< EcalElectronicsMapping > ecalmapping;
      es.get< EcalMappingRcd >().get(ecalmapping);
      myMap_ -> deletePointers();
      myMap_ -> resetPointers();
      myMap_ -> setEcalElectronicsMapping(ecalmapping.product());
    }

  }

  // Get list of FEDS :
  std::vector<int> FEDS_to_unpack;
  if (REGIONAL_) {
        edm::Handle<EcalListOfFEDS> listoffeds;
        e.getByToken(fedsToken_, listoffeds);
        FEDS_to_unpack = listoffeds -> GetList();
  }



  // Step A: Get Inputs    

  edm::Handle<FEDRawDataCollection> rawdata;  
  e.getByToken(dataToken_,rawdata);


  // Step B: encapsulate vectors in actual collections and set unpacker pointers

  // create the collection of Ecal Digis
  std::auto_ptr<EBDigiCollection> productDigisEB(new EBDigiCollection);
  productDigisEB->reserve(1700);
  theUnpacker_->setEBDigisCollection(&productDigisEB);
  
  // create the collection of Ecal Digis
  std::auto_ptr<EEDigiCollection> productDigisEE(new EEDigiCollection);
  theUnpacker_->setEEDigisCollection(&productDigisEE);
  
  // create the collection for headers
  std::auto_ptr<EcalRawDataCollection> productDccHeaders(new EcalRawDataCollection);
  theUnpacker_->setDccHeadersCollection(&productDccHeaders); 

  // create the collection for invalid gains
  std::auto_ptr< EBDetIdCollection> productInvalidGains(new EBDetIdCollection);
  theUnpacker_->setInvalidGainsCollection(&productInvalidGains); 

  // create the collection for invalid gain Switch
  std::auto_ptr< EBDetIdCollection> productInvalidGainsSwitch(new EBDetIdCollection);
  theUnpacker_->setInvalidGainsSwitchCollection(&productInvalidGainsSwitch);
   
  // create the collection for invalid chids
  std::auto_ptr< EBDetIdCollection> productInvalidChIds(new EBDetIdCollection);
  theUnpacker_->setInvalidChIdsCollection(&productInvalidChIds);
  
  ///////////////// make EEDetIdCollections for these ones
    
  // create the collection for invalid gains
  std::auto_ptr<EEDetIdCollection> productInvalidEEGains(new EEDetIdCollection);
  theUnpacker_->setInvalidEEGainsCollection(&productInvalidEEGains); 
    
  // create the collection for invalid gain Switch
  std::auto_ptr<EEDetIdCollection> productInvalidEEGainsSwitch(new EEDetIdCollection);
  theUnpacker_->setInvalidEEGainsSwitchCollection(&productInvalidEEGainsSwitch);
    
  // create the collection for invalid chids
  std::auto_ptr<EEDetIdCollection> productInvalidEEChIds(new EEDetIdCollection);
  theUnpacker_->setInvalidEEChIdsCollection(&productInvalidEEChIds);

  ///////////////// make EEDetIdCollections for these ones    

  // create the collection for EB srflags       
  std::auto_ptr<EBSrFlagCollection> productEBSrFlags(new EBSrFlagCollection);
  theUnpacker_->setEBSrFlagsCollection(&productEBSrFlags);
  
  // create the collection for EB srflags       
  std::auto_ptr<EESrFlagCollection> productEESrFlags(new EESrFlagCollection);
  theUnpacker_->setEESrFlagsCollection(&productEESrFlags);

  // create the collection for ecal trigger primitives
  std::auto_ptr<EcalTrigPrimDigiCollection> productEcalTps(new EcalTrigPrimDigiCollection);
  theUnpacker_->setEcalTpsCollection(&productEcalTps);
  /////////////////////// collections for problems pertaining towers are already EE+EB communal

  // create the collection for ecal trigger primitives
  std::auto_ptr<EcalPSInputDigiCollection> productEcalPSs(new EcalPSInputDigiCollection);
  theUnpacker_->setEcalPSsCollection(&productEcalPSs);
  /////////////////////// collections for problems pertaining towers are already EE+EB communal

  // create the collection for invalid TTIds
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidTTIds(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidTTIdsCollection(&productInvalidTTIds);
 
   // create the collection for invalid TTIds
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidZSXtalIds(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidZSXtalIdsCollection(&productInvalidZSXtalIds);


 
  // create the collection for invalid BlockLengths
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidBlockLengths(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidBlockLengthsCollection(&productInvalidBlockLengths);

  // MEMs Collections
  // create the collection for the Pn Diode Digis
  std::auto_ptr<EcalPnDiodeDigiCollection> productPnDiodeDigis(new EcalPnDiodeDigiCollection);
  theUnpacker_->setPnDiodeDigisCollection(&productPnDiodeDigis);

  // create the collection for invalid Mem Tt id 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemTtIds(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidMemTtIdsCollection(& productInvalidMemTtIds);
  
  // create the collection for invalid Mem Block Size 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemBlockSizes(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidMemBlockSizesCollection(& productInvalidMemBlockSizes);
  
  // create the collection for invalid Mem Block Size 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemChIds(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidMemChIdsCollection(& productInvalidMemChIds);
 
  // create the collection for invalid Mem Gain Errors 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemGains(new EcalElectronicsIdCollection);
  theUnpacker_->setInvalidMemGainsCollection(& productInvalidMemGains); 
  //  double TIME_START = clock(); 
  

  // Step C: unpack all requested FEDs    
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {

    if (REGIONAL_) {
      std::vector<int>::const_iterator fed_it = find(FEDS_to_unpack.begin(), FEDS_to_unpack.end(), *i);
      if (fed_it == FEDS_to_unpack.end()) continue;
    }

  
    // get fed raw data and SM id
    const FEDRawData& fedData = rawdata->FEDData(*i);
    const size_t length = fedData.size();

    LogDebug("EcalRawToDigi") << "raw data length: " << length ;
    //if data size is not null interpret data
    if ( length >= EMPTYEVENTSIZE ){
      
      if(myMap_->setActiveDCC(*i)){

        const int smId = myMap_->getActiveSM();
        LogDebug("EcalRawToDigi") << "Getting FED = " << *i <<"(SM = "<<smId<<")"<<" data size is: " << length;

        const uint64_t* data = (uint64_t*) fedData.data();
        theUnpacker_->unpack(data, length, smId, *i);

        LogDebug("EcalRawToDigi") <<" in EE :"<<productDigisEE->size()
                                  <<" in EB :"<<productDigisEB->size();
      }
    }
    
  }// loop on FEDs
  
  //if(nevts_>1){   //NUNO
  //  double TIME_END = clock(); //NUNO
  //  RUNNING_TIME_ += TIME_END-TIME_START; //NUNO
  // }
  
  
  // Add collections to the event 
  
  if(put_){
    
    if( headerUnpacking_){ 
      e.put(productDccHeaders); 
    }
    
    if(feUnpacking_){
      productDigisEB->sort();
      e.put(productDigisEB,"ebDigis");
      productDigisEE->sort();
      e.put(productDigisEE,"eeDigis");
      e.put(productInvalidGains,"EcalIntegrityGainErrors");
      e.put(productInvalidGainsSwitch, "EcalIntegrityGainSwitchErrors");
      e.put(productInvalidChIds, "EcalIntegrityChIdErrors");
      // EE (leaving for now the same names as in EB)
      e.put(productInvalidEEGains,"EcalIntegrityGainErrors");
      e.put(productInvalidEEGainsSwitch, "EcalIntegrityGainSwitchErrors");
      e.put(productInvalidEEChIds, "EcalIntegrityChIdErrors");
      // EE
      e.put(productInvalidTTIds,"EcalIntegrityTTIdErrors");
      e.put(productInvalidZSXtalIds,"EcalIntegrityZSXtalIdErrors");
      e.put(productInvalidBlockLengths,"EcalIntegrityBlockSizeErrors");
      e.put(productPnDiodeDigis);
    }
    if(memUnpacking_){
      e.put(productInvalidMemTtIds,"EcalIntegrityMemTtIdErrors");
      e.put(productInvalidMemBlockSizes,"EcalIntegrityMemBlockSizeErrors");
      e.put(productInvalidMemChIds,"EcalIntegrityMemChIdErrors");
      e.put(productInvalidMemGains,"EcalIntegrityMemGainErrors");
    }
    if(srpUnpacking_){
      e.put(productEBSrFlags);
      e.put(productEESrFlags);
    }
    if(tccUnpacking_){
      e.put(productEcalTps,"EcalTriggerPrimitives");
      e.put(productEcalPSs,"EcalPseudoStripInputs");
    }
  }
  
//if(nevts_>1){   //NUNO
//  double TIME_END = clock(); //NUNO 
//  RUNNING_TIME_ += TIME_END-TIME_START; //NUNO
//}
  
}

EcalRawToDigi::~EcalRawToDigi()
{
  
  //cout << "EcalDCCUnpackingModule  " << "N events        " << (nevts_-1)<<endl;
  //cout << "EcalDCCUnpackingModule  " << " --- SETUP time " << endl;
  //cout << "EcalDCCUnpackingModule  " << "Time (sys)      " << SETUP_TIME_ << endl;
  //cout << "EcalDCCUnpackingModule  " << "Time in sec.    " << SETUP_TIME_/ CLOCKS_PER_SEC  << endl;
  //cout << "EcalDCCUnpackingModule  " << " --- Per event  " << endl;
  
  //RUNNING_TIME_ = RUNNING_TIME_ / (nevts_-1);
  
  //cout << "EcalDCCUnpackingModule  "<< "Time (sys)      " << RUNNING_TIME_  << endl;
  //cout << "EcalDCCUnpackingModule  "<< "Time in sec.    " << RUNNING_TIME_ / CLOCKS_PER_SEC  << endl;
  
  
  
  if(myMap_      ) delete myMap_;
  if(theUnpacker_) delete theUnpacker_;
  
}


#include "EventFilter/HcalRawToDigi/interface/HcalRawToDigi.h"

/*------------------------------------------------------------------*/
/* EcalRawToDigi::EcalRawToDigi                                     */
/* class constructor                                                */
/*------------------------------------------------------------------*/
EcalRawToDigi::EcalRawToDigi(edm::ParameterSet const& conf):
  //unpacker
  unpacker_(
	    conf.getUntrackedParameter<int>("EcalFirstFED",FEDNumbering::getEcalFEDIds().first),
	    conf.getParameter<int>("firstSample"),
	    conf.getParameter<int>("lastSample")),

  //
  filter_(conf.getParameter<bool>("FilterDataQuality"),
	  conf.getParameter<bool>("FilterDataQuality"),
	  false,
	  0, 
	  0, 
	  -1),

  //define the FED unpack list
  fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>())),

  //get first FED
  firstFED_(conf.getUntrackedParameter<int>("EcalFirstFED",FEDNumbering::getEcalFEDIds().first)),
  
  //unpack calibration
  unpackCalib_(conf.getUntrackedParameter<bool>("UnpackCalib",false)) {

  //if there are FEDs to unpack fill the vector of the fedUnpackList_
  if (fedUnpackList_.empty()) 
    for (int i=FEDNumbering::getEcalFEDIds().first; i<=FEDNumbering::getEcalFEDIds().second; i++)
      fedUnpackList_.push_back(i);
  
  //print the FEDs to unpack to the loger
  std::ostringstream loggerOutput_;
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    loggerOutput_ << fedUnpackList_[i] << " ";
  edm::LogInfo("ECAL") << "EcalRawToDigi will unpack FEDs ( " << loggerOutput_.str() << ")";
    
  // products produced...
  //  produces<HBHEDigiCollection>();
  //  produces<HFDigiCollection>();
  //  produces<HODigiCollection>();
  //  produces<HcalTrigPrimDigiCollection>();
  //  if (unpackCalib_)
  //    produces<HcalCalibDigiCollection>();

  //allocate a new mapper and parse default map file
  myMap_ = new DCCMapper();
  myMap_->readDCCMapFile("EventFilter/EcalRawToDigi/interface/DCCMap.txt");
}

/*-----------------------------------------------------------------------*/
/* EcalRawToDigi::produce                                                */
/* Functions that gets called by framework every event                   */
/*-----------------------------------------------------------------------*/
void EcalRawToDigi::produce(edm::Event& e, const edm::EventSetup& es) {

  /*
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawraw;  

  // edm::ProcessNameSelector s("PROD"); 
  e.getByType(rawraw);           // HACK!
  // get the mapping
  edm::ESHandle<HcalDbService> pSetup;
  es.get<HcalDbRecord>().get( pSetup );
  const HcalElectronicsMap* readoutMap=pSetup->getHcalMapping();


  // Step B: Create empty output  : three vectors for three classes...
  std::vector<HBHEDataFrame> hbhe;
  std::vector<HODataFrame> ho;
  std::vector<HFDataFrame> hf;
  std::vector<HcalTriggerPrimitiveDigi> htp;
  std::vector<HcalCalibDataFrame> hc;
*/

  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) {

    //get fed raw data
    const FEDRawData &fed = rawraw->FEDData(*i);
    
    //get SM id from dcc Id_ 
    ulong smId myMap_->getSMId(*i);
    
    unpacker_.unpack(fed,*readoutMap,hbhe,ho,hf,hc,htp);
  }

/*
  // Step B: encapsulate vectors in actual collections
  std::auto_ptr<HBHEDigiCollection> hbhe_prod(new HBHEDigiCollection()); 
  std::auto_ptr<HFDigiCollection> hf_prod(new HFDigiCollection());
  std::auto_ptr<HODigiCollection> ho_prod(new HODigiCollection());
  std::auto_ptr<HcalTrigPrimDigiCollection> htp_prod(new HcalTrigPrimDigiCollection());  

  hbhe_prod->swap_contents(hbhe);
  hf_prod->swap_contents(hf);
  ho_prod->swap_contents(ho);
  htp_prod->swap_contents(htp);

  // Step C2: filter FEDs, if required
  if (filter_.active()) {
    HBHEDigiCollection filtered_hbhe=filter_.filter(*hbhe_prod);
    HODigiCollection filtered_ho=filter_.filter(*ho_prod);
    HFDigiCollection filtered_hf=filter_.filter(*hf_prod);
    
    hbhe_prod->swap(filtered_hbhe);
    ho_prod->swap(filtered_ho);
    hf_prod->swap(filtered_hf);    
  }


  // Step D: Put outputs into event
  // just until the sorting is proven
  hbhe_prod->sort();
  ho_prod->sort();
  hf_prod->sort();
  htp_prod->sort();

  e.put(hbhe_prod);
  e.put(ho_prod);
  e.put(hf_prod);
  e.put(htp_prod);

  /// calib
  if (unpackCalib_) {
    std::auto_ptr<HcalCalibDigiCollection> hc_prod(new HcalCalibDigiCollection());
    hc_prod->swap_contents(hc);
    hc_prod->sort();
    e.put(hc_prod);
  }

*/
}

/*------------------------------------------------------*/
/* EcalRawToDigi::~EcalRawToDigi()                      */
/* Virtual destructor                                   */
/*------------------------------------------------------*/
EcalRawToDigi::~EcalRawToDigi() { 
  //free mapper
  delete myMap_;
}  

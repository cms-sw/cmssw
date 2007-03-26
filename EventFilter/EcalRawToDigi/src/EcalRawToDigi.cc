/** 
 * \class EcalRawToDigi
 *
 * This class takes care of unpacking ECAL's raw data info
 *
 * \author Pedro Silva (adapted from HcalRawToDigi and ECALTBRawToDigi)
 *
 * \version 1.0 
 * \date June 08, 2006  
 *
 */

#include "DataFormats/Common/interface/Handle.h"
#include "EventFilter/EcalRawToDigi/interface/EcalRawToDigi.h"
#include "EventFilter/EcalRawToDigi/src/DCCMapper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
          
/*------------------------------------------------------------------*/
/* EcalRawToDigi::EcalRawToDigi                                     */
/* class constructor                                                */
/*------------------------------------------------------------------*/
EcalRawToDigi::EcalRawToDigi(edm::ParameterSet const& conf):
  //define the FED unpack list
  fedUnpackList_(conf.getUntrackedParameter<std::vector<int> >("FEDs", std::vector<int>())),
  //get first FED
  firstFED_(conf.getUntrackedParameter<int>("EcalFirstFED",FEDNumbering::getEcalFEDIds().first))
{

  //if there are FEDs to unpack fill the vector of the fedUnpackList_
  if (fedUnpackList_.empty()) 
    for (int i=FEDNumbering::getEcalFEDIds().first; i<=FEDNumbering::getEcalFEDIds().second; i++)
      fedUnpackList_.push_back(i);
  
  //print the FEDs to unpack to the loger
  std::ostringstream loggerOutput_;
  for (unsigned int i=0; i<fedUnpackList_.size(); i++) 
    loggerOutput_ << fedUnpackList_[i] << " ";
  edm::LogInfo("EcalRawToDigi") << "EcalRawToDigi will unpack FEDs ( " << loggerOutput_.str() << ")";
    
  //formatter = new EcalTBDaqFormatter();
 
  // digis
  produces<EBDigiCollection>();
  produces<EcalPnDiodeDigiCollection>();
  produces<EcalRawDataCollection>();

  // crystals' integrity
  produces<EBDetIdCollection>("EcalIntegrityDCCSizeErrors");
  produces<EcalTrigTowerDetIdCollection>("EcalIntegrityTTIdErrors");
  produces<EcalTrigTowerDetIdCollection>("EcalIntegrityBlockSizeErrors");
  produces<EBDetIdCollection>("EcalIntegrityChIdErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchStayErrors");
  produces<EBDetIdCollection>("EcalIntegrityGainSwitchStayErrors");

  // mem channels' integrity
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemTtIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemBlockSize");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemChIdErrors");
  produces<EcalElectronicsIdCollection>("EcalIntegrityMemGainErrors");

  //allocate a new mapper and parse default map file
  myMap_ = new DCCMapper();
  myMap_->readDCCMapFile(conf.getUntrackedParameter<std::string>("DCCMapFile",""));

  formatter_ = new EcalDCCDaqFormatter();
  formatter_->setDCCMapper(myMap_);
  formatter_->setEcalFirstFED(firstFED_);
}

/*-----------------------------------------------------------------------*/
/* EcalRawToDigi::produce                                                */
/* Functions that gets called by framework every event                   */
/*-----------------------------------------------------------------------*/
void EcalRawToDigi::produce(edm::Event& e, const edm::EventSetup& es) {
  // Step A: Get Inputs 
  edm::Handle<FEDRawDataCollection> rawdata;  
  e.getByType(rawdata);
  
  // Step B: encapsulate vectors in actual collections

  // create the collection of Ecal Digis
  auto_ptr<EBDigiCollection> productEb(new EBDigiCollection);

  // create the collection of Ecal PN's
  auto_ptr<EcalPnDiodeDigiCollection> productPN(new EcalPnDiodeDigiCollection);
  
  //create the collection of Ecal DCC Header
  auto_ptr<EcalRawDataCollection> productDCCHeader(new EcalRawDataCollection);

  // create the collection of Ecal Integrity DCC Size
  auto_ptr<EBDetIdCollection> productDCCSize(new EBDetIdCollection);

  // create the collection of Ecal Integrity TT Id
  auto_ptr<EcalTrigTowerDetIdCollection> productTTId(new EcalTrigTowerDetIdCollection);

  // create the collection of Ecal Integrity TT Block Size
  auto_ptr<EcalTrigTowerDetIdCollection> productBlockSize(new EcalTrigTowerDetIdCollection);

  // create the collection of Ecal Integrity Ch Id
  auto_ptr<EBDetIdCollection> productChId(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain
  auto_ptr<EBDetIdCollection> productGain(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain Switch
  auto_ptr<EBDetIdCollection> productGainSwitch(new EBDetIdCollection);

  // create the collection of Ecal Integrity Gain Switch Stay
  auto_ptr<EBDetIdCollection> productGainSwitchStay(new EBDetIdCollection);

  // create the collection of Ecal Integrity Mem towerBlock_id errors
  auto_ptr<EcalElectronicsIdCollection> productMemTtId(new EcalElectronicsIdCollection);
  
  // create the collection of Ecal Integrity Mem gain errors
  auto_ptr< EcalElectronicsIdCollection> productMemBlockSize(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity Mem gain errors
  auto_ptr< EcalElectronicsIdCollection> productMemGain(new EcalElectronicsIdCollection);

  // create the collection of Ecal Integrity Mem ch_id errors
  auto_ptr<EcalElectronicsIdCollection> productMemChIdErrors(new EcalElectronicsIdCollection);
  
  
  // Step C: unpack all requested FEDs
  for (std::vector<int>::const_iterator i=fedUnpackList_.begin(); i!=fedUnpackList_.end(); i++) 
    {
      try{
	//get fed raw data and SM id
	const FEDRawData &fedData = rawdata->FEDData(*i);
	
	
	//if data size is no null interpret data
	if (fedData.size())
	  {
	      ulong smId_ = myMap_->getSMId(*i);
	      
	      //for debug purposes
	      edm::LogInfo("EcalRawToDigi") << "Getting FED nb: " << *i << " data size is: " << fedData.size() << endl;
	      
	      // do the data unpacking and fill the collections
	      formatter_->interpretRawData(fedData,  *productEb, *productPN, 
					   *productDCCHeader, *productDCCSize, *productTTId, *productBlockSize, 
					   *productChId, *productGain, *productGainSwitch, *productGainSwitchStay, 
					   *productMemTtId,  *productMemBlockSize,*productMemGain,  *productMemChIdErrors);      
	  }
	
	}
      catch(ECALParserException &e){
	  cout << "Exception caught after looping over feds: " << e.what() << endl;
      }
	catch(ECALParserBlockException &e){
	  cout << "Exception caught after looping over feds: " << e.what() << endl;
	}

    }// end loop on fed
  
  // Step D: Put outputs into event 
  e.put(productPN);
  e.put(productEb);
  e.put(productDCCHeader);
  
  e.put(productDCCSize,"EcalIntegrityDCCSizeErrors");
  e.put(productTTId,"EcalIntegrityTTIdErrors");
  e.put(productBlockSize,"EcalIntegrityBlockSizeErrors");
  e.put(productChId,"EcalIntegrityChIdErrors");
  e.put(productGain,"EcalIntegrityGainErrors");
  e.put(productGainSwitch,"EcalIntegrityGainSwitchErrors");
  e.put(productGainSwitchStay,"EcalIntegrityGainSwitchStayErrors");
  
  e.put(productMemTtId,"EcalIntegrityMemTtIdErrors");
  e.put(productMemBlockSize,"EcalIntegrityMemBlockSize");
  e.put(productMemChIdErrors,"EcalIntegrityMemChIdErrors");
  e.put(productMemGain,"EcalIntegrityMemGainErrors"); 
  
}


/*------------------------------------------------------*/
/* EcalRawToDigi::~EcalRawToDigi()                      */
/* Virtual destructor                                   */
/*------------------------------------------------------*/
EcalRawToDigi::~EcalRawToDigi() { 
  //free mapper
  delete myMap_;

  //free formatter
  //  delete formatter;
}  

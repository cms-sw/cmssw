#include "EventFilter/EcalRawToDigi/plugins/EcalRawToRecHitByproductProducer.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorkerRecord.h"
#include "EventFilter/EcalRawToDigi/interface/EcalUnpackerWorker.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/LazyGetter.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitComparison.h"
#include "FWCore/Framework/interface/Event.h"


EcalRawToRecHitByproductProducer::EcalRawToRecHitByproductProducer(const edm::ParameterSet& iConfig)
{
  workerName_ = iConfig.getParameter<std::string>("workerName");

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
}


// ------------ method called to produce the data  ------------
void
EcalRawToRecHitByproductProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // The next two lines are a temporary fix so that this
  // module will run without a fatal exception in unscheduled
  // mode. In scheduled mode, these two lines should have no
  // effect, but in unscheduled mode this will ensure the module
  // that creates the objects this module puts into the Event
  // is executed before this module.  This does NOT ensure the
  // objects are properly filled.  The plan is that this part
  // of the code will be completely rewritten in the near future.
  // In this rewrite the EcalUnpackerWorker will no longer be
  // written to either the Event or EventSetup and this problem
  // will be resolved in a permanent manner. 
  edm::Handle<edm::LazyGetter<EcalRecHit> > lgetter;
  iEvent.getByLabel("hltEcalRawToRecHitFacility", lgetter);

  //retrieve the unpacker worker
  edm::ESHandle<EcalUnpackerWorkerBase> workerESH;
  iSetup.get<EcalUnpackerWorkerRecord>().get(workerName_, workerESH);
  const EcalUnpackerWorker * worker =  dynamic_cast<const EcalUnpackerWorker *>(&*workerESH);
  if (!worker) 
    edm::LogError("IncorrectConfiguration")<<"worker "<< workerName_ <<" could not be cast in EcalUnpackerWorker type."
					       <<" this must be a configuration mistake. Expect a segfault right now.";

  LogDebug("EcalRawToRecHit|Byproducts")<<"worker retrieved.";

  //prepare output collections
  std::auto_ptr<EBDigiCollection> productDigisEB(worker->productDigisEB);
  
  // create the collection of Ecal Digis
  std::auto_ptr<EEDigiCollection> productDigisEE(worker->productDigisEE);

  // create the collection for headers
  std::auto_ptr<EcalRawDataCollection> productDccHeaders(worker->productDccHeaders);

  // create the collection for invalid gains
  std::auto_ptr< EBDetIdCollection> productInvalidGains(worker->productInvalidGains);

  // create the collection for invalid gain Switch
  std::auto_ptr< EBDetIdCollection> productInvalidGainsSwitch(worker->productInvalidGainsSwitch);
  
  // create the collection for invalid chids
  std::auto_ptr< EBDetIdCollection> productInvalidChIds(worker->productInvalidChIds);
  
  ///////////////// make EEDetIdCollections for these ones
    
  // create the collection for invalid gains
  std::auto_ptr<EEDetIdCollection> productInvalidEEGains(worker->productInvalidEEGains);
    
  // create the collection for invalid gain Switch
  std::auto_ptr<EEDetIdCollection> productInvalidEEGainsSwitch(worker->productInvalidEEGainsSwitch);
    
  // create the collection for invalid chids
  std::auto_ptr<EEDetIdCollection> productInvalidEEChIds(worker->productInvalidEEChIds);
  
  ///////////////// make EEDetIdCollections for these ones    
    
  // create the collection for EB srflags       
  std::auto_ptr<EBSrFlagCollection> productEBSrFlags(worker->productEBSrFlags);
  
  // create the collection for EB srflags       
  std::auto_ptr<EESrFlagCollection> productEESrFlags(worker->productEESrFlags);
  
  // create the collection for ecal trigger primitives
  std::auto_ptr<EcalTrigPrimDigiCollection> productEcalTps(worker->productTps);

  // create the collection for ecal trigger primitives
  std::auto_ptr<EcalPSInputDigiCollection> productEcalPSs(worker->productPSs);

  /////////////////////// collections for problems pertaining towers are already EE+EB communal

  // create the collection for invalid TTIds
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidTTIds(worker->productInvalidTTIds);
 
  // create the collection for invalid XtalIds
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidZSXtalIds(worker->productInvalidZSXtalIds);
 
  // create the collection for invalid BlockLengths
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidBlockLengths(worker->productInvalidBlockLengths);
  
  // MEMs Collections
  // create the collection for the Pn Diode Digis
  std::auto_ptr<EcalPnDiodeDigiCollection> productPnDiodeDigis(worker->productPnDiodeDigis);
  
  // create the collection for invalid Mem Tt id 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemTtIds(worker->productInvalidMemTtIds);
  
  // create the collection for invalid Mem Block Size 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemBlockSizes(worker->productInvalidMemBlockSizes);
  
  // create the collection for invalid Mem Block Size 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemChIds(worker->productInvalidMemChIds);
  
  // create the collection for invalid Mem Gain Errors 
  std::auto_ptr<EcalElectronicsIdCollection> productInvalidMemGains(worker->productInvalidMemGains);
  

  //---------------------------   
  //write outputs to the event
  //---------------------------   

  iEvent.put(productDigisEB,"ebDigis");
  iEvent.put(productDigisEE,"eeDigis");
  iEvent.put(productDccHeaders); 
  iEvent.put(productInvalidGains,"EcalIntegrityGainErrors");
  iEvent.put(productInvalidGainsSwitch, "EcalIntegrityGainSwitchErrors");
  iEvent.put(productInvalidChIds, "EcalIntegrityChIdErrors");
  // EE (leaving for now the same names as in EB)
  iEvent.put(productInvalidEEGains,"EcalIntegrityGainErrors");
  iEvent.put(productInvalidEEGainsSwitch, "EcalIntegrityGainSwitchErrors");
  iEvent.put(productInvalidEEChIds, "EcalIntegrityChIdErrors");
  // EE
  iEvent.put(productInvalidTTIds,"EcalIntegrityTTIdErrors");
  iEvent.put(productInvalidZSXtalIds,"EcalIntegrityZSXtalIdErrors");
  iEvent.put(productInvalidBlockLengths,"EcalIntegrityBlockSizeErrors");
  iEvent.put(productPnDiodeDigis);
  // errors  
  iEvent.put(productInvalidMemTtIds,"EcalIntegrityMemTtIdErrors");
  iEvent.put(productInvalidMemBlockSizes,"EcalIntegrityMemBlockSizeErrors");
  iEvent.put(productInvalidMemChIds,"EcalIntegrityMemChIdErrors");
  iEvent.put(productInvalidMemGains,"EcalIntegrityMemGainErrors");

  // flags
  iEvent.put(productEBSrFlags);
  iEvent.put(productEESrFlags);

  // trigger primitives 
  iEvent.put(productEcalTps,"EcalTriggerPrimitives");
  iEvent.put(productEcalPSs,"EcalPseudoStripInputs");

  //make new collections.
  worker->update(iEvent);
}

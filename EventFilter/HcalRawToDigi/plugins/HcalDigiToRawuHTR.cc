#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "EventFilter/HcalRawToDigi/interface/HcalUHTRData.h"
#include "DataFormats/HcalDigi/interface/HcalQIESample.h"

#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "FWCore/Utilities/interface/CRC16.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

#include "PackerHelp.cc"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

/* QUESTION: what about dual FED readout? */
/* QUESTION: what do I do if the number of 16-bit words
   are not divisible by 4? -- these need to
   fit into the 64-bit words of the FEDRawDataFormat */


using namespace std;

class HcalDigiToRawuHTR : public edm::EDProducer {
public:
  explicit HcalDigiToRawuHTR(const edm::ParameterSet&);
  ~HcalDigiToRawuHTR();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  void getData(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  int _verbosity;
  std::string electronicsMapLabel_;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  edm::EDGetTokenT<HcalDataFrameContainer<QIE10DataFrame> > tok_QIE10DigiCollection_;
  edm::Handle<QIE10DigiCollection> qie10DigiCollection;
  edm::EDGetTokenT<HcalDataFrameContainer<QIE11DataFrame> > tok_QIE11DigiCollection_;
  edm::Handle<QIE11DigiCollection> qie11DigiCollection;
  edm::EDGetTokenT<HBHEDigiCollection> tok_HBHEDigiCollection_;
  edm::Handle<HBHEDigiCollection> hbheDigiCollection;
  edm::EDGetTokenT<HFDigiCollection> tok_HFDigiCollection_;
  edm::Handle<HFDigiCollection> hfDigiCollection;
  edm::EDGetTokenT<HcalTrigPrimDigiCollection> tok_TPDigiCollection_;
  edm::Handle<HcalTrigPrimDigiCollection> tpDigiCollection;

};

HcalDigiToRawuHTR::HcalDigiToRawuHTR(const edm::ParameterSet& iConfig) :
  _verbosity(iConfig.getUntrackedParameter<int>("Verbosity")),
  electronicsMapLabel_(iConfig.getUntrackedParameter<std::string>("ElectronicsMap",""))
{
  produces<FEDRawDataCollection>("");
  tok_QIE10DigiCollection_ = consumes<HcalDataFrameContainer<QIE10DataFrame> >(edm::InputTag("hcalDigis"));
  tok_QIE11DigiCollection_ = consumes<HcalDataFrameContainer<QIE11DataFrame> >(edm::InputTag("hcalDigis"));
  tok_HBHEDigiCollection_ = consumes<HBHEDigiCollection >(edm::InputTag("hcalDigis"));
  tok_HFDigiCollection_ = consumes<HFDigiCollection>(edm::InputTag("hcalDigis"));
  tok_TPDigiCollection_ = consumes<HcalTrigPrimDigiCollection>(edm::InputTag("hcalDigis"));
}

HcalDigiToRawuHTR::~HcalDigiToRawuHTR(){}

void HcalDigiToRawuHTR::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  using namespace edm;

  edm::ESHandle<HcalDbService> pSetup;
  iSetup.get<HcalDbRecord>().get( pSetup );
  edm::ESHandle<HcalElectronicsMap> item;
  iSetup.get<HcalElectronicsMapRcd>().get(electronicsMapLabel_,item);
  const HcalElectronicsMap* readoutMap = item.product();

  //collection to be inserted into event
  std::auto_ptr<FEDRawDataCollection> fed_buffers(new FEDRawDataCollection());
  
  //
  //  Extracting All the Collections containing useful Info
  iEvent.getByToken(tok_QIE10DigiCollection_,qie10DigiCollection);
  const QIE10DigiCollection& qie10dc=*(qie10DigiCollection);
  iEvent.getByToken(tok_QIE11DigiCollection_,qie11DigiCollection);
  const QIE11DigiCollection& qie11dc=*(qie11DigiCollection);
  iEvent.getByToken(tok_HBHEDigiCollection_,hbheDigiCollection);
  const HBHEDigiCollection& qie8hbhedc=*(hbheDigiCollection);
  iEvent.getByToken(tok_HFDigiCollection_,hfDigiCollection);
  const HFDigiCollection& qie8hfdc=*(hfDigiCollection);
  iEvent.getByToken(tok_TPDigiCollection_,tpDigiCollection);
  const HcalTrigPrimDigiCollection& qietpdc=*(tpDigiCollection);

  // first argument is the fedid (minFEDID+crateId)
  map<int,HCalFED*> fedMap;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // QIE10 precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  uHTRpacker uhtrs;
  // loop over each digi and allocate memory for each

  for (unsigned int j=0; j < qie10dc.size(); j++){
    QIE10DataFrame qiedf = static_cast<QIE10DataFrame>(qie10dc[j]);
    DetId detid = qiedf.detid();
    HcalElectronicsId eid(readoutMap->lookup(detid));
    int crateId = eid.crateId();
    int slotId = eid.slot();
    int uhtrIndex = ((slotId&0xF)<<8) | (crateId&0xFF);

    /* Defining a custom index that will encode only
       the information about the crate and slot of a 
       given channel:   crate: bits 0-7
                        slot:  bits 8-12 */

    if( ! uhtrs.exist( uhtrIndex ) ){
      uhtrs.newUHTR( uhtrIndex );
    }
    uhtrs.addChannel(uhtrIndex,qiedf,_verbosity);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // QIE11 precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  //uHTRpacker uhtrs;
  // loop over each digi and allocate memory for each
  for (unsigned int j=0; j < qie11dc.size(); j++){
    QIE11DataFrame qiedf = static_cast<QIE11DataFrame>(qie11dc[j]);
    DetId detid = qiedf.detid();
    HcalElectronicsId eid(readoutMap->lookup(detid));
    int crateId = eid.crateId();
    int slotId = eid.slot();
    int uhtrIndex = ((slotId&0xF)<<8) | (crateId&0xFF);

    if( ! uhtrs.exist(uhtrIndex) ){
      uhtrs.newUHTR( uhtrIndex );
    }
    uhtrs.addChannel(uhtrIndex,qiedf,_verbosity);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // HF (QIE8) precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // loop over each digi and allocate memory for each
  for(HFDigiCollection::const_iterator qiedf=qie8hfdc.begin();qiedf!=qie8hfdc.end();qiedf++){
    //HFDataFrame qiedf = static_cast<HFDataFrame>(qie8hfdc[j]);
    HcalElectronicsId eid = qiedf->elecId();
    int crateId = eid.crateId();
    int slotId = eid.slot();
    int uhtrIndex = (crateId&0xFF) | ((slotId&0xF)<<8) ; 

    if( ! uhtrs.exist(uhtrIndex) ){
      uhtrs.newUHTR( uhtrIndex );
    }
    uhtrs.addChannel(uhtrIndex,qiedf,_verbosity);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // HBHE (QIE8) precision data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // loop over each digi and allocate memory for each
  for(HBHEDigiCollection::const_iterator qiedf=qie8hbhedc.begin();qiedf!=qie8hbhedc.end();qiedf++){
    //HFDataFrame qiedf = static_cast<HFDataFrame>(qie8hbhedc[j]);
    HcalElectronicsId eid = qiedf->elecId();
    int crateId = eid.crateId();
    int slotId = eid.slot();
    int uhtrIndex = (crateId&0xFF) | ((slotId&0xF)<<8) ; 

    if( ! uhtrs.exist(uhtrIndex) ){
      uhtrs.newUHTR( uhtrIndex );
    }
    uhtrs.addChannel(uhtrIndex,qiedf,_verbosity);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // TP data
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  // loop over each digi and allocate memory for each

  // --- I left off here ... need to specify and include the correct digi collection ---

  for(HcalTrigPrimDigiCollection::const_iterator qiedf=qietpdc.begin();qiedf!=qietpdc.end();qiedf++){
    //HFDataFrame qiedf = static_cast<HFDataFrame>(qie8hbhedc[j]);
    HcalElectronicsId eid(qiedf->id().rawId());
    
    int crateId = eid.crateId();
    int slotId = eid.slot();
    int uhtrIndex = (crateId&0xFF) | ((slotId&0xF)<<8) ; 

    if( ! uhtrs.exist(uhtrIndex) ){
      uhtrs.newUHTR( uhtrIndex );
    }
    uhtrs.addChannel(uhtrIndex,qiedf,_verbosity);
  }

  // -----------------------------------------------------
  // -----------------------------------------------------
  // loop over each uHTR and format data
  // -----------------------------------------------------
  // -----------------------------------------------------
  // loop over each uHTR and format data
  int idxuhtr =-1;
  for( UHTRMap::iterator uhtr = uhtrs.uhtrs.begin() ; uhtr != uhtrs.uhtrs.end() ; ++uhtr){

    idxuhtr ++;
   
    uint64_t crateId = (uhtr->first)&0xFF;
    uint64_t slotId =  (uhtr->first&0xF00)>>8;

    uhtrs.finalizeHeadTail(&(uhtr->second),_verbosity);
    int fedId = FEDNumbering::MINHCALuTCAFEDID + crateId;
    if( fedMap.find(fedId) == fedMap.end() ){
      /* QUESTION: where should the orbit number come from? */
      fedMap[fedId] = new HCalFED(fedId,iEvent.id().event(),iEvent.orbitNumber(),iEvent.bunchCrossing());
    }
    fedMap[fedId]->addUHTR(uhtr->second,crateId,slotId);
  }// end loop over uhtr containers

  /* ------------------------------------------------------
     ------------------------------------------------------
           putting together the FEDRawDataCollection
     ------------------------------------------------------
     ------------------------------------------------------ */
  for( map<int,HCalFED*>::iterator fed = fedMap.begin() ; fed != fedMap.end() ; ++fed ){

    int fedId = fed->first;

    FEDRawData* rawData =  fed->second->formatFEDdata(); // fed->second.fedData;
    FEDRawData& fedRawData = fed_buffers->FEDData(fedId);
    fedRawData = *rawData;
    delete rawData;

    FEDHeader hcalFEDHeader(fedRawData.data());
//    hcalFEDHeader.set(fedRawData.data(), 1, iEvent.id().event(), 1, fedId);
    hcalFEDHeader.set(fedRawData.data(), 1, iEvent.id().event(), iEvent.bunchCrossing(), fedId);
    FEDTrailer hcalFEDTrailer(fedRawData.data()+(fedRawData.size()-8));
    hcalFEDTrailer.set(fedRawData.data()+(fedRawData.size()-8), fedRawData.size()/8, evf::compute_crc(fedRawData.data(),fedRawData.size()), 0, 0);

  }// end loop over FEDs with data

  iEvent.put(fed_buffers);
  
}


// ------------ method called once each job just before starting event loop  ------------
void HcalDigiToRawuHTR::beginJob(){}

// ------------ method called once each job just after ending the event loop  ------------
void HcalDigiToRawuHTR::endJob(){}

// ------------ method called when starting to processes a run  ------------
void HcalDigiToRawuHTR::beginRun(edm::Run const&, edm::EventSetup const&){}

// ------------ method called when ending the processing of a run  ------------
void HcalDigiToRawuHTR::endRun(edm::Run const&, edm::EventSetup const&){}

// ------------ method called when starting to processes a luminosity block  ------------
void HcalDigiToRawuHTR::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&){}

// ------------ method called when ending the processing of a luminosity block  ------------
void HcalDigiToRawuHTR::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&){}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void HcalDigiToRawuHTR::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalDigiToRawuHTR);

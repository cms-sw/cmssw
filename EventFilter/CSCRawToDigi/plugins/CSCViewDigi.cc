/// -*- C++ -*-
//
// Package:    CSCViewDigi
// Class:      CSCViewDigi
// 
/**\class CSCViewDigi CSCViewDigi.cc 

 Location: EventFilter/CSCRawToDigi/plugins/CSCViewDigi.cc 

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alexandre Sakharov
//         Created:  Sun May 10 15:43:28 CEST 2009
// $Id: CSCViewDigi.cc,v 1.4 2010/06/11 15:50:27 asakharo Exp $
//
//

#include "EventFilter/CSCRawToDigi/interface/CSCViewDigi.h"
//#include "CSCViewDigi.h"

//
// constructors and destructor
//
CSCViewDigi::CSCViewDigi(const edm::ParameterSet& conf)
: wireDigiTag_(conf.getParameter<edm::InputTag>("wireDigiTag")),
 alctDigiTag_(conf.getParameter<edm::InputTag>("alctDigiTag")),
 clctDigiTag_(conf.getParameter<edm::InputTag>("clctDigiTag")),
 corrclctDigiTag_(conf.getParameter<edm::InputTag>("corrclctDigiTag")),
 stripDigiTag_(conf.getParameter<edm::InputTag>("stripDigiTag")),
 comparatorDigiTag_(conf.getParameter<edm::InputTag>("comparatorDigiTag")),
 rpcDigiTag_(conf.getParameter<edm::InputTag>("rpcDigiTag")),
 statusDigiTag_(conf.getParameter<edm::InputTag>("statusDigiTag")),
 DDUstatusDigiTag_(conf.getParameter<edm::InputTag>("DDUstatusDigiTag")),
 DCCstatusDigiTag_(conf.getParameter<edm::InputTag>("DCCstatusDigiTag"))
{
WiresDigiDump=conf.getUntrackedParameter<bool>("WiresDigiDump", false);
StripDigiDump=conf.getUntrackedParameter<bool>("StripDigiDump", false);
AlctDigiDump=conf.getUntrackedParameter<bool>("AlctDigiDump", false);
ClctDigiDump=conf.getUntrackedParameter<bool>("ClctDigiDump", false);
CorrClctDigiDump=conf.getUntrackedParameter<bool>("CorrClctDigiDump", false);
ComparatorDigiDump=conf.getUntrackedParameter<bool>("ComparatorDigiDump", false);
RpcDigiDump=conf.getUntrackedParameter<bool>("RpcDigiDump", false);
StatusDigiDump=conf.getUntrackedParameter<bool>("StatusDigiDump", false);
DDUStatusDigiDump=conf.getUntrackedParameter<bool>("DDUStatus",false);
DCCStatusDigiDump=conf.getUntrackedParameter<bool>("DCCStatus",false);
}


CSCViewDigi::~CSCViewDigi()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
CSCViewDigi::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::Handle<CSCWireDigiCollection> wires;
   edm::Handle<CSCStripDigiCollection> strips;
   edm::Handle<CSCComparatorDigiCollection> comparators;
   edm::Handle<CSCRPCDigiCollection> rpcs;
   edm::Handle<CSCALCTDigiCollection> alcts;
   edm::Handle<CSCCLCTDigiCollection> clcts;
   edm::Handle<CSCCorrelatedLCTDigiCollection> corrclcts;
   edm::Handle<CSCDCCFormatStatusDigiCollection> statusdigis;
   edm::Handle<CSCDDUStatusDigiCollection> DDUstatusdigi;
   edm::Handle<CSCDCCStatusDigiCollection> DCCstatusdigi;

   iEvent.getByLabel(wireDigiTag_,wires);
   iEvent.getByLabel(stripDigiTag_,strips);
   iEvent.getByLabel(comparatorDigiTag_,comparators);
   iEvent.getByLabel(rpcDigiTag_,rpcs);
   iEvent.getByLabel(alctDigiTag_,alcts);
   iEvent.getByLabel(clctDigiTag_,clcts);
   iEvent.getByLabel(corrclctDigiTag_,corrclcts);
   
   if(StatusDigiDump)
    iEvent.getByLabel(statusDigiTag_,statusdigis);
   
   if(DDUStatusDigiDump)
    iEvent.getByLabel(DDUstatusDigiTag_,DDUstatusdigi);

   if(DCCStatusDigiDump)
    iEvent.getByLabel(DCCstatusDigiTag_,DCCstatusdigi);
  
  
  if(WiresDigiDump){ 
   std::cout << std::endl;
   std::cout << "Event " << iEvent.id() << std::endl;
   std::cout << std::endl;
   std::cout << "********WIRES Digis********" << std::endl;
     for (CSCWireDigiCollection::DigiRangeIterator j=wires->begin(); j!=wires->end(); j++) {
         std::cout << "Wire digis from "<< CSCDetId((*j).first) << std::endl;
         std::vector<CSCWireDigi>::const_iterator digiItr = (*j).second.first;
         std::vector<CSCWireDigi>::const_iterator last = (*j).second.second;
         for( ; digiItr != last; ++digiItr) {
         digiItr->print();
         }
      }
   }
   
   if(StripDigiDump){ 
    std::cout << std::endl;
    std::cout << "Event " << iEvent.id() << std::endl;
    std::cout << std::endl;
    std::cout << "********STRIPS Digis********" << std::endl;
     for (CSCStripDigiCollection::DigiRangeIterator j=strips->begin(); j!=strips->end(); j++) {
         std::cout << "Strip digis from "<< CSCDetId((*j).first) << std::endl;
         std::vector<CSCStripDigi>::const_iterator digiItr = (*j).second.first;
         std::vector<CSCStripDigi>::const_iterator last = (*j).second.second;
         for( ; digiItr != last; ++digiItr) {
         digiItr->print();
         }
      }
   }
   
   if(ComparatorDigiDump){
    std::cout << std::endl;
    std::cout << "Event " << iEvent.id() << std::endl;
    std::cout << std::endl;
    std::cout << "********COMPARATOR Digis********" << std::endl;
     for (CSCComparatorDigiCollection::DigiRangeIterator j=comparators->begin(); j!=comparators->end(); j++) {
         std::cout << "Comparator digis from "<< CSCDetId((*j).first) << std::endl;
         std::vector<CSCComparatorDigi>::const_iterator digiItr = (*j).second.first;
         std::vector<CSCComparatorDigi>::const_iterator last = (*j).second.second;
         for( ; digiItr != last; ++digiItr) {
         digiItr->print();
         }
      }
   }
   
   if(RpcDigiDump){ 
    std::cout << std::endl;
    std::cout << "Event " << iEvent.id() << std::endl;
    std::cout << std::endl;
    std::cout << "********RPC Digis********" << std::endl;
     for (CSCRPCDigiCollection::DigiRangeIterator j=rpcs->begin(); j!=rpcs->end(); j++) {
         std::cout << "RPC digis from "<< CSCDetId((*j).first) << std::endl;
         std::vector<CSCRPCDigi>::const_iterator digiItr = (*j).second.first;
         std::vector<CSCRPCDigi>::const_iterator last = (*j).second.second;
         for( ; digiItr != last; ++digiItr) {
         digiItr->print();
         }
      }
   }
 
  if(AlctDigiDump){
   std::cout << std::endl;
   std::cout << "Event " << iEvent.id() << std::endl;
   std::cout << std::endl;
   std::cout << "********ALCT Digis********" << std::endl;
    for (CSCALCTDigiCollection::DigiRangeIterator j=alcts->begin(); j!=alcts->end(); j++) {
        std::vector<CSCALCTDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCALCTDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
        digiItr->print();
        }
     }
   }
   
   if(ClctDigiDump){
    std::cout << std::endl;
    std::cout << "Event " << iEvent.id() << std::endl;
    std::cout << std::endl;
    std::cout << "********CLCT Digis********" << std::endl;
    for (CSCCLCTDigiCollection::DigiRangeIterator j=clcts->begin(); j!=clcts->end(); j++) {
        std::vector<CSCCLCTDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCCLCTDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
        digiItr->print();
        }
     }
   }
   
   if(CorrClctDigiDump){
    std::cout << std::endl;
    std::cout << "Event " << iEvent.id() << std::endl;
    std::cout << std::endl;
    std::cout << "********CorrelatedCLCT Digis********" << std::endl;
    for (CSCCorrelatedLCTDigiCollection::DigiRangeIterator j=corrclcts->begin(); j!=corrclcts->end(); j++) {
        std::vector<CSCCorrelatedLCTDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCCorrelatedLCTDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
        digiItr->print();
        }
     }
   }
   
   if(StatusDigiDump){
    std::cout << std::endl;
    std::cout << "Event " << iEvent.id() << std::endl;
    std::cout << std::endl;
    std::cout << "********STATUS Digis********" << std::endl;
    for (CSCDCCFormatStatusDigiCollection::DigiRangeIterator j=statusdigis->begin(); j!=statusdigis->end(); j++) {
        std::vector<CSCDCCFormatStatusDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCDCCFormatStatusDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
        digiItr->print();
        }
     }
   }

   if(DDUStatusDigiDump){
     std::cout << std::endl;
     std::cout << "Event " << iEvent.id() << std::endl;
     std::cout << std::endl;
     std::cout << "********DDU STATUS Digis********" << std::endl;
     for (CSCDDUStatusDigiCollection::DigiRangeIterator j=DDUstatusdigi->begin(); j!=DDUstatusdigi->end(); j++){
        std::vector<CSCDDUStatusDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCDDUStatusDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
        digiItr->print();
        }
     }
   }

   if(DCCStatusDigiDump){
     std::cout << std::endl;
     std::cout << "Event " << iEvent.id() << std::endl;
     std::cout << std::endl;
     std::cout << "********DCC STATUS Digis********" << std::endl;
     for (CSCDCCStatusDigiCollection::DigiRangeIterator j=DCCstatusdigi->begin(); j!=DCCstatusdigi->end(); j++){
        std::vector<CSCDCCStatusDigi>::const_iterator digiItr = (*j).second.first;
        std::vector<CSCDCCStatusDigi>::const_iterator last = (*j).second.second;
        for( ; digiItr != last; ++digiItr) {
        digiItr->print();
        }
     }
   }
   
#ifdef THIS_IS_AN_EVENT_EXAMPLE
   Handle<ExampleData> pIn;
   iEvent.getByLabel("example",pIn);
#endif
   
#ifdef THIS_IS_AN_EVENTSETUP_EXAMPLE
   ESHandle<SetupData> pSetup;
   iSetup.get<SetupRecord>().get(pSetup);
#endif
}


// ------------ method called once each job just after ending the event loop  ------------
void 
CSCViewDigi::endJob() {
}

//define this as a plug-in
//DEFINE_FWK_MODULE(CSCViewDigi);

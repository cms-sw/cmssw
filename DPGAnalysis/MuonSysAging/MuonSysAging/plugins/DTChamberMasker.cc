// -*- C++ -*-
//
// Package:    L1Trigger/DTChamberMasker
// Class:      DTChamberMasker
// 
/**\class DTChamberMasker DTChamberMasker.cc L1Trigger/DTChamberMasker/plugins/DTChamberMasker.cc

 Description: Class to mask DT digis or trigger segments on a chamber by chamber basis

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Carlo Battilana
//         Created:  Sun, 11 Jan 2015 15:12:51 GMT
//
//


// system include files
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <regex>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"

#include "DataFormats/DTDigi/interface/DTLocalTriggerCollection.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "CondFormats/MuonSystemAging/interface/MuonSystemAging.h"
#include "CondFormats/DataRecord/interface/MuonSystemAgingRcd.h"


//
// class declaration
//

class DTChamberMasker : public edm::EDProducer {
public:
  explicit DTChamberMasker(const edm::ParameterSet&);
  ~DTChamberMasker();

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  virtual void beginJob() override;
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
      
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;

  void createMaskedChamberCollection(edm::ESHandle<DTGeometry> &); 

  // ----------member data ---------------------------

  edm::InputTag digiTag_, trPhTag_, trThTag_;

  edm::EDGetTokenT <DTDigiCollection> m_digiTag;
  edm::EDGetTokenT < L1MuDTChambPhContainer > m_trPhTag;
  edm::EDGetTokenT <L1MuDTChambThContainer> m_trThTag;

  bool m_doTriggerFromDDU;
  
  std::vector<std::regex> m_maskedChRegEx;
  std::vector<DTChamberId> m_enabledCh;  
  
};

//
// constants, enums and typedefs
//


//
// static data member definitions
//
//vector<L1MuRegionalCand>
//
// constructors and destructor
//
DTChamberMasker::DTChamberMasker(const edm::ParameterSet& iConfig) :
  digiTag_( iConfig.getParameter<edm::InputTag>("digiTag") ),
  trPhTag_( iConfig.getParameter<edm::InputTag>("triggerPrimPhTag")),
  trThTag_(iConfig.getParameter<edm::InputTag>("triggerPrimThTag") ),
  m_doTriggerFromDDU( iConfig.getParameter<bool>("doTriggerFromDDU") ) 
{
  m_digiTag = consumes<DTDigiCollection>( digiTag_),
  m_trPhTag = consumes<L1MuDTChambPhContainer>( trPhTag_ ),
  m_trThTag = consumes<L1MuDTChambThContainer> (trThTag_),



  produces<DTDigiCollection>();
  if (m_doTriggerFromDDU)
    produces<DTLocalTriggerCollection>();
  
  produces<L1MuDTChambPhContainer>();
  produces<L1MuDTChambThContainer>();

/*  for ( auto regStr : iConfig.getParameter<std::vector<std::string>>("maskedChRegEx") )
    m_maskedChRegEx.push_back(std::regex(regStr));
*/
}


DTChamberMasker::~DTChamberMasker()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
DTChamberMasker::produce(edm::Event& event, const edm::EventSetup& iSetup)
{
  
 
  std::unique_ptr<DTDigiCollection> filteredDigis(new DTDigiCollection());
  std::unique_ptr<DTLocalTriggerCollection> filteredTrigsDDU(new DTLocalTriggerCollection());

  std::unique_ptr<L1MuDTChambPhContainer> filteredPhTrigPrimsCont(new L1MuDTChambPhContainer());
  std::unique_ptr<L1MuDTChambThContainer> filteredThTrigPrimsCont(new L1MuDTChambThContainer());

  std::vector<L1MuDTChambPhDigi> filteredPhTrigPrims;
  std::vector<L1MuDTChambThDigi> filteredThTrigPrims;


  if( !digiTag_.label().empty())
    {
      edm::Handle<DTDigiCollection> dtDigis;
      event.getByToken(m_digiTag, dtDigis);
  
      DTDigiCollection::DigiRangeIterator dtLayerIdIt  = dtDigis->begin();
      DTDigiCollection::DigiRangeIterator dtLayerIdEnd = dtDigis->end();
      
      for (; dtLayerIdIt != dtLayerIdEnd; ++dtLayerIdIt)
	{
	  
	  DTChamberId chId = ((*dtLayerIdIt).first).chamberId();
	  if (std::find(m_enabledCh.begin(),m_enabledCh.end(),chId) != m_enabledCh.end())
	    filteredDigis->put((*dtLayerIdIt).second,(*dtLayerIdIt).first);
	  
	}
      // CB write the code to apply masks to the DDU collection here
      //      if (m_doTriggerFromDDU)
      //	{
      //        }

    }

  if (!trPhTag_.label().empty())
   {
      edm::Handle<L1MuDTChambPhContainer> phTrigPrimDigis;
      event.getByToken(m_trPhTag, phTrigPrimDigis);
      
      for (auto const trPhDigi : (*phTrigPrimDigis->getContainer()) )
	{
	  
	  DTChamberId chId = DTChamberId(trPhDigi.whNum(), trPhDigi.stNum(), (trPhDigi.scNum()+1));
	  if (std::find(m_enabledCh.begin(),m_enabledCh.end(),chId) != m_enabledCh.end())
	    filteredPhTrigPrims.push_back(trPhDigi);	  
	  
	}

   }
      
  if (!trThTag_.label().empty())
     {
      edm::Handle<L1MuDTChambThContainer> thTrigPrimDigis;
      event.getByToken(m_trThTag, thTrigPrimDigis);

      for (auto const trThDigi : (*thTrigPrimDigis->getContainer()) )
	{
	  
	  DTChamberId chId = DTChamberId(trThDigi.whNum(), trThDigi.stNum(), (trThDigi.scNum()+1));
	  if (std::find(m_enabledCh.begin(),m_enabledCh.end(),chId) != m_enabledCh.end())
	    filteredThTrigPrims.push_back(trThDigi);	  
	  
	}

    }

  filteredPhTrigPrimsCont->setContainer(filteredPhTrigPrims);
  filteredThTrigPrimsCont->setContainer(filteredThTrigPrims);
  
  event.put(std::move(filteredDigis));
  if (m_doTriggerFromDDU)
    event.put(std::move(filteredTrigsDDU));
  event.put(std::move(filteredPhTrigPrimsCont));
  event.put(std::move(filteredThTrigPrimsCont));

}

// ------------ method called once each job just before starting event loop  ------------
void 
DTChamberMasker::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
DTChamberMasker::endJob()
{

}

// ------------ method called when starting to processes a run  ------------
void
DTChamberMasker::beginRun(edm::Run const& run, edm::EventSetup const& iSetup)
{

  edm::ESHandle<DTGeometry> dtGeom;
  iSetup.get<MuonGeometryRecord>().get(dtGeom);

  if (m_enabledCh.size() == 0)
    {
       edm::ESHandle<MuonSystemAging> mcData;
       iSetup.get<MuonSystemAgingRcd>().get(mcData);
       const MuonSystemAging* myMC=mcData.product();
       std::vector<std::string> mcV = myMC->m_DTchambers;
       for(unsigned int i = 0; i < mcV.size();++i)m_maskedChRegEx.push_back(std::regex(mcV.at(i)));

      createMaskedChamberCollection(dtGeom);



    }

}

  

void
DTChamberMasker::createMaskedChamberCollection(edm::ESHandle<DTGeometry> & dtGeom)
{

  const std::vector<const DTChamber*> chambers = dtGeom->chambers();

  for ( const DTChamber *ch : chambers)
   {

     DTChamberId chId = ch->id();

     std::string chTag = "WH" + std::to_string(chId.wheel())
                       + "_ST" + std::to_string(chId.station())
                       + "_SEC" + std::to_string(chId.sector());

     bool isMaskedCh = false;

     for ( const std::regex chRegex : m_maskedChRegEx)
       if ( std::regex_match(chTag,chRegex) )
	 isMaskedCh = true;
     if (!isMaskedCh)
       	 m_enabledCh.push_back(chId);
         
   }
  
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
DTChamberMasker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DTChamberMasker);

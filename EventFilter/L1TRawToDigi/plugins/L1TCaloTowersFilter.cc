// -*- C++ -*-
//
// Package:    L1TValidationEventFilter
// Class:      L1TValidationEventFilter
// 
/**\class L1TValidationEventFilter L1TValidationEventFilter.cc EventFilter/L1TRawToDigi/src/L1TValidationEventFilter.cc

Description: <one line class summary>
Implementation:
nn<Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <iostream>


#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"

#include "DataFormats/L1TCalorimeter/interface/CaloTower.h"

//
// class declaration
//

namespace l1t {

class L1TCaloTowersFilter : public edm::EDFilter {
public:
  explicit L1TCaloTowersFilter(const edm::ParameterSet&);
  ~L1TCaloTowersFilter() override;
  
private:
  void beginJob() override ;
  bool filter(edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
  
  // ----------member data ---------------------------
  edm::EDGetTokenT<FEDRawDataCollection> fedData_;
  edm::EDGetToken m_towerToken;

  int period_;       // validation event period

};



//
// constructors and destructor
//
L1TCaloTowersFilter::L1TCaloTowersFilter(const edm::ParameterSet& iConfig) :
  period_( iConfig.getUntrackedParameter<int>("period", 107) )
{
  //now do what ever initialization is needed

  // fedData_ = consumes<FEDRawDataCollection>(iConfig.getParameter<edm::InputTag>("inputTag"));
  edm::InputTag towerTag = iConfig.getParameter<edm::InputTag>("towerToken");
  m_towerToken = consumes<l1t::CaloTowerBxCollection>(towerTag);
}


L1TCaloTowersFilter::~L1TCaloTowersFilter()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
L1TCaloTowersFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;

  Handle< BXVector<l1t::CaloTower> > towers;
  iEvent.getByToken(m_towerToken,towers);
  
  // edm::Handle<FEDRawDataCollection> feds;
  // iEvent.getByToken(fedData_, feds);

  if (towers->size() == 0) {
    LogDebug("L1TCaloTowersFilter") << "Event does not contain towers." << std::endl;
    return false;
  }

  LogDebug("L1TCaloTowersFilter") << "Event does contains towers." << std::endl;
  return true;

  /*
  if (!feds.isValid()) {
    LogError("L1T") << "Cannot unpack: no FEDRawDataCollection found";
    return false;
  }

  const FEDRawData& l1tRcd = feds->FEDData(1024);

  const unsigned char *data = l1tRcd.data();
  FEDHeader header(data);

  bool fatEvent = (header.lvl1ID() % period_ == 0 );

  return fatEvent;
  */
}

// ------------ method called once each job just before starting event loop  ------------
void 
L1TCaloTowersFilter::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1TCaloTowersFilter::endJob() {

}

}

using namespace l1t;
//define this as a plug-in
DEFINE_FWK_MODULE(L1TCaloTowersFilter);


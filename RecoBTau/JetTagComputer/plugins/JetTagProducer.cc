// -*- C++ -*-
//
// Package:    JetTagProducer
// Class:      JetTagProducer
// 
/**\class JetTagProducer JetTagProducer.cc RecoBTag/JetTagProducer/src/JetTagProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: JetTagProducer.cc,v 1.8 2007/09/21 08:22:01 saout Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"
#include "JetTagProducer.h"

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//
JetTagProducer::JetTagProducer(const edm::ParameterSet& iConfig) : 
  m_config(iConfig) {

  m_tagInfo = iConfig.getParameter<edm::InputTag>("tagInfo");
  m_jetTagComputer = iConfig.getParameter<string>("jetTagComputer");

  produces<reco::JetTagCollection>();

}

JetTagProducer::~JetTagProducer()
{
}

//
// member functions
//
// ------------ method called to produce the data  ------------
void
JetTagProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  Handle< View<BaseTagInfo> > tagInfoHandle;
  iEvent.getByLabel(m_tagInfo,tagInfoHandle);
       
  edm::ESHandle<JetTagComputer> computer;
  iSetup.get<JetTagComputerRecord>().get( m_jetTagComputer, computer );
  m_computer = computer.product() ;
  m_computer->setEventSetup(iSetup);

  std::auto_ptr<reco::JetTagCollection> jetTagCollection(new reco::JetTagCollection());
   
  View<BaseTagInfo>::const_iterator it = tagInfoHandle->begin();
  for (int cc=0; it != tagInfoHandle->end(); it++, cc++)
  {
    JetTag jt(m_computer->discriminator(*it));
    jt.setTagInfo(tagInfoHandle->refAt(cc));
    jetTagCollection->push_back(jt);    
  }
  
  iEvent.put(jetTagCollection);
}

// define it as plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagProducer);

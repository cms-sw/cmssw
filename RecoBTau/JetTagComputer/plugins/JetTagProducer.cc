// -*- C++ -*-
//
// Package:    JetTagProducer
// Class:      JetTagProducer
// 
/**\class JetTagProducer JetTagProducer.cc RecoBTag/JetTagProducer/src/JetTagProducer.cc

 Description: Uses a JetTagComputer to produce JetTags from TagInfos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Thu Apr  6 09:56:23 CEST 2006
// $Id: JetTagProducer.cc,v 1.11 2010/02/11 00:13:36 wmtan Exp $
//
//


// system include files
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/BaseTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputerRecord.h"

#include "JetTagProducer.h"

using namespace std;
using namespace reco;
using namespace edm;

//
// constructors and destructor
//
JetTagProducer::JetTagProducer(const ParameterSet& iConfig) :
  m_computer(0),
  m_jetTagComputer(iConfig.getParameter<string>("jetTagComputer")),
  m_tagInfos(iConfig.getParameter< vector<InputTag> >("tagInfos"))
{
  produces<JetTagCollection>();
}

JetTagProducer::~JetTagProducer()
{
}

//
// member functions
//
// internal method called on first event to locate and setup JetTagComputer
void JetTagProducer::setup(const edm::EventSetup& iSetup)
{
  edm::ESHandle<JetTagComputer> computer;
  iSetup.get<JetTagComputerRecord>().get( m_jetTagComputer, computer );
  m_computer = computer.product();
  m_computer->setEventSetup(iSetup);

  // finalize the JetTagProducer <-> JetTagComputer glue setup
  vector<string> inputLabels(m_computer->getInputLabels());

  // backward compatible case, use default tagInfo
  if (inputLabels.empty())
    inputLabels.push_back("tagInfo");

  if (m_tagInfos.size() != inputLabels.size()) {
    std::string message("VInputTag size mismatch - the following taginfo "
                        "labels are needed:\n");
    for(vector<string>::const_iterator iter = inputLabels.begin();
        iter != inputLabels.end(); ++iter)
      message += "\"" + *iter + "\"\n";
    throw edm::Exception(errors::Configuration) << message;
  }
}

// map helper - for some reason RefToBase lacks operator < (...)
namespace {
  struct JetRefCompare :
       public binary_function<RefToBase<Jet>, RefToBase<Jet>, bool> {
    inline bool operator () (const RefToBase<Jet> &j1,
                             const RefToBase<Jet> &j2) const
    { return j1.id() < j2.id() || (j1.id() == j2.id() && j1.key() < j2.key()); }
  };
}

// ------------ method called to produce the data  ------------
void
JetTagProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  if (m_computer)
    m_computer->setEventSetup(iSetup);
  else
    setup(iSetup);

  // now comes the tricky part:
  // we need to collect all requested TagInfos belonging to the same jet

  typedef vector<const BaseTagInfo*> TagInfoPtrs;
  typedef RefToBase<Jet> JetRef;
  typedef map<JetRef, TagInfoPtrs, JetRefCompare> JetToTagInfoMap;

  JetToTagInfoMap jetToTagInfos;

  // retrieve all requested TagInfos
  vector< Handle< View<BaseTagInfo> > > tagInfoHandles(m_tagInfos.size());
  unsigned int nTagInfos = m_tagInfos.size();
  for(unsigned int i = 0; i < nTagInfos; i++) {
    Handle< View<BaseTagInfo> > &tagInfoHandle = tagInfoHandles[i];
    iEvent.getByLabel(m_tagInfos[i], tagInfoHandle);

    for(View<BaseTagInfo>::const_iterator iter = tagInfoHandle->begin();
        iter != tagInfoHandle->end(); iter++) {
      TagInfoPtrs &tagInfos = jetToTagInfos[iter->jet()];
      if (tagInfos.empty())
        tagInfos.resize(nTagInfos);

      tagInfos[i] = &*iter;
    }
  }

  // take first tagInfo
  Handle< View<BaseTagInfo> > &tagInfoHandle = tagInfoHandles[0];
  auto_ptr<JetTagCollection> jetTagCollection;
  if (tagInfoHandle.product()->size() > 0) {
    RefToBase<Jet> jj = tagInfoHandle->begin()->jet();
    jetTagCollection.reset(new JetTagCollection(RefToBaseProd<Jet>(jj)));
  } else
    jetTagCollection.reset(new JetTagCollection());

  // now loop over the map and compute all JetTags
  for(JetToTagInfoMap::const_iterator iter = jetToTagInfos.begin();
      iter != jetToTagInfos.end(); iter++) {
    const TagInfoPtrs &tagInfoPtrs = iter->second;

    JetTagComputer::TagInfoHelper helper(tagInfoPtrs);
    float discriminator = (*m_computer)(helper);

    (*jetTagCollection)[iter->first] = discriminator;
  }

  iEvent.put(jetTagCollection);
}

// define it as plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagProducer);

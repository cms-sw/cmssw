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
// $Id: JetTagProducer.cc,v 1.4 2007/10/07 02:50:04 saout Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

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
  m_jetTagComputer(iConfig.getParameter<string>("jetTagComputer"))
{
  vector<string> inputTags = iConfig.getParameterNamesForType<InputTag>();

  for(vector<string>::const_iterator iter = inputTags.begin();
      iter != inputTags.end(); iter++) {
    InputTag inputTag = iConfig.getParameter<InputTag>(*iter);
    m_tagInfoLabels[*iter] = inputTag;
  }

  produces<JetTagCollection>();
}

JetTagProducer::~JetTagProducer()
{
}

//
// member functions
//
// ------------ method to set up the TagInfo mapping ------------

void 
JetTagProducer::setup(Event& iEvent)
{
  vector<string> inputLabels(m_computer->m_inputLabels);

  // backward compatible case, use default tagInfo
  if (inputLabels.empty())
    inputLabels.push_back("tagInfo");

  // collect all TagInfos from the ParameterSet that the JetTagComputer wants
  for(vector<string>::const_iterator iter = inputLabels.begin();
      iter != inputLabels.end(); iter++) {
    map<string, InputTag>::const_iterator pos = m_tagInfoLabels.find(*iter);
    if (pos == m_tagInfoLabels.end())
      throw cms::Exception("InputTagMissing") << "JetTagProducer is missing "
      			"a TagInfo InputTag \"" << *iter << "\"" << std::endl;

    m_tagInfos.push_back(pos->second);
  }

  m_computer->m_setupDone = true;
}

// map helper
namespace {
  struct JetRefCompare :
	public std::binary_function<RefToBase<Jet>, RefToBase<Jet>, bool> {
    inline bool operator () (const RefToBase<Jet> &j1,
                             const RefToBase<Jet> &j2) const
    { return j1.key() < j2.key(); }
  };
}

// ------------ method called to produce the data  ------------
void
JetTagProducer::produce(Event& iEvent, const EventSetup& iSetup)
{
  ESHandle<JetTagComputer> computer;
  iSetup.get<JetTagComputerRecord>().get( m_jetTagComputer, computer );
  m_computer = computer.product();
  m_computer->setEventSetup(iSetup);

  // finalize the JetTagProducer <-> JetTagComputer glue setup
  if (!m_computer->m_setupDone)
    setup(iEvent);

  // now comes the tricky part:
  // we need to collect all requested TagInfos belonging to the same jet

  typedef std::vector<const BaseTagInfo*> TagInfoPtrs;
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
  JetTagCollection * jtc;
  if (tagInfoHandle.product()->size() > 0) {
    RefToBase<Jet> jj = tagInfoHandle->begin()->jet();
    jtc = new JetTagCollection(RefToBaseProd<Jet>(jj));
  } else
    jtc = new JetTagCollection();
  std::auto_ptr<JetTagCollection> jetTagCollection(jtc);

  // now loop over the map and compute all JetTags
  for(JetToTagInfoMap::const_iterator iter = jetToTagInfos.begin();
      iter != jetToTagInfos.end(); iter++) {
    const TagInfoPtrs &tagInfoPtrs = iter->second;

    JetTagComputer::TagInfoHelper helper(tagInfoPtrs);
    float discriminator = (*m_computer)(helper);

//    JetTag jetTag(discriminator);
//    jetTag.setTagInfo(tagInfoRef);
//    jetTagCollection->push_back(jetTag);

    (*jetTagCollection)[iter->first] = discriminator;
  }

  iEvent.put(jetTagCollection);
}

// define it as plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagProducer);

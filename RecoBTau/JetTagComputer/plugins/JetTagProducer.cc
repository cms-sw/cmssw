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
#include "FWCore/Framework/interface/makeRefToBaseProdFrom.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

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
  std::vector<edm::InputTag> m_tagInfos = iConfig.getParameter< vector<InputTag> >("tagInfos");
  nTagInfos = m_tagInfos.size();
  for(unsigned int i = 0; i < nTagInfos; i++) {
    token_tagInfos.push_back( consumes<View<BaseTagInfo> >(m_tagInfos[i]) );
  }

  produces<JetTagCollection>();
}

JetTagProducer::~JetTagProducer()
{
}

//
// member functions
//

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
  edm::ESHandle<JetTagComputer> computer;
  iSetup.get<JetTagComputerRecord>().get( m_jetTagComputer, computer );

  if (recordWatcher_.check(iSetup) ) {
    unsigned int nLabels = computer->getInputLabels().size();
    if (nLabels == 0) ++nLabels;
    if (nTagInfos != nLabels) {

      vector<string> inputLabels(computer->getInputLabels());
      // backward compatible case, use default tagInfo
      if (inputLabels.empty())
        inputLabels.push_back("tagInfo");
      std::string message("VInputTag size mismatch - the following taginfo "
                          "labels are needed:\n");
      for(vector<string>::const_iterator iter = inputLabels.begin();
          iter != inputLabels.end(); ++iter)
        message += "\"" + *iter + "\"\n";
      throw edm::Exception(errors::Configuration) << message;
    }
  }

  // now comes the tricky part:
  // we need to collect all requested TagInfos belonging to the same jet

  typedef vector<const BaseTagInfo*> TagInfoPtrs;
  typedef RefToBase<Jet> JetRef;
  typedef map<JetRef, TagInfoPtrs, JetRefCompare> JetToTagInfoMap;

  JetToTagInfoMap jetToTagInfos;

  // retrieve all requested TagInfos
  vector< Handle< View<BaseTagInfo> > > tagInfoHandles(nTagInfos);
  for(unsigned int i = 0; i < nTagInfos; i++) {
    Handle< View<BaseTagInfo> > &tagInfoHandle = tagInfoHandles[i];
    iEvent.getByToken(token_tagInfos[i], tagInfoHandle);

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
    jetTagCollection.reset(new JetTagCollection(edm::makeRefToBaseProdFrom(jj, iEvent)));
  } else
    jetTagCollection.reset(new JetTagCollection());

  // now loop over the map and compute all JetTags
  for(JetToTagInfoMap::const_iterator iter = jetToTagInfos.begin();
      iter != jetToTagInfos.end(); iter++) {
    const TagInfoPtrs &tagInfoPtrs = iter->second;

    JetTagComputer::TagInfoHelper helper(tagInfoPtrs);
    float discriminator = (*computer)(helper);

    (*jetTagCollection)[iter->first] = discriminator;
  }

  iEvent.put(jetTagCollection);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void
JetTagProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<std::string>("jetTagComputer","combinedMVAComputer");
  {
    std::vector<edm::InputTag> tagInfos;
    tagInfos.push_back(edm::InputTag("impactParameterTagInfos"));
    tagInfos.push_back(edm::InputTag("inclusiveSecondaryVertexFinderTagInfos"));
    tagInfos.push_back(edm::InputTag("softPFMuonsTagInfos"));
    tagInfos.push_back(edm::InputTag("softPFElectronsTagInfos"));
    desc.add<std::vector<edm::InputTag> >("tagInfos",tagInfos);
  }
  descriptions.addDefault(desc);
}

// define it as plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetTagProducer);

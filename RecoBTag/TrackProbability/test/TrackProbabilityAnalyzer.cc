// -*- C++ -*-
//
// Package:    TrackProbabilityAnalyzer
// Class:      TrackProbabilityAnalyzer
// 
/**\class TrackProbabilityAnalyzer TrackProbabilityAnalyzer.cc RecoBTag/TrackProbabilityAnalyzer/src/TrackProbabilityAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// $Id: TrackProbabilityAnalyzer.cc,v 1.2 2006/12/07 09:59:35 arizzi Exp $
//
//


// system include files
#include <memory>
#include <string>
#include <iostream>
using namespace std;

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/TrackProbabilityTagInfo.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/Math/interface/Vector3D.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace reco;

//
// class decleration
//

class TrackProbabilityAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrackProbabilityAnalyzer(const edm::ParameterSet&);
      ~TrackProbabilityAnalyzer() {}

      virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:
     string m_assoc;
     string m_jets;
};

//
// constructors and destructor
//
TrackProbabilityAnalyzer::TrackProbabilityAnalyzer(const edm::ParameterSet& iConfig)
{
}

void
TrackProbabilityAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;
  
  Handle<JetTagCollection> jetsHandle;
  Handle<TrackProbabilityTagInfoCollection> jetsInfoHandle;
  iEvent.getByLabel("trackProbabilityJetTags", jetsHandle);
  iEvent.getByLabel("trackProbabilityJetTags", jetsInfoHandle);
  const JetTagCollection & jets = *(jetsHandle.product());
  const TrackProbabilityTagInfoCollection & info = *(jetsInfoHandle.product());

  for (JetTagCollection::size_type i = 0; i < jets.size(); ++i) {
    cout << jets[i].discriminator() <<endl;
    cout  << "Number of associated tracks " << jets[i].tracks().size() << endl;
    const edm::RefVector<TrackCollection> & tracks=  jets[i].tracks();
    for (edm::RefVector<TrackCollection>::iterator j = tracks.begin(); j != tracks.end(); ++j) {
      cout << (*j)->pt() << endl;
    }
  }    
 
  for(TrackProbabilityTagInfoCollection::size_type i = 0 ; i < info.size() ; ++i  )
  {
       cout << i << endl;
  //    cout << &(info[i]) << endl;
     cout << info[i].discriminator(0,0.005) << " " << info[i].discriminator(1,0.005) << endl;
    for(int j = 0 ; j < info[i].selectedTracks(0); j++)
     {
       cout << info[i].track(j,0).pt() << " " << info[i].probability(j,0) << endl; 
     }
    
 }

}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TrackProbabilityAnalyzer);

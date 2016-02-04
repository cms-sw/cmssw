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
// $Id: TrackProbabilityAnalyzer.cc,v 1.8 2007/10/06 19:31:01 arizzi Exp $
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
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

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

#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
//#include "CondFormats/DataRecord/interface/BTagTrackProbabilityRcd.h"

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
  static int above =0;
  static int tot =0;
  using namespace edm;
  using namespace reco;
//  using namespace eventsetup;
/*  ESHandle<TrackProbabilityCalibration> calib;
  iSetup.get<BTagTrackProbabilityRcd>().get(calib);
 
  const TrackProbabilityCalibration *  ca= calib.product();
  cout << "Bin for 0.35 " << ca->data[5].histogram.findBin(0.35) << endl;
 
  return; 
  */Handle<JetTagCollection> jetsHandle;
  Handle<TrackProbabilityTagInfoCollection> jetsInfoHandle;
  iEvent.getByLabel("trackProbabilityJetTags", jetsHandle);
  iEvent.getByLabel("trackProbabilityJetTags", jetsInfoHandle);
  const JetTagCollection & jets = *(jetsHandle.product());
  const TrackProbabilityTagInfoCollection & info = *(jetsInfoHandle.product());

  for (JetTagCollection::size_type i = 0; i < jets.size(); ++i) {
   // cout << jets[i].first <<endl;
  //  cout  << "Number of associated tracks " << jets[i].tracks().size() << endl;
  //  const edm::RefVector<TrackCollection> & tracks=  jets[i].tracks();
    //for (edm::RefVector<TrackCollection>::iterator j = tracks.begin(); j != tracks.end(); ++j) {
    //  cout << (*j)->pt() << endl;
  //  }
  }    
 
  for(TrackProbabilityTagInfoCollection::size_type i = 0 ; i < info.size() ; ++i  )
  {
       cout << i << endl;
  //    cout << &(info[i]) << endl;
    cout << info[i].discriminator(0,0.005) << " " << info[i].discriminator(1,0.005) << endl;
    if(info[i].discriminator(0,0.005) > 90) above++;
    tot++;
    cout << above << " " << tot << endl;
    for(int j = 0 ; j < info[i].selectedTracks(0); j++)
     {
       cout << info[i].track(j,0).pt() << " " << info[i].probability(j,0) << endl; 
     }
    
 }

}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackProbabilityAnalyzer);

// -*- C++ -*-
//
// Package:    TauImpactParameterTest
// Class:      TauImpactParameterTest
//
/**\class TauImpactParameterTest TauImpactParameterTest.cc RecoTauTag/ImpactParameter/test/TauImpactParameterTest.cc

 Description: EDAnalyzer to show how to get the tau impact parameter
 Implementation:

*/
//
// Original Author:  Sami Lehti
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/TauImpactParameterInfo.h"

#include <iostream>
#include <string>

using namespace std;
using namespace edm;
using namespace reco;

class TauImpactParameterTest : public edm::EDAnalyzer {

  public:
        TauImpactParameterTest(const edm::ParameterSet&);
        ~TauImpactParameterTest();

        virtual void analyze(const edm::Event&, const edm::EventSetup&);
        virtual void beginJob();
        virtual void endJob();

  private:
        string jetTagSrc;
};


TauImpactParameterTest::TauImpactParameterTest(const edm::ParameterSet& iConfig){
	jetTagSrc = iConfig.getParameter<string>("JetTagProd");
}

TauImpactParameterTest::~TauImpactParameterTest(){

}

void TauImpactParameterTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){

	Handle<TauImpactParameterInfoCollection> tauHandle;
	iEvent.getByLabel(jetTagSrc,tauHandle);

	const TauImpactParameterInfoCollection & tauIpInfo = *(tauHandle.product());
	cout << "Found " << tauIpInfo.size() << " Tau candidates" << endl;

	TauImpactParameterInfoCollection::const_iterator iJet;
	int i = 0;
	for (iJet = tauIpInfo.begin(); iJet != tauIpInfo.end(); iJet++) {

	    cout << "  Candidate " << i << endl;

	    const TrackRefVector& tracks = iJet->getIsolatedTauTag()->selectedTracks();

            RefVector<TrackCollection>::const_iterator iTrack;
	    for (iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++){

                cout << "    track pt, eta " << (*iTrack)->pt() << " " 
                                             << (*iTrack)->eta() << endl;

		const reco::TauImpactParameterTrackData* trackData = iJet->getTrackData(*iTrack);

		if(! trackData == 0){
		  Measurement1D tip = trackData->transverseIp;

		  cout << "          ip,sip,err " << tip.value()
		       << " " 	 		  << tip.significance()
                       << " "                     << tip.error() << endl; 
		}else{
		  cout << "    track data = 0! " << endl;
		}
	    }

	    cout <<"    discriminator = "<< iJet->discriminator() <<endl;
	    i++;
	}
}

void TauImpactParameterTest::beginJob(){
}

void TauImpactParameterTest::endJob(){
}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(TauImpactParameterTest);

/* \class QCDSingleJetFilter
 *
 * QCDSingleJetFilter for CSA07 Excercise
 *
 * author:  Andreas Oehler (andreas.oehler@cern.ch)
 * see header
 */

//MyHeadeR:
#include <QCDAnalysis/Skimming/interface/QCDSingleJetFilter.h>

// User include files

#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/InputTag.h>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/JetReco/interface/GenJet.h"

//rand:
//#include <FWCore/ServiceRegistry/interface/Service.h>
//#include <FWCore/Utilities/interface/RandomNumberGenerator.h>
//#include <CLHEP/Random/RandFlat.h>

// C++
#include <algorithm>
#include <iostream>
#include <vector>
#include <math.h>

using namespace std;
using namespace edm;
//using namespace reco;

//detruktor
QCDSingleJetFilter::~QCDSingleJetFilter(){
  //delete theFlatDistrib;
}

// Constructor
//QCDSingleJetFilter::QCDSingleJetFilter(const edm::ParameterSet& pset):theFlatDistrib(0),theTriggerJetCollectionA(pset.getParameter<edm::InputTag>("TriggerJetCollectionA")),theTrigCollB(pset.getParameter<edm::InputTag>("TriggerJetCollectionB")){
QCDSingleJetFilter::QCDSingleJetFilter(const edm::ParameterSet& pset):
  theTriggerJetCollectionAToken(consumes<reco::CaloJetCollection>(pset.getParameter<edm::InputTag>("TriggerJetCollectionA"))),
  theTrigCollBToken(consumes<reco::CaloJetCollection>(pset.getParameter<edm::InputTag>("TriggerJetCollectionB"))){

  // Local Debug flag
  //debug              = pset.getParameter<bool>("DebugHiggsToZZ4LeptonsSkim");

  //getConfigParameter:
  theMinPt = pset.getParameter<double>("MinPt");
  //prescale taken out for convenience
  //thePreScale = pset.getParameter<double>("PreScale");
  //thePreScale=fabs(thePreScale);
  //if (thePreScale<1) thePreScale=0;

  // Eventually, HLT objects:

  //get Random-Service running:
  //edm::Service<edm::RandomNumberGenerator> rng;
  //if (!rng.isAvailable()) {
  //  throw cms::Exception("QCDSingleJetFilter")<<"QCDSingleJetFilter requires RandomNumberGeneratorService\n"
  //    "--borked setup\n";
  //}
  //CLHEP::HepRandomEngine& engine = rng->getEngine();
  //theFlatDistrib = new CLHEP::RandFlat(engine,0.0,1.0);
}




// Filter event
bool QCDSingleJetFilter::filter(edm::Event& event, const edm::EventSetup& setup) {
    bool keepEvent = false;
    using namespace edm;
    using namespace std;

    //now get right Jet-Collection:
    edm::Handle<reco::CaloJetCollection>  theTriggerCollectionJetsA;
    edm::Handle<reco::CaloJetCollection>  theTrigCollJetsB;

    event.getByToken(theTriggerJetCollectionAToken,theTriggerCollectionJetsA);
    event.getByToken(theTrigCollBToken,theTrigCollJetsB);

    for (reco::CaloJetCollection::const_iterator iter=theTriggerCollectionJetsA->begin();iter!=theTriggerCollectionJetsA->end();++iter){
      if ((*iter).pt()>=theMinPt) {
	keepEvent=true;
	break;
      }
    }

    for (reco::CaloJetCollection::const_iterator iter=theTrigCollJetsB->begin();iter!=theTrigCollJetsB->end();++iter){
      if ((*iter).pt()>=theMinPt) {
	keepEvent=true;
	break;
      }
    }


    //double randval = theFlatDistrib->fire();
    //if (thePreScale<1) keepEvent=false;
    //else if ((randval>(1.0/thePreScale))&&keepEvent) keepEvent=false;
//   cout<<"KeepEvent?: "<<keepEvent<<endl;

    return keepEvent;

}


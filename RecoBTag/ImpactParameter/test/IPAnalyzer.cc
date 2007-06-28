// -*- C++ -*-
//
// Package:    IPAnalyzer
// Class:      IPAnalyzer
// 
/**\class IPAnalyzer IPAnalyzer.cc RecoBTag/IPAnalyzer/src/IPAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Wed Apr 12 11:12:49 CEST 2006
// $Id: IPAnalyzer.cc,v 1.2 2007/05/10 22:05:04 arizzi Exp $
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
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"

#include "DataFormats/Math/interface/Vector3D.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

using namespace reco;

//
// class decleration
//

class IPAnalyzer : public edm::EDAnalyzer {
   public:
      explicit IPAnalyzer(const edm::ParameterSet&);
      ~IPAnalyzer() {}

      virtual void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup);

   private:
     edm::InputTag m_ipassoc;
     edm::InputTag m_assoc;
     edm::InputTag m_jets;
};

//
// constructors and destructor
//
IPAnalyzer::IPAnalyzer(const edm::ParameterSet& iConfig)
{
  m_jets  = iConfig.getParameter<edm::InputTag>("jets");
  m_assoc = iConfig.getParameter<edm::InputTag>("association");
  m_ipassoc = iConfig.getParameter<edm::InputTag>("ipassociation");
}

void
IPAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  Handle<TrackIPTagInfoCollection> ipHandle;
  iEvent.getByLabel(m_ipassoc, ipHandle);
  const TrackIPTagInfoCollection & ip = *(ipHandle.product());
  cout << "Found " << ip.size() << " TagInfo" << endl;


  cout << boolalpha;
  cout << fixed;



   TrackIPTagInfoCollection::const_iterator it = ip.begin();
   for(; it != ip.end(); it++)
     {
      cout << "Jet pt: " << it->jet()->pt() << endl;
      cout << "Tot tracks: " << it->tracks().size() << endl;    
      TrackRefVector selTracks=it->selectedTracks();
      int n=selTracks.size();
      cout << "Sel tracks: " << n << endl; 
      cout << " Pt  \t d len \t jet dist \t p3d \t p2d\t ip3d \t ip2d " << endl; 
      for(int j=0;j< n;j++)
      {
        cout << selTracks[j]->pt() << "\t";
        cout << it->decayLengths()[j].value() << "\t";
        cout << it->jetDistances()[j].value() << "\t";
        cout << it->probabilities(0)[j]<< "\t";
        cout << it->probabilities(1)[j]<< "\t";
        cout << it->impactParameters(0)[j].significance() << "\t";
        cout << it->impactParameters(1)[j].significance() << endl;
      }

  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(IPAnalyzer);

/**\class JetHTJetPlusHOFilter JetHTJetPlusHOFilter.cc Test/BarrelJetFilter/src/BarrelJetFilter.cc

Skimming of JetHT data set for the study of HO absolute weight calculation
* 
* Skimming Efficiency : ~ 5 %

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]

*/
//
// Original Author: Gobinda Majumder & Suman Chatterjee 
//         Created:  Fri Dec 16 14:52:17 IST 2011
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;



class JetHTJetPlusHOFilter : public edm::EDFilter {
   public:
      explicit JetHTJetPlusHOFilter(const edm::ParameterSet&);
      ~JetHTJetPlusHOFilter();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&);

      // ----------member data ---------------------------
  int Nevt;
  int Njetp;
  int Npass;
  double jtptthr;
  double jtetath;
  double hothres;
  bool isOthHistFill;

  edm::EDGetTokenT<reco::PFJetCollection> tok_PFJets_;
  edm::EDGetTokenT<reco::PFClusterCollection> tok_hoht_;   

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
JetHTJetPlusHOFilter::JetHTJetPlusHOFilter(const edm::ParameterSet& iConfig) { 

  tok_PFJets_ = consumes<reco::PFJetCollection>( iConfig.getParameter<edm::InputTag>("PFJets"));
  tok_hoht_ = consumes<reco::PFClusterCollection>( iConfig.getParameter<edm::InputTag>("particleFlowClusterHO"));


   //now do what ever initialization is needed
  jtptthr = iConfig.getUntrackedParameter<double>("Ptcut", 200.0);
  jtetath = iConfig.getUntrackedParameter<double>("Etacut",1.5);
  hothres = iConfig.getUntrackedParameter<double>("HOcut", 8.0);

  Nevt = 0;
  Njetp=0;
  Npass=0;


}


JetHTJetPlusHOFilter::~JetHTJetPlusHOFilter()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//

// ------------ method called on each new Event  ------------
bool
JetHTJetPlusHOFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;  
  Nevt++;

  edm::Handle<reco::PFJetCollection> PFJets;
  iEvent.getByToken(tok_PFJets_,PFJets);   
  bool passed=false;
  vector <pair<double, double> > jetdirection;
  vector<double> jetspt; 
  if (PFJets.isValid()) {
    for (unsigned jet = 0; jet<PFJets->size(); jet++) {
      if(((*PFJets)[jet].pt()<jtptthr)||(abs((*PFJets)[jet].eta())>jtetath)) continue;
      
      std::pair<double, double> etaphi((*PFJets)[jet].eta(),(*PFJets)[jet].phi());
      jetdirection.push_back(etaphi);
      jetspt.push_back((*PFJets)[jet].pt());
      passed = true;
    }
  }
  
  if (!passed) return false;
  Njetp++;
  bool isJetDir=false;
	
  edm::Handle<PFClusterCollection> hoht;
  iEvent.getByToken(tok_hoht_,hoht);
  if (hoht.isValid()) {
    if ((*hoht).size()>0) {
      for (PFClusterCollection::const_iterator ij=(*hoht).begin(); ij!=(*hoht).end(); ij++){
				double hoenr = (*ij).energy();
				if (hoenr <hothres) continue;
				
				const math::XYZPoint&  cluster_pos = ij->position();
				
				double hoeta = cluster_pos.eta() ;
				double hophi = cluster_pos.phi() ;
				
				for (unsigned ijet = 0; ijet< jetdirection.size(); ijet++) {
					double delta = deltaR2(jetdirection[ijet].first, jetdirection[ijet].second, hoeta, hophi);
					if (delta <0.5) { 
						isJetDir=true;   break;
					}
				}
      }
    }
  }
	
  if (isJetDir) {Npass++;}
  
  return isJetDir;
	
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetHTJetPlusHOFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetHTJetPlusHOFilter);

/*
End of JetHTJetPlusHOFilter with event 20 Jetpassed 9 passed 1
*/

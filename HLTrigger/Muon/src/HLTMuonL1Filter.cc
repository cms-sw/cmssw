/** \class HLTMuonL1Filter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL1Filter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"


//
// constructors and destructor
//
HLTMuonL1Filter::HLTMuonL1Filter(const edm::ParameterSet& iConfig) :
   candTag_   (iConfig.getParameter< edm::InputTag > ("CandTag")),
   max_Eta_   (iConfig.getParameter<double> ("MaxEta")),
   min_Pt_ (iConfig.getParameter<double> ("MinPt")),
   min_Quality_    (iConfig.getParameter<int> ("MinQuality")),
   min_N_ (iConfig.getParameter<int> ("MinN"))
{

   LogDebug("HLTMuonL1Filter")
      << " CandTag/MaxEta/MinPt/MinQuality/MinN : " 
      << candTag_.encode()
      << " " << max_Eta_
      << " " << min_Pt_
      << " " << min_Quality_
      << " " << min_N_;

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonL1Filter::~HLTMuonL1Filter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonL1Filter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace trigger;
   using namespace l1extra;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterproduct (new TriggerFilterObjectWithRefs(path(),module()));

   // get hold of muons
   Handle<TriggerFilterObjectWithRefs> mucands;
   iEvent.getByLabel (candTag_,mucands);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   vector<L1MuonParticleRef> l1mu;
   mucands->getObjects(TriggerL1Mu,l1mu);
   for (unsigned int i=0; i<l1mu.size(); i++) {
      L1MuonParticleRef muon = L1MuonParticleRef(l1mu[i]);

      LogDebug("HLTMuonL1Filter") 
            << " Muon in loop: q*pt= " << muon->charge()*muon->pt() 
            << ", eta= " << muon->eta();
      float eta   =  muon->eta();
      if (fabs(eta)>max_Eta_) continue;
      float pt    =  muon->pt();
      if (pt<min_Pt_) continue;

      int quality =  999;
      if ( !muon->gmtMuonCand().empty() ) {
            quality = muon->gmtMuonCand().quality();
      }
      LogDebug("HLTMuonL1Filter") 
            << " Muon in loop, quality= " << quality;
      if (quality<min_Quality_) continue;

      n++;
      filterproduct->addObject(TriggerL1Mu,muon);
   }
   vector<L1MuonParticleRef> vref;
   filterproduct->getObjects(TriggerL1Mu,vref);
   for (unsigned int i=0; i<vref.size(); i++) {
      L1MuonParticleRef mu = L1MuonParticleRef(vref[i]);
      LogDebug("HLTMuonL1Filter")
           << " Muon passing filter: pt= " << mu->pt() << ", eta: " 
            << mu->eta();
   }

   // filter decision
   const bool accept (n >= min_N_);

   // put filter object into the Event
   iEvent.put(filterproduct);

   LogDebug("HLTMuonL1Filter") << " >>>>> Result of HLTMuonL1Filter is " << accept << ", number of muons passing thresholds= " << n; 

   return accept;
}

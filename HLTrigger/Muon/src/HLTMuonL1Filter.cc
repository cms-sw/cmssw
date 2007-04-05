/** \class HLTMuonL1Filter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL1Filter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
//#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"

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
   produces<reco::HLTFilterObjectWithRefs>();
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
   using namespace reco;
   using namespace l1extra;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs>
     filterproduct (new HLTFilterObjectWithRefs(path(),module()));

   // get hold of muons
   Handle<HLTFilterObjectWithRefs> mucands;
   iEvent.getByLabel (candTag_,mucands);

   // look at all mucands,  check cuts and add to filter object
   int n = 0;
   for (unsigned int i=0; i<mucands->size(); i++) {
      RefToBase<Candidate> cand = mucands->getParticleRef(i);
      if (typeid(*cand)!=typeid(L1MuonParticle)) continue;
      L1MuonParticleRef muon = cand.castTo<L1MuonParticleRef>();
      if (muon.isNull()) continue;

      LogDebug("HLTMuonL1Filter") 
            << " Muon in loop: pt= " << muon->pt() 
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
      filterproduct->putParticle(cand);
   }

   for (unsigned int i=0; i<filterproduct->size(); i++) {
      RefToBase<Candidate> mu = filterproduct->getParticleRef(i);
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

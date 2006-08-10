/** \class HLTLevel1Seed
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/10 17:09:24 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTLevel1Seed.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// constructors and destructor
//
HLTLevel1Seed::HLTLevel1Seed(const edm::ParameterSet& iConfig) :
  andOr_  (iConfig.getParameter<bool> ("andOr" )),
  byName_ (iConfig.getParameter<bool> ("byName"))
{
   unsigned int n;

   if (byName_) {
     // have names, need to get slot numbers
     L1SeedsByName_= iConfig.getParameter<std::vector<std::string > >("L1SeedsByName");
     n=L1SeedsByName_.size();
     L1SeedsByType_.resize(n);
     std::string name;
     for (unsigned int i=0; i!=n; i++) {
       name=L1SeedsByName_[i];
       L1SeedsByType_[i]=(unsigned int) (l1extra::L1ParticleMap::triggerType(name));
     }
   } else {
     // have slot numbers, need to get names
     L1SeedsByType_= iConfig.getParameter<std::vector<unsigned int> >("L1SeedsByType");
     n=L1SeedsByType_.size();
     L1SeedsByName_.resize(n);
     l1extra::L1ParticleMap::L1TriggerType type;
     for (unsigned int i=0; i!=n; i++) {
       type=(l1extra::L1ParticleMap::L1TriggerType) (L1SeedsByType_[i]);
       L1SeedsByName_[i]=l1extra::L1ParticleMap::triggerName(type);
     }
   }

   LogDebug("") << "Level-1 triggers requested: " << n << " - andOr mode: " << andOr_ << " - byName: " << byName_;
   if (n>0) {
     LogDebug("") << "  Level-1 triggers requestd: type and name:";
     for (unsigned int i=0; i!=n; i++) {
       LogTrace("") << " " << L1SeedsByType_[i] << " " << L1SeedsByName_[i];
     }
   }

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTLevel1Seed::~HLTLevel1Seed()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTLevel1Seed::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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
     filterobject (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RefToBase<Candidate> ref;

   // get hold of (single?) L1ParticleMapCollection
   Handle<L1ParticleMapCollection> l1pmch;
   iEvent.getByType(l1pmch);
   if (l1pmch.isValid()) {
     iEvent.put(filterobject);
     LogDebug("") << "No L1ParticleMapCollection found!";
     return false;
   }
   const unsigned int m(l1pmch->size());

   // check requested L1 triggers, and get index into L1ParticleMapCollection
   const unsigned int n(L1SeedsByName_.size());
   vector<int> index(n);
   L1ParticleMap::L1TriggerType l1tt;
   for (unsigned int i=0; i!=n; i++) {
     index[i]=-1;
     l1tt= (L1ParticleMap::L1TriggerType) (L1SeedsByType_[i]);
     for (unsigned int j=0; j!=m; j++) {
       const L1ParticleMap& l1pm((*l1pmch)[j]);
       if ( (l1tt==l1pm.triggerType()) && (l1pm.triggerDecision()) ) {
	 index[i]=j;
	 break;
       }
     }
   }

   // have all (and-mode) or at least one (or-mode) trigger fired?
   unsigned int fired(0);
   for (unsigned int i=0; i!=n; i++) if (index[i]>=0) fired++;
   const bool accept( ((!andOr_) && (fired==n)) ||
		      (( andOr_) && (fired!=0)) );

   // in case of accept, record all physics objects of all triggers
   if (accept) {
     unsigned int m(0);
     for (unsigned int i=0; i!=n; i++) {
       const L1ParticleMap& l1pm((*l1pmch)[index[i]]);
       if (index[i]>=0) {
	 // em particles (gamma+electron)
	 m=l1pm.emParticles().size();
         for (unsigned int j=0; j!=m; i++) {
	   ref=RefToBase<Candidate>( l1pm.emParticles()[j] );
	   filterobject->putParticle(ref);
	 }
	 // muon particles
	 m=l1pm.muonParticles().size();
         for (unsigned int j=0; j!=m; i++) {
	   ref=RefToBase<Candidate>( l1pm.muonParticles()[j] );
	   filterobject->putParticle(ref);
	 }
	 // tau particles
	 m=l1pm.tauParticles().size();
         for (unsigned int j=0; j!=m; i++) {
	   ref=RefToBase<Candidate>( l1pm.tauParticles()[j] );
	   filterobject->putParticle(ref);
	 }
	 // jet particles
	 m=l1pm.jetParticles().size();
         for (unsigned int j=0; j!=m; i++) {
	   ref=RefToBase<Candidate>( l1pm.jetParticles()[j] );
	   filterobject->putParticle(ref);
	 }
	 // (single global) met "particle"
         if (l1pm.etMissParticle() != L1EtMissParticleRefProd() ) {
	   ref=RefToBase<Candidate>( l1pm.etMissParticle() );
	   filterobject->putParticle(ref);
	 }

       }
     }
   }

   // put filter object into the Event
   iEvent.put(filterobject);

   return accept;
}

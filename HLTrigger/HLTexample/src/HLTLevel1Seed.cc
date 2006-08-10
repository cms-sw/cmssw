/** \class HLTLevel1Seed
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/10 17:31:14 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTLevel1Seed.h"

#include "FWCore/Framework/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
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
   if (!l1pmch.isValid()) {
     iEvent.put(filterobject);
     LogDebug("") << "No L1ParticleMapCollection found!";
     return false;
   }
   const unsigned int m(l1pmch->size());

   // get index into L1ParticleMapCollection for requested L1 triggers
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

   // filter result: have all (and-mode) / at least one of (or-mode)
   // the requested L1 triggers fired?
   unsigned int fired(0);
   for (unsigned int i=0; i!=n; i++) if (index[i]>=0) fired++;
   const bool accept( ((!andOr_) && (fired==n)) ||
		      (( andOr_) && (fired!=0)) );

   // number of of particles of each type eventually recorded in filter object
   unsigned int ne(0);
   unsigned int nm(0);
   unsigned int nt(0);
   unsigned int nj(0);
   unsigned int nM(0);

   // in case of accept, record all particles used
   // by any requested L1 triggers which fired
   if (accept) {

     // we do not want to store duplicates, hence we determine
     // which L1 particle is used by one of the requested triggers
     // and then store the used L1 particles only once

     // get hold of overall collections of particles of each type
     // will check later which are used by which requested L1 trigger
     Handle<L1EmParticleCollection  > l1eh;
     iEvent.getByType(l1eh);
     unsigned int Ne(0);
     if (l1eh.isValid()) Ne=l1eh->size();
     vector<unsigned int> ve(Ne,0);
     // keeps track how often each Em particle is used

     Handle<L1MuonParticleCollection> l1mh;
     iEvent.getByType(l1mh);
     unsigned int Nm(0);
     if (l1mh.isValid()) Nm=l1mh->size();
     vector<unsigned int> vm(Nm,0);
     // keeps track how often each Muon particle is used

     Handle<L1JetParticleCollection>  l1th; // taus are stored as jets
     iEvent.getByType(l1th);
     unsigned int Nt(0);
     if (l1th.isValid()) Nt=l1th->size();
     vector<unsigned int> vt(Nt,0);
     // keeps track how often each Tau particle is used

     Handle<L1JetParticleCollection>  l1jh;
     iEvent.getByType(l1jh);
     unsigned int Nj(0);
     if (l1jh.isValid()) Nj=l1th->size();
     vector<unsigned int> vj(Nj,0);
     // keeps track how often each Jet particle is used

     Handle<L1EtMissParticle> l1Mh;
     iEvent.getByType(l1Mh);
     unsigned int NM(0);
     if (l1Mh.isValid()) NM=1;
     vector<unsigned int> vM(NM,0);
     // keeps track how often the global EtMiss "particle" is used

     // loop over requested triggers and count which particles are used
     for (unsigned int i=0; i!=n; i++) {
       const L1ParticleMap& l1pm((*l1pmch)[index[i]]);
       if (index[i]>=0) { // requested and fired!
         unsigned int m(0);
	 // em particles (gamma+electron)
	 m=l1pm.emParticles().size();
         for (unsigned int j=0; j!=m; i++) ve[l1pm.emParticles()[j].key()]++;
	 // muon particles
	 m=l1pm.muonParticles().size();
         for (unsigned int j=0; j!=m; i++) vm[l1pm.muonParticles()[j].key()]++;
	 // tau particles
	 m=l1pm.tauParticles().size();
         for (unsigned int j=0; j!=m; i++) vt[l1pm.tauParticles()[j].key()]++;
	 // jet particles
	 m=l1pm.jetParticles().size();
         for (unsigned int j=0; j!=m; i++) vj[l1pm.jetParticles()[j].key()]++;
	 // (single global) met "particle"
         if (l1pm.etMissParticle() != L1EtMissParticleRefProd() ) vM[0]++;
       }
     }

     // record used physics objects in filterobject
     for (unsigned int i=0; i!=Ne; i++) {
       if (ve[i]>0) {
	 ref=RefToBase<Candidate>(L1EmParticleRef  (l1eh,i));
	 filterobject->putParticle(ref);
	 ne++;
       }
     }
     for (unsigned int i=0; i!=Nm; i++) {
       if (vm[i]>0) {
	 ref=RefToBase<Candidate>(L1MuonParticleRef(l1mh,i));
	 filterobject->putParticle(ref);
	 nm++;
       }
     }
     for (unsigned int i=0; i!=Nt; i++) {
       if (vt[i]>0) {             // taus are stored as jets
	 ref=RefToBase<Candidate>(L1JetParticleRef (l1th,i));
	 filterobject->putParticle(ref);
	 nt++;
       }
     }
     for (unsigned int i=0; i!=Nj; i++) {
       if (vj[i]>0) {
	 ref=RefToBase<Candidate>(L1JetParticleRef (l1jh,i));
	 filterobject->putParticle(ref);
	 nj++;
       }
     }
     for (unsigned int i=0; i!=NM; i++) {
       if (vM[i]>0) {
	 ref=RefToBase<Candidate>(L1EtMissParticleRefProd(l1Mh));
	 filterobject->putParticle(ref);
	 nM++;
       }
     }

   }

   // put filter object into the Event
   iEvent.put(filterobject);

   LogDebug("") << "Number of e/m/t/j/M particles used: "
		<< ne << " " << nm << " " << nt << " " << nj << " " << nM;

   return accept;

}

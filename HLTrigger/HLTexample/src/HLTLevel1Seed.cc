/** \class HLTLevel1Seed
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/22 17:20:45 $
 *  $Revision: 1.16 $
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
  L1ExtraTag_(iConfig.getParameter<edm::InputTag> ("L1ExtraTag")),
  andOr_     (iConfig.getParameter<bool> ("andOr" )),
  byName_    (iConfig.getParameter<bool> ("byName"))
{
  const string invalid("@@invalid@@");
 
  unsigned int n(0);

  if (byName_) {
    // read names, then get slot numbers
    L1SeedsByName_= iConfig.getParameter<std::vector<std::string > >("L1Seeds");
    n=L1SeedsByName_.size();
    L1SeedsByType_.resize(n);
    std::string name;
    for (unsigned int i=0; i!=n; i++) {
      name=L1SeedsByName_[i];
      L1SeedsByType_[i]=l1extra::L1ParticleMap::triggerType(name);
    }
  } else {
    // read slot numbers, then get names
    L1SeedsByType_= iConfig.getParameter<std::vector<unsigned int> >("L1Seeds");
    n=L1SeedsByType_.size();
    L1SeedsByName_.resize(n);
    for (unsigned int i=0; i!=n; i++) {
      if (L1SeedsByType_[i]<l1extra::L1ParticleMap::kNumOfL1TriggerTypes) {
	l1extra::L1ParticleMap::L1TriggerType 
	  type(static_cast<l1extra::L1ParticleMap::L1TriggerType>(L1SeedsByType_[i]));
	L1SeedsByName_[i]=l1extra::L1ParticleMap::triggerName(type);
      } else {
	L1SeedsByName_[i]=invalid;
      }
    }
  }
  
  // for empty input vectors (n=0), default to all triggers!
  if (n==0) {
    n=(unsigned int) (l1extra::L1ParticleMap::kNumOfL1TriggerTypes);
    L1SeedsByName_.resize(n);
    L1SeedsByType_.resize(n);
    for (unsigned int i=0; i!=n; i++) {
      L1SeedsByType_[i]=i;
      l1extra::L1ParticleMap::L1TriggerType type;
      type=(l1extra::L1ParticleMap::L1TriggerType) (L1SeedsByType_[i]);
      L1SeedsByName_[i]=l1extra::L1ParticleMap::triggerName(type);
    }
  }
  
  LogDebug("") << "Level-1 triggers: " +L1ExtraTag_.encode()
	       << " - Number requested: " << n 
	       << " - andOr mode: " << andOr_
	       << " - byName: " << byName_;
  if (n>0) {
    LogDebug("") << "  Level-1 triggers requestd: type, name and status:";
    for (unsigned int i=0; i!=n; i++) {
      LogTrace("") << " " << L1SeedsByType_[i]
		   << " " << L1SeedsByName_[i]
		   << " " <<( (L1SeedsByType_[i]<l1extra::L1ParticleMap::kNumOfL1TriggerTypes) &&
			      (L1SeedsByName_[i]!=invalid) ) ;
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


   // get hold of L1GlobalReadoutRecord
   Handle<L1GlobalTriggerReadoutRecord> l1gtrrh;
   try {iEvent.getByLabel(L1ExtraTag_,l1gtrrh);}
   catch (...) {
     LogDebug("") << "L1GlobalTriggerReadoutRecord with label ["+L1ExtraTag_.encode()+"] not found!";
     //     iEvent.put(filterobject);
     //     return false;
   }
   if (l1gtrrh.isValid()) {
     LogDebug("") << "L1GlobalTriggerReadoutRecord decision: " << l1gtrrh->decision();
   } else {
     LogDebug("") << "L1GlobalTriggerReadoutRecord with label ["+L1ExtraTag_.encode()+"] not valid!";
     //     iEvent.put(filterobject);
     //     return false;
   }

   // get hold of L1ParticleMapCollection
   Handle<L1ParticleMapCollection> l1pmch;
   try {iEvent.getByLabel(L1ExtraTag_,l1pmch);}
   catch (...) {
     LogDebug("") << "L1ParticleMapCollection with label ["+L1ExtraTag_.encode()+"] not found!";
     iEvent.put(filterobject);
     return false;
   }
   if (l1pmch.isValid()) {
     LogDebug("") << "L1ParticleMapCollection contains " << l1pmch->size() << " maps.";
   } else {
     LogDebug("") << "L1ParticleMapCollection with label ["+L1ExtraTag_.encode()+"] not valid!";
     iEvent.put(filterobject);
     return false;
   }
   const unsigned int m(l1pmch->size());


   // get indices into L1ParticleMapCollection for requested 
   // L1 triggers which have fired this event
   const unsigned int n(L1SeedsByName_.size());
   vector<int> index(n,-1);
   L1ParticleMap::L1TriggerType l1tt;
   for (unsigned int i=0; i!=n; i++) {
     l1tt=static_cast<L1ParticleMap::L1TriggerType>(L1SeedsByType_[i]);
     for (unsigned int j=0; j!=m; j++) {
       const L1ParticleMap& l1pm((*l1pmch)[j]);
       if ( (l1tt==l1pm.triggerType()) && (l1pm.triggerDecision()) ) {
	 index[i]=j;
	 break;
       }
     }
   }


   // HLT Level1Seed Filter Result:
   // have all (and-mode) / at least one (or-mode)
   // of the requested L1 triggers fired this event?
   unsigned int fired(0);
   for (unsigned int i=0; i!=n; i++) if (index[i]>=0) fired++;
   const bool accept( ((!andOr_) && (fired==n)) ||
		      (( andOr_) && (fired!=0)) );
   LogDebug("") << "Accept = " << accept;

   // number of of particles of each type eventually recorded in filter object
   unsigned int ne(0); // em
   unsigned int nm(0); // muon
   unsigned int nt(0); // tau
   unsigned int nj(0); // jets
   unsigned int nM(0); // mets

   // in case of accept, record all particles used
   // by any requested L1 triggers which has fired
   if (accept) {

     // we do not want to store duplicates, hence we determine
     // which L1 particle is used by the requested triggers
     // and then store the used L1 particles only once

     // get hold of overall collections of particles of each type
     // will check later which are used by which requested L1 trigger
     Handle<L1EmParticleCollection  > l1eh;
     try {iEvent.getByLabel(L1ExtraTag_,l1eh);} catch(...) {;}
     unsigned int Ne(0);
     if (l1eh.isValid()) Ne=l1eh->size();
     //     LogDebug("") << "L1EmParticleCollection size = " << Ne;
     vector<unsigned int> ve(Ne,0);
     // keeps track how often each Em particle is used

     Handle<L1MuonParticleCollection> l1mh;
     try {iEvent.getByLabel(L1ExtraTag_,l1mh);} catch (...) {;}
     unsigned int Nm(0);
     if (l1mh.isValid()) Nm=l1mh->size();
     //     LogDebug("") << "L1MuonParticleCollection size = " << Nm;
     vector<unsigned int> vm(Nm,0);
     // keeps track how often each Muon particle is used

     Handle<L1JetParticleCollection>  l1th; // taus are stored as jets
     InputTag L1ExtraTauTag(InputTag(L1ExtraTag_.label(),"Tau"));
     try {iEvent.getByLabel(L1ExtraTauTag,l1th);} catch (...) {;}
     unsigned int Nt(0);
     if (l1th.isValid()) Nt=l1th->size();
     //     LogDebug("") << "L1TauParticleCollection size = " << Nt;
     vector<unsigned int> vt(Nt,0);
     // keeps track how often each Tau particle is used

     Handle<L1JetParticleCollection>  l1jh;
     InputTag L1ExtraJetTag(InputTag(L1ExtraTag_.label(),"ForCen"));
     try {iEvent.getByLabel(L1ExtraJetTag,l1jh);} catch (...) {;}
     unsigned int Nj(0);
     if (l1jh.isValid()) Nj=l1jh->size();
     //     LogDebug("") << "L1JetParticleCollection size = " << Nj;
     vector<unsigned int> vj(Nj,0);
     // keeps track how often each Jet particle is used

     Handle<L1EtMissParticle> l1Mh;
     try {iEvent.getByLabel(L1ExtraTag_,l1Mh);} catch (...) {;}
     unsigned int NM(0);
     if (l1Mh.isValid()) NM=1;
     //     LogDebug("") << "L1EtMissParticle size = " << NM;
     vector<unsigned int> vM(NM,0);
     // keeps track how often the global EtMiss "particle" is used

     // loop over requested triggers
     for (unsigned int i=0; i!=n; i++) {
       //       LogDebug("") << "Accessing L1 trigger: " << i
       //		    << "=" << L1SeedsByName_[i]
       //		    << ":" << L1SeedsByType_[i]
       //		    << " " << index[i];
       // has requested trigger fired?
       if (index[i]>=0) { // requested and fired!
	 // if yes, count which particles of each type have been used
	 const L1ParticleMap& l1pm((*l1pmch)[index[i]]);
         unsigned int m(0);

	 // em particles (gamma+electron)
	 m=l1pm.emParticles().size();
	 //	 LogDebug("") << "e " << m;
         for (unsigned int j=0; j!=m; j++) ve[l1pm.emParticles()[j].key()]++;

	 // muon particles
	 m=l1pm.muonParticles().size();
	 //	 LogDebug("") << "m " << m;
         for (unsigned int j=0; j!=m; j++) vm[l1pm.muonParticles()[j].key()]++;

	 // tau particles
	 m=l1pm.tauParticles().size();
	 //	 LogDebug("") << "t " << m;
         for (unsigned int j=0; j!=m; j++) vt[l1pm.tauParticles()[j].key()]++;

	 // jet particles
	 m=l1pm.jetParticles().size();
	 //	 LogDebug("") << "j " << m;
         for (unsigned int j=0; j!=m; j++) vj[l1pm.jetParticles()[j].key()]++;

	 // (single global) met "particle"
	 //	 LogDebug("") << "Ma " << vM[0];
         if (l1pm.etMissParticle() != L1EtMissParticleRefProd() ) vM[0]++;
	 //	 LogDebug("") << "Mb " << vM[0];
       }
     }

     // record used physics objects in filterobject
     //     LogDebug("") << "Inserting into filter object:";

     // em particles (gamma+electron)
     for (unsigned int i=0; i!=Ne; i++) if (ve[i]>0) {
       ref=RefToBase<Candidate>(L1EmParticleRef  (l1eh,i));
       filterobject->putParticle(ref);
       ne++;
     }
     //     LogDebug("") << "Inserted e: " << ne;

     // muon particles
     for (unsigned int i=0; i!=Nm; i++) if (vm[i]>0) {
       ref=RefToBase<Candidate>(L1MuonParticleRef(l1mh,i));
       filterobject->putParticle(ref);
       nm++;
     }
     //     LogDebug("") << "Inserted m: " << nm;

     // tau particles (taus are stored as jets!)
     for (unsigned int i=0; i!=Nt; i++) if (vt[i]>0) {
       ref=RefToBase<Candidate>(L1JetParticleRef (l1th,i));
       filterobject->putParticle(ref);
       nt++;
     }
     //     LogDebug("") << "Inserted t: " << nt;

     // jet particles
     for (unsigned int i=0; i!=Nj; i++) if (vj[i]>0) {
       ref=RefToBase<Candidate>(L1JetParticleRef (l1jh,i));
       filterobject->putParticle(ref);
       nj++;
     }
     //     LogDebug("") << "Inserted j: " << nj;

     // (single global) met "particle"
     for (unsigned int i=0; i!=NM; i++) if (vM[i]>0) {
       ref=RefToBase<Candidate>(L1EtMissParticleRefProd(l1Mh));
       filterobject->putParticle(ref);
       nM++;
     }
     //     LogDebug("") << "Inserted M: " << nM;

   } // if (accept)

   // put filter object into the Event
   iEvent.put(filterobject);

   LogDebug("") << "Number of e/m/t/j/M particles used: "
		<< ne << " " << nm << " " << nt << " " << nj << " " << nM;

   return accept;

}

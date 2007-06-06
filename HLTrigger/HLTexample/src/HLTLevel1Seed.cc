/** \class HLTLevel1Seed
 *
 * See header file for documentation
 *
 *  $Date: 2007/04/13 15:57:58 $
 *  $Revision: 1.24 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTLevel1Seed.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cassert>

//
// constructors and destructor
//
HLTLevel1Seed::HLTLevel1Seed(const edm::ParameterSet& iConfig) :
  l1CollectionsTag_ (iConfig.getParameter<edm::InputTag> ("L1ExtraCollections")),
  l1ParticleMapTag_ (iConfig.getParameter<edm::InputTag> ("L1ExtraParticleMap")),
  l1GTReadoutRecTag_(iConfig.getParameter<edm::InputTag> ("L1GTReadoutRecord")),
  andOr_   (iConfig.getParameter<bool> ("andOr" )),
  byName_  (iConfig.getParameter<bool> ("byName"))
{
  const string invalid("@@invalid@@");
 
  unsigned int n(0);

  if (byName_) {
    // get names, then derive slot numbers
    L1SeedsByName_= iConfig.getParameter<std::vector<std::string > >("L1Seeds");
    n=L1SeedsByName_.size();
    L1SeedsByType_.resize(n);
    std::string name;
    for (unsigned int i=0; i!=n; i++) {
      name=L1SeedsByName_[i];
      L1SeedsByType_[i]=l1extra::L1ParticleMap::triggerType(name);
    }
  } else {
    // get slot numbers, then derive names
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
  
  // for empty input vectors (n=0), default to all L1 triggers!
  if (n==0) {
    n=static_cast<unsigned int>(l1extra::L1ParticleMap::kNumOfL1TriggerTypes);
    L1SeedsByName_.resize(n);
    L1SeedsByType_.resize(n);
    for (unsigned int i=0; i!=n; i++) {
      L1SeedsByType_[i]=i;
      l1extra::L1ParticleMap::L1TriggerType type;
      type=(l1extra::L1ParticleMap::L1TriggerType) (L1SeedsByType_[i]);
      L1SeedsByName_[i]=l1extra::L1ParticleMap::triggerName(type);
    }
  }
  
  LogDebug("") << "Level-1 triggers: "
	       << "L1ExtraCollections: " + l1CollectionsTag_.encode()
	       << "L1ExtraParticleMap: " + l1ParticleMapTag_.encode()
	       << "L1GTReadoutRecord : " + l1GTReadoutRecTag_.encode()
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
   Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
   try {iEvent.getByLabel(l1GTReadoutRecTag_,L1GTRR);} catch (...) {;}
   if (L1GTRR.isValid()) {
     LogDebug("") << "L1GlobalTriggerReadoutRecord decision: " << L1GTRR->decision();
   } else {
     LogDebug("") << "L1GlobalTriggerReadoutRecord with label ["+l1GTReadoutRecTag_.encode()+"] not found!";
     // try to carry on!
   }


   // get hold of L1ParticleMapCollection
   Handle<L1ParticleMapCollection> L1PMC;
   try {iEvent.getByLabel(l1ParticleMapTag_,L1PMC);} catch (...) {;}
   if (L1PMC.isValid()) {
     LogDebug("") << "L1ParticleMapCollection contains " << L1PMC->size() << " maps.";
   } else {
     LogDebug("") << "L1ParticleMapCollection with label ["+l1ParticleMapTag_.encode()+"] not found!";
     iEvent.put(filterobject);
     return false;
   }
   const unsigned int m(L1PMC->size());
   assert (m==static_cast<unsigned int>(l1extra::L1ParticleMap::kNumOfL1TriggerTypes));

   // for requested L1 trigger i which has fired this event,
   // get index j into L1ParticleMapCollection
   // and store it in index[i]=j for later use
   const unsigned int n(L1SeedsByName_.size());
   vector<int> index(n,-1);
   L1ParticleMap::L1TriggerType l1tt;
   for (unsigned int i=0; i!=n; i++) {
     l1tt=static_cast<L1ParticleMap::L1TriggerType>(L1SeedsByType_[i]);
     for (unsigned int j=0; j!=m; j++) {
       const L1ParticleMap& L1PM((*L1PMC)[j]);
       if ( (l1tt==L1PM.triggerType()) && (L1PM.triggerDecision()) ) {
	 assert (j==L1SeedsByType_[i]);
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


   // number of L1 particles of each type eventually 
   // recorded in the HLT filter object
   unsigned int nEmIso    (0); // Em  (   Isolated)
   unsigned int nEmNonIso (0); // Em  (NonIsolated)
   unsigned int nMuon     (0); // Muon
   unsigned int nJetFor   (0); // Jet (Forward)
   unsigned int nJetCen   (0); // Jet (Central)
   unsigned int nJetTau   (0); // Jet (Tau)
   unsigned int nEtM      (0); // EtMiss (in fact only one "global" object)


   // in case of accept, record all particles used
   // by any requested L1 triggers which has fired
   if (accept) {

     // we do not want to store duplicates, hence we determine
     // which of all L1 particles are used by the requested triggers
     // and then store the used L1 particles only once

     // first, get hold of overall collections of particles of each type -
     // then, later, check which are used by which requested L1 trigger

     // Em (Isolated)
     InputTag L1EmIsoTag(InputTag(l1CollectionsTag_.label(),"Isolated"));
     Handle<L1EmParticleCollection> L1EmIso;
     try {iEvent.getByLabel(L1EmIsoTag,L1EmIso);} catch(...) {;}
     unsigned int NEmIso(0);
     if (L1EmIso.isValid()) NEmIso=L1EmIso->size();
     LogDebug("") << "L1EmParticleCollection(Isolated) size = " << NEmIso;
     vector<unsigned int> vEmIso(NEmIso,0);
     // keeps track how often each Isolated Em particle is used

     // Em (NonIsolated)
     InputTag L1EmNonIsoTag(InputTag(l1CollectionsTag_.label(),"NonIsolated"));
     Handle<L1EmParticleCollection> L1EmNonIso;
     try {iEvent.getByLabel(L1EmNonIsoTag,L1EmNonIso);} catch(...) {;}
     unsigned int NEmNonIso(0);
     if (L1EmNonIso.isValid()) NEmNonIso=L1EmNonIso->size();
     LogDebug("") << "L1EmParticleCollection(NonIsolated) size = " << NEmNonIso;
     vector<unsigned int> vEmNonIso(NEmNonIso,0);
     // keeps track how often each NonIsolated Em particle is used

     // Muon
     Handle<L1MuonParticleCollection> L1Muon;
     try {iEvent.getByLabel(l1CollectionsTag_,L1Muon);} catch (...) {;}
     unsigned int NMuon(0);
     if (L1Muon.isValid()) NMuon=L1Muon->size();
     LogDebug("") << "L1MuonParticleCollection size = " << NMuon;
     vector<unsigned int> vMuon(NMuon,0);
     // keeps track how often each Muon particle is used

     // Jets (Forward)
     InputTag L1JetForTag(InputTag(l1CollectionsTag_.label(),"Forward"));
     Handle<L1JetParticleCollection> L1JetFor;
     try {iEvent.getByLabel(L1JetForTag,L1JetFor);} catch(...) {;}
     unsigned int NJetFor(0);
     if (L1JetFor.isValid()) NJetFor=L1JetFor->size();
     LogDebug("") << "L1JetParticleCollection(Forward) size = " << NJetFor;
     vector<unsigned int> vJetFor(NJetFor,0);
     // keeps track how often each Forward Jet particle is used

     // Jets (Central)
     InputTag L1JetCenTag(InputTag(l1CollectionsTag_.label(),"Central"));
     Handle<L1JetParticleCollection> L1JetCen;
     try {iEvent.getByLabel(L1JetCenTag,L1JetCen);} catch(...) {;}
     unsigned int NJetCen(0);
     if (L1JetCen.isValid()) NJetCen=L1JetCen->size();
     LogDebug("") << "L1JetParticleCollection(Central) size = " << NJetCen;
     vector<unsigned int> vJetCen(NJetCen,0);
     // keeps track how often each Central Jet particle is used

     // Jets (Tau)
     InputTag L1JetTauTag(InputTag(l1CollectionsTag_.label(),"Tau"));
     Handle<L1JetParticleCollection> L1JetTau;
     try {iEvent.getByLabel(L1JetTauTag,L1JetTau);} catch(...) {;}
     unsigned int NJetTau(0);
     if (L1JetTau.isValid()) NJetTau=L1JetTau->size();
     LogDebug("") << "L1JetParticleCollection(Tau) size = " << NJetTau;
     vector<unsigned int> vJetTau(NJetTau,0);
     // keeps track how often each Tau Jet particle is used

     // (single global) MET
     Handle<L1EtMissParticle> L1EtM;
     try {iEvent.getByLabel(l1CollectionsTag_,L1EtM);} catch (...) {;}
     unsigned int NEtM(0);
     if (L1EtM.isValid()) NEtM=1;
     LogDebug("") << "L1EtMissParticle size = " << NEtM;
     vector<unsigned int> vEtM(NEtM,0);
     // keeps track how often the single global EtMiss "particle" is used


     // loop over requested triggers and count L1 objects
     // of those requested L1 triggers which fired
     for (unsigned int i=0; i!=n; i++) {
       //       LogDebug("") << "Accessing L1 trigger: " << i
       //		    << "=" << L1SeedsByName_[i]
       //		    << ":" << L1SeedsByType_[i]
       //		    << " " << index[i];
       //
       // has requested trigger fired?
       if (index[i]>=0) { // requested and fired!
	 // if yes, count which particles of each type have been used
	 const L1ParticleMap& L1PM((*L1PMC)[index[i]]);
         unsigned int m(0);

	 // Em particles used by this trigger (Isolated or NonIsolated)
	 m=L1PM.emParticles().size();
	 //	 LogDebug("") << "Em " << m;
         for (unsigned int j=0; j!=m; j++) {
	   const L1EmParticleRef& L1EmPRef(L1PM.emParticles()[j]);
	   if (L1EmPRef.id() == L1EmIso.id()) {
	     vEmIso.at(L1EmPRef.key())++;
	   } else if (L1EmPRef.id() == L1EmNonIso.id()) {
	     vEmNonIso.at(L1EmPRef.key())++;
	   } else {
	     LogDebug("") << "Error in analysing Em particle "
			  << j << " of L1 trigger " << i;
	   }
	 }

	 // Muon particles used by this trigger
	 m=L1PM.muonParticles().size();
	 //	 LogDebug("") << "Muon " << m;
         for (unsigned int j=0; j!=m; j++) {
	   const L1MuonParticleRef& L1MuonPRef(L1PM.muonParticles()[j]);
	   if (L1MuonPRef.id() == L1Muon.id()) {
	     vMuon.at(L1MuonPRef.key())++;
	   } else {
	     LogDebug("") << "Error in analysing Muon particle "
			  << j << " of L1 trigger " << i;
	   }
	 }

	 // Jet particles used by this trigger (Forward, Central, or Tau)
	 m=L1PM.jetParticles().size();
	 //	 LogDebug("") << "Jet " << m;
         for (unsigned int j=0; j!=m; j++) {
	   const L1JetParticleRef& L1JetPRef(L1PM.jetParticles()[j]);
	   if (L1JetPRef.id() == L1JetFor.id()) {
	     vJetFor.at(L1JetPRef.key())++;
	   } else if (L1JetPRef.id() == L1JetCen.id()) {
	     vJetCen.at(L1JetPRef.key())++;
	   } else if (L1JetPRef.id() == L1JetTau.id()) {
	     vJetTau.at(L1JetPRef.key())++;
	   } else {
	     LogDebug("") << "Error in analysing Jet particle "
			  << j << " of L1 trigger " << i;
	   }
	 }

	 // (single global) met "particle"
	 //	 LogDebug("") << "Ma " << vEtM[0];
	 if (L1PM.etMissParticle().isNonnull()) {
	   const L1EtMissParticleRefProd& L1EtMPRefProd(L1PM.etMissParticle());
	   if (L1EtMPRefProd.id() == L1EtM.id()) {
	     vEtM.at(0)++;
	   } else {
	     LogDebug("") << "Error in analysing EtM particle "
			  << "of L1 trigger " << i;
	   }
	 }

       }
     }

     // finally, record these used L1 physics objects
     // in the HLT filterobject
     //     LogDebug("") << "Inserting into filter object:";

     // Em particles (Isolated)
     for (unsigned int i=0; i!=NEmIso; i++) {
       if (vEmIso[i]>0) {
	 ref=RefToBase<Candidate>(L1EmParticleRef(L1EmIso,i));
	 filterobject->putParticle(ref);
	 nEmIso++;
       }
     }

     // Em particles (NonIsolated)
     for (unsigned int i=0; i!=NEmNonIso; i++) {
       if (vEmNonIso[i]>0) {
	 ref=RefToBase<Candidate>(L1EmParticleRef(L1EmNonIso,i));
	 filterobject->putParticle(ref);
	 nEmNonIso++;
       }
     }

     // Muon particles
     for (unsigned int i=0; i!=NMuon; i++) {
       if (vMuon[i]>0) {
	 ref=RefToBase<Candidate>(L1MuonParticleRef(L1Muon,i));
	 filterobject->putParticle(ref);
	 nMuon++;
       }
     }

     // Jet particles (Forward)
     for (unsigned int i=0; i!=NJetFor; i++) {
       if (vJetFor[i]>0) {
	 ref=RefToBase<Candidate>(L1JetParticleRef(L1JetFor,i));
	 filterobject->putParticle(ref);
	 nJetFor++;
       }
     }

     // Jet particles (Central)
     for (unsigned int i=0; i!=NJetCen; i++) {
       if (vJetCen[i]>0) {
	 ref=RefToBase<Candidate>(L1JetParticleRef(L1JetCen,i));
	 filterobject->putParticle(ref);
	 nJetCen++;
       }
     }

     // Jet particles (Tau)
     for (unsigned int i=0; i!=NJetTau; i++) {
       if (vJetTau[i]>0) {
	 ref=RefToBase<Candidate>(L1JetParticleRef(L1JetTau,i));
	 filterobject->putParticle(ref);
	 nJetTau++;
       }
     }

     // (single global) met "particle"
     for (unsigned int i=0; i!=NEtM; i++) {
       if (vEtM[i]>0) {
	 ref=RefToBase<Candidate>(L1EtMissParticleRefProd(L1EtM));
	 filterobject->putParticle(ref);
	 nEtM++;
       }
     }

   } // if (accept)


   // put filter object into the Event
   iEvent.put(filterobject);

   LogDebug("") << "Number of EmIso/EmNonIso/Muon/JetFor/JetCen/JetTau/EtM particles used:"
		<< " " << nEmIso
		<< " " << nEmNonIso
		<< " " << nMuon
		<< " " << nJetFor
		<< " " << nJetCen
		<< " " << nJetTau
		<< " " << nEtM
		;

   return accept;

}

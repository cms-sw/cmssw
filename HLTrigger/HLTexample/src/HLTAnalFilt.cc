/** \class HLTAnalFilt
 *
 * See header file for documentation
 *
 *  $Date: 2007/02/17 14:39:20 $
 *  $Revision: 1.21 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTexample/interface/HLTAnalFilt.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include<typeinfo>

//
// constructors and destructor
//
 
HLTAnalFilt::HLTAnalFilt(const edm::ParameterSet& iConfig) :
  inputTag_(iConfig.getParameter<edm::InputTag>("inputTag"))
{
  LogDebug("") << "Input: " << inputTag_.encode();
}

HLTAnalFilt::~HLTAnalFilt()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void
HLTAnalFilt::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   // get hold of (single?) TriggerResults object
   vector<Handle<TriggerResults> > trhv;
   iEvent.getManyByType(trhv);
   const unsigned int n(trhv.size());
   LogDebug("") << "Number of TriggerResults objects found: " << n;
   for (unsigned int i=0; i!=n; i++) {
     LogDebug("") << "TriggerResult object " << i << " bits: " << *(trhv[i]);
   }

   // get hold of requested filter object
   Handle<HLTFilterObjectWithRefs> ref;
   try {iEvent.getByLabel(inputTag_,ref);} catch(...) {;}
   if (ref.isValid()) {
     const unsigned int n(ref->size());
     LogDebug("") << inputTag_.encode() + " Size = " << n;
     for (unsigned int i=0; i!=n; i++) {
       // some Xchecks
       Particle particle(ref->getParticle(i));
       const Candidate* candidate((ref->getParticleRef(i)).get());
       LogTrace("") << i << " E: " 
		    << particle.energy() << " " << candidate->energy() << " "  
		    << typeid(*candidate).name() << " "
		    << particle.eta() << " " << particle.phi() ;
     }

     //
     // using HLTFilterObjectsWithRefs like a ConcreteCollection:
     //
     HLTFilterObjectWithRefs::const_iterator a(ref->begin());
     HLTFilterObjectWithRefs::const_iterator o(ref->end());
     HLTFilterObjectWithRefs::const_iterator i;
     const HLTFilterObjectWithRefs& V(*ref);
     LogTrace("") << "Size: " << V.size();
     for (i=a; i!=o; i++) {
       unsigned int I(i-a);
       LogTrace("") << "Const_Iterator: " << I << " " << typeid(*i).name()
		    << " " << i->energy();
       LogTrace("") << "Handle->at(i):  " << I << " " << typeid(ref->at(I)).name()
		    << " " << (ref->at(I)).energy();
       LogTrace("") << "Vector[i]:      " << I << " " << typeid(V[I]).name()
		    << " " << V[I].energy();
       LogTrace("") << "Vector.at(i):   " << I << " " << typeid(V.at(I)).name()
		    << " " << V.at(I).energy();
       LogTrace("") << "                " << I << " " << typeid(&(*i)).name();
       LogTrace("") << "                " << I << " " << typeid(  *i ).name();
       LogTrace("") << "                " << I << " " << typeid(   i ).name();
     }
     //
   } else {
     LogDebug("") << "Filterobject " + inputTag_.encode() + " not found!";
   }

   return;
}

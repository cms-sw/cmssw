/** \class HLTDoublet
 *
 * See header file for documentation
 *
 *  $Date: 2007/03/26 11:39:20 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTfilters/interface/HLTDoublet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<cmath>

//
// constructors and destructor
//
HLTDoublet::HLTDoublet(const edm::ParameterSet& iConfig) :
  inputTag1_(iConfig.getParameter<edm::InputTag>("inputTag1")),
  inputTag2_(iConfig.getParameter<edm::InputTag>("inputTag2")),
  min_Dphi_ (iConfig.getParameter<double>("MinDphi")),
  max_Dphi_ (iConfig.getParameter<double>("MaxDphi")),
  min_Deta_ (iConfig.getParameter<double>("MinDeta")),
  max_Deta_ (iConfig.getParameter<double>("MaxDeta")),
  min_Minv_ (iConfig.getParameter<double>("MinMinv")),
  max_Minv_ (iConfig.getParameter<double>("MaxMinv")),
  min_N_    (iConfig.getParameter<int>("MinN"))
{
   // same collections to be compared?
   same_ = (inputTag1_.encode()==inputTag2_.encode());

   cutdphi_ = (min_Dphi_ <= max_Dphi_); // cut active?
   cutdeta_ = (min_Deta_ <= max_Deta_); // cut active?
   cutminv_ = (min_Minv_ <= max_Minv_); // cut active?

   LogDebug("") << "InputTags and cuts : " 
		<< inputTag1_.encode() << " " << inputTag2_.encode()
		<< " Dphi [" << min_Dphi_ << " " << max_Dphi_ << "]"
                << " Deta [" << min_Deta_ << " " << max_Deta_ << "]"
                << " Minv [" << min_Minv_ << " " << max_Minv_ << "]"
                << " MinN =" << min_N_
		<< " same/dphi/deta/minv "
		<< same_ << cutdphi_ << cutdeta_ << cutminv_;

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTDoublet::~HLTDoublet()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTDoublet::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<HLTFilterObjectWithRefs>
     filterobject (new HLTFilterObjectWithRefs(path(),module()));
   // Ref to Candidate object to be recorded in filter object
   RefToBase<Candidate> r1,r2;


   // get hold of pre-filtered object collections
   Handle<HLTFilterObjectWithRefs> coll1,coll2;
   iEvent.getByLabel (inputTag1_,coll1);
   iEvent.getByLabel (inputTag2_,coll2);

   int n(0);
   const unsigned int n1(coll1->size());
   const unsigned int n2(coll2->size());
   Particle p1,p2,p;
   for (unsigned int i1=0; i1!=n1; i1++) {
     p1=coll1->getParticle(i1);
     r1=coll1->getParticleRef(i1);
     unsigned int I(0);
     if (same_) {I=i1+1;}
     for (unsigned int i2=I; i2!=n2; i2++) {
       p2=coll2->getParticle(i2);
       r2=coll2->getParticleRef(i2);

       double Dphi(abs(p1.phi()-p2.phi()));
       if (Dphi>M_PI) Dphi=2.0*M_PI-Dphi;

       double Deta(abs(p1.eta()-p2.eta()));

       p.setP4(Particle::LorentzVector(p1.px()+p2.px(),p1.py()+p2.py(),p1.pz()+p2.pz(),p1.energy()+p2.energy()));
       double Minv(abs(p.mass()));

       if ( ( (!cutdphi_) || (min_Dphi_ <= Dphi) && (Dphi <= max_Dphi_) ) &&
            ( (!cutdeta_) || (min_Deta_ <= Deta) && (Deta <= max_Deta_) ) &&
            ( (!cutminv_) || (min_Minv_ <= Minv) && (Minv <= max_Minv_) ) ) {
	 n++;
         filterobject->putParticle(r1);
         filterobject->putParticle(r2);
       }

     }
   }

   // filter decision
   const bool accept(n>=min_N_);

   // put filter object into the Event
   iEvent.put(filterobject);

   return accept;
}

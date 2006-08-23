/** \class HLTDoublet
 *
 * See header file for documentation
 *
 *  $Date: 2006/08/14 16:29:12 $
 *  $Revision: 1.14 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTexample/interface/HLTDoublet.h"

#include "FWCore/Framework/interface/Handle.h"

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
  Min_Dphi_ (iConfig.getParameter<double>("MinDphi")),
  Max_Dphi_ (iConfig.getParameter<double>("MaxDphi")),
  Min_Deta_ (iConfig.getParameter<double>("MinDeta")),
  Max_Deta_ (iConfig.getParameter<double>("MaxDeta")),
  Min_Minv_ (iConfig.getParameter<double>("MinMinv")),
  Max_Minv_ (iConfig.getParameter<double>("MaxMinv")),
  Min_N_    (iConfig.getParameter<int>("MinN"))
{
   // same collections to be compared?
   same = (inputTag1_.encode()==inputTag2_.encode());

   cutdphi = (Min_Dphi_ <= Max_Dphi_); // cut active?
   cutdeta = (Min_Deta_ <= Max_Deta_); // cut active?
   cutminv = (Min_Minv_ <= Max_Minv_); // cut active?

   LogDebug("") << "InputTags and cuts : " 
		<< inputTag1_.encode() << " " << inputTag2_.encode()
		<< " Dphi [" << Min_Dphi_ << " " << Max_Dphi_ << "]"
                << " Deta [" << Min_Deta_ << " " << Max_Deta_ << "]"
                << " Minv [" << Min_Minv_ << " " << Max_Minv_ << "]"
                << " MinN =" << Min_N_
		<< " same/dphi/deta/minv " << same << cutdphi << cutdeta << cutminv;

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
   ParticleKinematics p1,p2,p;
   for (unsigned int i1=0; i1!=n1; i1++) {
     p1=coll1->getParticle(i1);
     r1=coll1->getParticleRef(i1);
     unsigned int I(0);
     if (same) {I=i1+1;}
     for (unsigned int i2=I; i2!=n2; i2++) {
       p2=coll2->getParticle(i2);
       r2=coll2->getParticleRef(i2);

       double Dphi_(abs(p1.phi()-p2.phi()));
       if (Dphi_>M_PI) Dphi_=2.0*M_PI-Dphi_;

       double Deta_(abs(p1.eta()-p2.eta()));
       p=ParticleKinematics(math::XYZTLorentzVector(p1.px()+p2.px(),p1.py()+p2.py(),p1.pz()+p2.pz(),p1.energy()+p2.energy()));

       double Minv_(abs(p.mass()));

       if ( ( (!cutdphi) || (Min_Dphi_ <= Dphi_) && (Dphi_ <= Max_Dphi_) ) &&
            ( (!cutdeta) || (Min_Deta_ <= Deta_) && (Deta_ <= Max_Deta_) ) &&
            ( (!cutminv) || (Min_Minv_ <= Minv_) && (Minv_ <= Max_Minv_) ) ) {
	 n++;
         filterobject->putParticle(r1);
         filterobject->putParticle(r2);
       }

     }
   }

   // filter decision
   const bool accept(n>=Min_N_);

   // put filter object into the Event
   iEvent.put(filterobject);

   return accept;
}

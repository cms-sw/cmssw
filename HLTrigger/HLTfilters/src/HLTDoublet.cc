/** \class HLTDoublet
 *
 * See header file for documentation
 *
 *  $Date: 2008/04/22 08:00:54 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTfilters/interface/HLTDoublet.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include<cmath>

//
// constructors and destructor
//
template<typename T1, int Tid1, typename T2, int Tid2>
HLTDoublet<T1,Tid1,T2,Tid2>::HLTDoublet(const edm::ParameterSet& iConfig) :
  inputTag1_(iConfig.template getParameter<edm::InputTag>("inputTag1")),
  inputTag2_(iConfig.template getParameter<edm::InputTag>("inputTag2")),
  saveTags_ (iConfig.template getUntrackedParameter<bool>("saveTags",false)),
  min_Dphi_ (iConfig.template getParameter<double>("MinDphi")),
  max_Dphi_ (iConfig.template getParameter<double>("MaxDphi")),
  min_Deta_ (iConfig.template getParameter<double>("MinDeta")),
  max_Deta_ (iConfig.template getParameter<double>("MaxDeta")),
  min_Minv_ (iConfig.template getParameter<double>("MinMinv")),
  max_Minv_ (iConfig.template getParameter<double>("MaxMinv")),
  min_N_    (iConfig.template getParameter<int>("MinN")),
  coll1_(),
  coll2_()
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
   produces<trigger::TriggerFilterObjectWithRefs>();
}

template<typename T1, int Tid1, typename T2, int Tid2>
HLTDoublet<T1,Tid1,T2,Tid2>::~HLTDoublet()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template<typename T1, int Tid1, typename T2, int Tid2>
bool
HLTDoublet<T1,Tid1,T2,Tid2>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // All HLT filters must create and fill an HLT filter object,
   // recording any reconstructed physics objects satisfying (or not)
   // this HLT filter, and place it in the Event.

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterobject (new TriggerFilterObjectWithRefs(path(),module()));
   if (saveTags_) {
     filterobject->addCollectionTag(inputTag1_);
     filterobject->addCollectionTag(inputTag2_);
   }
   bool accept(false);

   // get hold of pre-filtered object collections
   T1Ref r1;
   T2Ref r2;
   Handle<TriggerFilterObjectWithRefs> coll1,coll2;
   if (iEvent.getByLabel (inputTag1_,coll1) && iEvent.getByLabel (inputTag2_,coll2)) {
     int n(0);
     coll1_.clear();
     coll1->getObjects(Tid1,coll1_);
     const size_type n1(coll1_.size());
     coll2_.clear();
     coll2->getObjects(Tid2,coll2_);
     const size_type n2(coll2_.size());

     Particle p1,p2,p;
     for (unsigned int i1=0; i1!=n1; i1++) {
       r1=coll1_[i1];
       p1=*r1;
       unsigned int I(0);
       if (same_) {I=i1+1;}
       for (unsigned int i2=I; i2!=n2; i2++) {
	 r2=coll2_[i2];
	 p2=*r2;

	 double Dphi(abs(p1.phi()-p2.phi()));
	 if (Dphi>M_PI) Dphi=2.0*M_PI-Dphi;
	 
	 double Deta(abs(p1.eta()-p2.eta()));
	 
	 p.setP4(Particle::LorentzVector(p1.px()+p2.px(),p1.py()+p2.py(),p1.pz()+p2.pz(),p1.energy()+p2.energy()));
	 double Minv(abs(p.mass()));
	 
	 if ( ( (!cutdphi_) || (min_Dphi_ <= Dphi) && (Dphi <= max_Dphi_) ) &&
	      ( (!cutdeta_) || (min_Deta_ <= Deta) && (Deta <= max_Deta_) ) &&
	      ( (!cutminv_) || (min_Minv_ <= Minv) && (Minv <= max_Minv_) ) ) {
	   n++;
	   filterobject->addObject(Tid1,r1);
	   filterobject->addObject(Tid2,r2);
	 }
	 
       }
     }
     // filter decision
     accept = accept || (n>=min_N_);
     iEvent.put(filterobject);
   }

   return accept;
}

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/JetMET/interface/HLTHemiDPhiFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TVector3.h"
#include "TLorentzVector.h"
//
// constructors and destructor
//
HLTHemiDPhiFilter::HLTHemiDPhiFilter(const edm::ParameterSet& iConfig) :
  inputTag_    (iConfig.getParameter<edm::InputTag>("inputTag")),
  min_dphi_      (iConfig.getParameter<double>       ("minDPhi"   )),
  accept_NJ_    (iConfig.getParameter<bool>       ("acceptNJ"   ))

{
   LogDebug("") << "Inputs/minDphi/acceptNJ : "
		<< inputTag_.encode() << " "	
		<< min_dphi_ << " "
		<< accept_NJ_ << ".";

   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTHemiDPhiFilter::~HLTHemiDPhiFilter()
{
}

void
HLTHemiDPhiFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("inputTag",edm::InputTag("hltRHemisphere"));
  desc.add<double>("minDPhi",2.9415);
  desc.add<bool>("acceptNJ",true);
  descriptions.add("hltHemiDPhiFilter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool 
HLTHemiDPhiFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace std;
   using namespace edm;
   using namespace reco;
   using namespace trigger;

   // The filter object
   auto_ptr<TriggerFilterObjectWithRefs>
     filterobject (new TriggerFilterObjectWithRefs(path(),module()));

   // get hold of collection of objects
   Handle< vector<math::XYZTLorentzVector> > hemispheres;
   iEvent.getByLabel (inputTag_,hemispheres);

   iEvent.put(filterobject);

   // check the the input collections are available
   if (not hemispheres.isValid())
     return false;

   if(hemispheres->size() ==0){  // the Hemisphere Maker will produce an empty collection of hemispheres if the number of jets in the
     return accept_NJ_;   // event is greater than the maximum number of jets
   }




  //***********************************
  // Get the set of hemisphere axes
   
   TLorentzVector j1R(hemispheres->at(0).x(),hemispheres->at(0).y(),hemispheres->at(0).z(),hemispheres->at(0).t());
   TLorentzVector j2R(hemispheres->at(1).x(),hemispheres->at(1).y(),hemispheres->at(1).z(),hemispheres->at(1).t());

   // compute the dPhi between them
  double dphi = 50.;
  dphi = deltaPhi(j1R.Phi(),j2R.Phi()); 
  
  // Dphi requirement  

  if(dphi<=min_dphi_) return true;

   // filter decision
   return false;
}


double HLTHemiDPhiFilter::deltaPhi(double v1, double v2)
{ // Computes the correctly normalized phi difference
  // v1, v2 = phi of object 1 and 2
 double diff = fabs(v2 - v1);
 double corr = 2*acos(-1.) - diff;
 if (diff < acos(-1.)){ return diff;} else { return corr;} 
 
}

DEFINE_FWK_MODULE(HLTHemiDPhiFilter);


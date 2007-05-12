/** \class HLTPMMassFilter
 *
 * Original Author: Jeremy Werner
 * Institution: Princeton University, USA
 * Contact: Jeremy.Werner@cern.ch 
 * Date: February 21, 2007
 */

#include "HLTrigger/Egamma/interface/HLTPMMassFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"


//
// constructors and destructor
//
HLTPMMassFilter::HLTPMMassFilter(const edm::ParameterSet& iConfig)
{

  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  lowerMassCut_     = iConfig.getParameter<double> ("lowerMassCut");
  upperMassCut_     = iConfig.getParameter<double> ("upperMassCut");
  nZcandcut_           = iConfig.getParameter<int> ("nZcandcut");
  
   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTPMMassFilter::~HLTPMMassFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTPMMassFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{



  using namespace std;
  using namespace edm;
  using namespace reco;

   // The filter object
  std::auto_ptr<reco::HLTFilterObjectWithRefs> filterproduct (new reco::HLTFilterObjectWithRefs(path(),module()));
  // Ref to Candidate object to be recorded in filter object
  edm::RefToBase<reco::Candidate> ref;
  
  // get hold of filtered candidates

  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  int n = 0;

  double px[66];
  double py[66];
  double pz[66];
  double energy[66];

  unsigned int size = recoecalcands->size();
  if(size>66) size=66;


  for (unsigned int i=0; i<size; i++) {
    
    ref = recoecalcands->getParticleRef(i);

    px[i]=ref.get()->px();
    py[i]=ref.get()->py();
    pz[i]=ref.get()->pz();
    energy[i]=ref.get()->energy();
  }


  for(unsigned int jj=0;jj<size;jj++){
     for(unsigned int ii=0;ii<size;ii++){
       if(jj <ii){
	 double mass = sqrt((energy[jj]+energy[ii])*(energy[jj]+energy[ii]) - ((px[jj]+px[ii])*(px[jj]+px[ii])+(py[jj]+py[ii])*(py[jj]+py[ii])+(pz[jj]+pz[ii])*(pz[jj]+pz[ii])));

	 std::cout<<"mass ="<<mass<<std::endl;

	 if(mass>= lowerMassCut_ && mass<=upperMassCut_){
	   n++;
	   ref = recoecalcands->getParticleRef(ii);
	   filterproduct->putParticle(ref);
	   ref = recoecalcands->getParticleRef(jj);
	   filterproduct->putParticle(ref);
	 }
       }
     }
  }


  // filter decision
  bool accept(n>=nZcandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}


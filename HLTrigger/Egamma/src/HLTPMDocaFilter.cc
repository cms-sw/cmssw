/** \class HLTPMDocaFilter
 * 
 *  Original Author: Jeremy Werner                          
 *  Institution: Princeton University, USA                                                                 *  Contact: Jeremy.Werner@cern.ch 
 *  Date: February 21, 2007     
 * $Id: $
 *
 */

#include "HLTrigger/Egamma/interface/HLTPMDocaFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/AssociationMap.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// constructors and destructor
//
HLTPMDocaFilter::HLTPMDocaFilter(const edm::ParameterSet& iConfig)
{
  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  docaDiffPerpCutHigh_     = iConfig.getParameter<double> ("docaDiffPerpCutHigh");
  docaDiffPerpCutLow_     = iConfig.getParameter<double> ("docaDiffPerpCutLow");
  nZcandcut_           = iConfig.getParameter<int> ("nZcandcut");

   //register your products
   produces<reco::HLTFilterObjectWithRefs>();
}

HLTPMDocaFilter::~HLTPMDocaFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTPMDocaFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
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

  double vx[66];
  double vy[66];
  double vz[66];

  unsigned int size = recoecalcands->size();
  if(size>66) size=66;

  for (unsigned int i=0; i< size; i++) {
    
    ref = recoecalcands->getParticleRef(i);

    vx[i]=ref.get()->vx();
    vy[i]=ref.get()->vy();
    vz[i]=ref.get()->vz();
    
  }

  for(unsigned int jj=0;jj<size;jj++){
     for(unsigned int ii=0;ii<size;ii++){
       if(jj <ii){

	 double docaDiffPerp = sqrt( (vx[jj]-vx[ii])*(vx[jj]-vx[ii])+(vy[jj]-vy[ii])*(vy[jj]-vy[ii]));
	 std::cout<<"docaDiffPerp= "<<docaDiffPerp<<std::endl;
	 if((docaDiffPerp>=docaDiffPerpCutLow_) && (docaDiffPerp<= docaDiffPerpCutHigh_)){
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


/** \class HLTPMDocaFilter
 * 
 *  Original Author: Jeremy Werner                          
 *  Institution: Princeton University, USA                                                                                                               *  Contact: Jeremy.Werner@cern.ch 
 *  Date: February 21, 2007     
 * 
 *
 */

#include "HLTrigger/Egamma/interface/HLTPMDocaFilter.h"

#include "FWCore/Framework/interface/Handle.h"

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

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/TrackReco/interface/Track.h"

//
// constructors and destructor
//
HLTPMDocaFilter::HLTPMDocaFilter(const edm::ParameterSet& iConfig)
{
  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  elecTag_     = iConfig.getParameter< edm::InputTag > ("elecTag");
  docaDiffPerpCutHigh_     = iConfig.getParameter<double> ("docaDiffPerpCutHigh");
  docaDiffPerpCutLow_     = iConfig.getParameter<double> ("docaDiffPerpCutLow");
  ncandcut_           = iConfig.getParameter<int> ("ncandcut");

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
  edm::RefToBase<reco::Candidate> candref;
  
  // get hold of filtered candidates

  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  //get hold of electron candidates
  Handle<ElectronCollection> electrons;
  iEvent.getByLabel(elecTag_,electrons);

  int n = 0;

  double vx[6];
  double vy[6];
  double vz[6];

  unsigned int numdone=0;

  ElectronCollection::const_iterator aelec(electrons->begin());
  ElectronCollection::const_iterator oelec(electrons->end());
  ElectronCollection::const_iterator ielec;
  for (ielec=aelec; ielec!=oelec; ielec++) {

    if(numdone<electrons->size()){

      reco::TrackRef trackref = ielec->track();


      vx[numdone]=(*trackref).vx();
      vy[numdone]=(*trackref).vy();
      vz[numdone]=(*trackref).vz();

      numdone++;
    }
  }

  int evtPassed=0;
  for(unsigned int jj=0;jj<numdone;jj++){
     for(unsigned int ii=0;ii<numdone;ii++){
       if(jj <ii){

	 double docaDiffPerp = sqrt( (vx[jj]-vx[ii])*(vx[jj]-vx[ii])+(vy[jj]-vy[ii])*(vy[jj]-vy[ii]));
	 if((docaDiffPerp>=docaDiffPerpCutLow_) && (docaDiffPerp<= docaDiffPerpCutHigh_)){
	   n++;
	   evtPassed++;
	 }
       }
     }
  }

  // filter decision
  bool accept(n>=ncandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}


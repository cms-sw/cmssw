/** \class HLTPMMassFilter
 *
 * Original Author: Jeremy Werner
 * Institution: Princeton University, USA
 * Contact: Jeremy.Werner@cern.ch 
 * Date: February 21, 2007
 */

#include "HLTrigger/Egamma/interface/HLTPMMassFilter.h"

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


//
// constructors and destructor
//
HLTPMMassFilter::HLTPMMassFilter(const edm::ParameterSet& iConfig)
{



  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  elecTag_     = iConfig.getParameter< edm::InputTag > ("elecTag");
  lowerMassCut_     = iConfig.getParameter<double> ("lowerMassCut");
  upperMassCut_     = iConfig.getParameter<double> ("upperMassCut");
  ncandcut_           = iConfig.getParameter<int> ("ncandcut");
  
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
  edm::RefToBase<reco::Candidate> candref;
  
  // get hold of filtered candidates

  edm::Handle<reco::HLTFilterObjectWithRefs> recoecalcands;
  iEvent.getByLabel (candTag_,recoecalcands);
  
  Handle<ElectronCollection> electrons;
  iEvent.getByLabel(elecTag_,electrons);

  int n = 0;

  double px[6];
  double py[6];
  double pz[6];
  double energy[6];

  unsigned int numdone=0;
  int got2pm=0;
  if(electrons->size()>1){
    
    got2pm++;
  }

  ElectronCollection::const_iterator aelec(electrons->begin());
  ElectronCollection::const_iterator oelec(electrons->end());
  ElectronCollection::const_iterator ielec;
  for (ielec=aelec; ielec!=oelec; ielec++) {

    if(numdone<electrons->size()){

      px[numdone]=ielec->px();
      py[numdone]=ielec->py();
      pz[numdone]=ielec->pz();
      energy[numdone]=ielec->energy();
      numdone++;
    }
  }

  int evtPassed=0;
  for(unsigned int jj=0;jj<numdone;jj++){
     for(unsigned int ii=0;ii<numdone;ii++){
       if(jj <ii){
	 double mass = sqrt((energy[jj]+energy[ii])*(energy[jj]+energy[ii]) - ((px[jj]+px[ii])*(px[jj]+px[ii])+(py[jj]+py[ii])*(py[jj]+py[ii])+(pz[jj]+pz[ii])*(pz[jj]+pz[ii])));

	 if(mass>= lowerMassCut_ && mass<=upperMassCut_){
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


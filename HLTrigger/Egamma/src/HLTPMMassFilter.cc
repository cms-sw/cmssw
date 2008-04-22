/** \class HLTPMMassFilter
 *
 * Original Author: Jeremy Werner
 * Institution: Princeton University, USA
 * Contact: Jeremy.Werner@cern.ch 
 * Date: February 21, 2007
 */

#include "HLTrigger/Egamma/interface/HLTPMMassFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

//
// constructors and destructor
//
HLTPMMassFilter::HLTPMMassFilter(const edm::ParameterSet& iConfig)
{

  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  lowerMassCut_     = iConfig.getParameter<double> ("lowerMassCut");
  upperMassCut_     = iConfig.getParameter<double> ("upperMassCut");
  nZcandcut_           = iConfig.getParameter<int> ("nZcandcut");

  store_ = iConfig.getUntrackedParameter<bool> ("SaveTag",false) ;
  relaxed_ = iConfig.getUntrackedParameter<bool> ("relaxed",true) ;
  L1IsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1IsoCand"); 
  L1NonIsoCollTag_= iConfig.getParameter< edm::InputTag > ("L1NonIsoCand"); 

   //register your products
    produces<trigger::TriggerFilterObjectWithRefs>();
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
  using namespace trigger;
  std::auto_ptr<trigger::TriggerFilterObjectWithRefs> filterproduct (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if( store_ ){filterproduct->addCollectionTag(L1IsoCollTag_);}
  if( store_ && relaxed_){filterproduct->addCollectionTag(L1NonIsoCollTag_);}
  // Ref to Candidate object to be recorded in filter object

  edm::Ref<reco::ElectronCollection> ref;


  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;

  iEvent.getByLabel (candTag_,PrevFilterOutput);

  std::vector<edm::Ref<reco::ElectronCollection> > electrons;
  PrevFilterOutput->getObjects(TriggerElectron, electrons);
  int n = 0;

  double px[66];
  double py[66];
  double pz[66];
  double energy[66];

  unsigned int size = electrons.size();
  if(size>66) size=66;


  for (unsigned int i=0; i<size; i++) {
    
    ref = electrons[i];

    px[i]=ref->px();
    py[i]=ref->py();
    pz[i]=ref->pz();
    energy[i]=ref->energy();
  }
  

  for(unsigned int jj=0;jj<size;jj++){
     for(unsigned int ii=jj+1;ii<size;ii++){
       	 double mass = sqrt((energy[jj]+energy[ii])*(energy[jj]+energy[ii]) - ((px[jj]+px[ii])*(px[jj]+px[ii])+(py[jj]+py[ii])*(py[jj]+py[ii])+(pz[jj]+pz[ii])*(pz[jj]+pz[ii])));
	 //std::cout<<"mass ="<<mass<<std::endl;
	 if(mass>= lowerMassCut_ && mass<=upperMassCut_){
	   n++;
	   ref = electrons[ii];
           filterproduct->addObject(TriggerElectron, ref);
           ref = electrons[jj];
           filterproduct->addObject(TriggerElectron, ref);
	 }
     }
  }

  

  // filter decision
  bool accept(n>=nZcandcut_);
  
  // put filter object into the Event
  iEvent.put(filterproduct);
  
  return accept;
}


/** \class HLTPMDocaFilter
 * 
 *  Original Author: Jeremy Werner                          
 *  Institution: Princeton University, USA                                                                 *  Contact: Jeremy.Werner@cern.ch 
 *  Date: February 21, 2007     
 * $Id: HLTPMDocaFilter.cc,v 1.9 2012/01/21 14:56:58 fwyzard Exp $
 *
 */

#include "HLTrigger/Egamma/interface/HLTPMDocaFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

//
// constructors and destructor
//
HLTPMDocaFilter::HLTPMDocaFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig) 
{
  candTag_            = iConfig.getParameter< edm::InputTag > ("candTag");
  docaDiffPerpCutHigh_     = iConfig.getParameter<double> ("docaDiffPerpCutHigh");
  docaDiffPerpCutLow_     = iConfig.getParameter<double> ("docaDiffPerpCutLow");
  nZcandcut_           = iConfig.getParameter<int> ("nZcandcut");
}

HLTPMDocaFilter::~HLTPMDocaFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTPMDocaFilter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // Ref to Candidate object to be recorded in filter object
  edm::Ref<reco::ElectronCollection> ref;

  edm::Handle<trigger::TriggerFilterObjectWithRefs> PrevFilterOutput;
  iEvent.getByLabel (candTag_,PrevFilterOutput);
  
  std::vector<edm::Ref<reco::ElectronCollection> > electrons;
  PrevFilterOutput->getObjects(TriggerElectron, electrons);
  
 
  int n = 0;

  unsigned int size = electrons.size();
  std::vector<double> vx(size);
  std::vector<double> vy(size);

  for (unsigned int i=0; i< size; i++) {
    ref = electrons[i];
    vx[i]=ref->vx();
    vy[i]=ref->vy();
  }

  for(unsigned int jj=0;jj<size;jj++){
     for(unsigned int ii=jj+1;ii<size;ii++){
	 double docaDiffPerp = sqrt( (vx[jj]-vx[ii])*(vx[jj]-vx[ii])+(vy[jj]-vy[ii])*(vy[jj]-vy[ii]));
	 // std::cout<<"docaDiffPerp= "<<docaDiffPerp<<std::endl;
	 if((docaDiffPerp>=docaDiffPerpCutLow_) && (docaDiffPerp<= docaDiffPerpCutHigh_)){
	   n++;
	   ref = electrons[ii];
	   filterproduct.addObject(TriggerElectron, ref);
	   ref = electrons[jj];
	   filterproduct.addObject(TriggerElectron, ref);

	 }
     }
  }
  

  // filter decision
  bool accept(n>=nZcandcut_);
  
  return accept;
}


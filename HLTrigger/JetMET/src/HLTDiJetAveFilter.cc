/** \class HLTDiJetAveFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include "HLTrigger/JetMET/interface/HLTDiJetAveFilter.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Math/interface/deltaPhi.h"

//
// constructors and destructor
//
HLTDiJetAveFilter::HLTDiJetAveFilter(const edm::ParameterSet& iConfig)
{
   inputJetTag_ = iConfig.getParameter< edm::InputTag > ("inputJetTag");
   saveTag_     = iConfig.getUntrackedParameter<bool>("saveTag",false);
   minEtAve_    = iConfig.getParameter<double> ("minEtAve"); 
   minEtJet3_   = iConfig.getParameter<double> ("minEtJet3"); 
   minDphi_     = iConfig.getParameter<double> ("minDphi"); 
   //register your products
   produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTDiJetAveFilter::~HLTDiJetAveFilter(){}


// ------------ method called to produce the data  ------------
bool
HLTDiJetAveFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;
  // The filter object
  auto_ptr<trigger::TriggerFilterObjectWithRefs> 
    filterobject (new trigger::TriggerFilterObjectWithRefs(path(),module()));
  if (saveTag_) filterobject->addCollectionTag(inputJetTag_);

  Handle<CaloJetCollection> recocalojets;
  iEvent.getByLabel(inputJetTag_,recocalojets);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(recocalojets->size() > 1){
    // events with two or more jets

    double etjet1=0., etjet2=0.,etjet3=0.;
    double phijet1=0.,phijet2=0;
    int countjets =0;

    int nmax=1;
    if (recocalojets->size() > 2) nmax=2;

    CaloJetRef JetRef1,JetRef2;

    for (CaloJetCollection::const_iterator recocalojet = recocalojets->begin(); 
	 recocalojet<=(recocalojets->begin()+nmax); recocalojet++) {
      
      if(countjets==0) {
	etjet1 = recocalojet->et();
	phijet1 = recocalojet->phi();
	JetRef1 = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
      }
      if(countjets==1) {
	etjet2 = recocalojet->et();
	phijet2 = recocalojet->phi();
	JetRef2 = CaloJetRef(recocalojets,distance(recocalojets->begin(),recocalojet));
      }
      if(countjets==2) {
	etjet3 = recocalojet->et();
      }
      countjets++;
    }
    
    double EtAve=(etjet1 + etjet2) / 2.;
    double Dphi = fabs(deltaPhi(phijet1,phijet2));

    if( EtAve>minEtAve_  && etjet3<minEtJet3_ && Dphi>minDphi_){
      filterobject->addObject(TriggerJet,JetRef1);
      filterobject->addObject(TriggerJet,JetRef2);
      n++;
    }
    
  } // events with two or more jets
  
  
  
  // filter decision
  bool accept(n>=1);
  
  // put filter object into the Event
  iEvent.put(filterobject);
  
  return accept;
}

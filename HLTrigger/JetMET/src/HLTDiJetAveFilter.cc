/** \class HLTDiJetAveFilter
 *
 *
 *  \author Leonard Apanasevich
 *
 */

#include "HLTrigger/JetMET/interface/HLTDiJetAveFilter.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

//
// constructors and destructor
//
template<typename T>
HLTDiJetAveFilter<T>::HLTDiJetAveFilter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  inputJetTag_ (iConfig.template getParameter< edm::InputTag > ("inputJetTag")),
  minPtAve_    (iConfig.template getParameter<double> ("minPtAve")),
  minPtJet3_   (iConfig.template getParameter<double> ("minPtJet3")),
  minDphi_     (iConfig.template getParameter<double> ("minDphi")),
  triggerType_ (iConfig.template getParameter<int> ("triggerType"))
{
  m_theJetToken = consumes<std::vector<T>>(inputJetTag_);
  LogDebug("") << "HLTDiJetAveFilter: Input/minPtAve/minPtJet3/minDphi/triggerType : "
	       << inputJetTag_.encode() << " "
	       << minPtAve_ << " "
	       << minPtJet3_ << " "
	       << minDphi_ << " "
	       << triggerType_;
}

template<typename T>
HLTDiJetAveFilter<T>::~HLTDiJetAveFilter(){}

template<typename T>
void
HLTDiJetAveFilter<T>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("inputJetTag",edm::InputTag("hltIterativeCone5CaloJets"));
  desc.add<double>("minPtAve",100.0);
  desc.add<double>("minPtJet3",99999.0);
  desc.add<double>("minDphi",-1.0);
  desc.add<int>("triggerType",trigger::TriggerJet);
  descriptions.add(defaultModuleLabel<HLTDiJetAveFilter<T>>(), desc);
}

// ------------ method called to produce the data  ------------
template<typename T>
bool
HLTDiJetAveFilter<T>::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  typedef vector<T> TCollection;
  typedef Ref<TCollection> TRef;

  // The filter object
  if (saveTags()) filterproduct.addCollectionTag(inputJetTag_);

  // get hold of collection of objects
  Handle<TCollection> objects;
  iEvent.getByToken (m_theJetToken,objects);

  // look at all candidates,  check cuts and add to filter object
  int n(0);

  if(objects->size() > 1){
    // events with two or more jets

    double ptjet1=0., ptjet2=0.,ptjet3=0.;
    double phijet1=0.,phijet2=0;
    int countjets =0;

    int nmax=1;
    if (objects->size() > 2) nmax=2;

    TRef JetRef1,JetRef2;

    typename TCollection::const_iterator i ( objects->begin() );
    for (; i<=(objects->begin()+nmax); i++) {
      if(countjets==0) {
	ptjet1 = i->pt();
	phijet1 = i->phi();
	JetRef1 = TRef(objects,distance(objects->begin(),i));
      }
      if(countjets==1) {
	ptjet2 = i->pt();
	phijet2 = i->phi();
	JetRef2 = TRef(objects,distance(objects->begin(),i));
      }
      if(countjets==2) {
	ptjet3 = i->pt();
      }
      ++countjets;
    }

    double PtAve=(ptjet1 + ptjet2) / 2.;
    double Dphi = std::abs(deltaPhi(phijet1,phijet2));

    if( PtAve>minPtAve_ && ptjet3<minPtJet3_ && Dphi>minDphi_){
      filterproduct.addObject(triggerType_,JetRef1);
      filterproduct.addObject(triggerType_,JetRef2);
      ++n;
    }

  } // events with two or more jets



  // filter decision
  bool accept(n>=1);

  return accept;
}

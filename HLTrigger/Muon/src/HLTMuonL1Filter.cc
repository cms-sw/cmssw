/** \class HLTMuonL1Filter
 *
 * See header file for documentation
 *
 *  \author J. Alcaraz
 *
 */

#include "HLTrigger/Muon/interface/HLTMuonL1Filter.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "FWCore/Utilities/interface/EDMException.h"

//
// constructors and destructor
//
HLTMuonL1Filter::HLTMuonL1Filter(const edm::ParameterSet& iConfig) :
  candTag_( iConfig.getParameter<edm::InputTag>("CandTag") ),
  previousCandTag_( iConfig.getParameter<edm::InputTag>("PreviousCandTag") ),
  maxEta_( iConfig.getParameter<double>("MaxEta") ),
  minPt_( iConfig.getParameter<double>("MinPt") ),
  minN_( iConfig.getParameter<int>("MinN") ),
  saveTag_( iConfig.getUntrackedParameter<bool>("SaveTag",false) ) 
{
  using namespace std;

  //set the quality bit mask
  qualityBitMask_ = 0;
  vector<int> selectQualities = iConfig.getParameter<vector<int> >("SelectQualities");
  for(size_t i=0; i<selectQualities.size(); i++){
    if(selectQualities[i] > 7){
      throw edm::Exception(edm::errors::Configuration) << "QualityBits must be smaller than 8!";
    }
    qualityBitMask_ |= 1<<selectQualities[i];
  }
 
  // dump parameters for debugging
  if(edm::isDebugEnabled()){
    ostringstream ss;
    ss<<"Constructed with parameters:"<<endl;
    ss<<"    CandTag = "<<candTag_.encode()<<endl;
    ss<<"    PreviousCandTag = "<<previousCandTag_.encode()<<endl;
    ss<<"    MaxEta = "<<maxEta_<<endl;
    ss<<"    MinPt = "<<minPt_<<endl;
    ss<<"    SelectQualities =";
    for(size_t i=0; i<8; i++){
      if((qualityBitMask_>>i) % 2) ss<<" "<<i;
    }
    ss<<endl;
    ss<<"    MinN = "<<minN_<<endl;
    ss<<"    SaveTag = "<<saveTag_;
    LogDebug("HLTMuonL1Filter")<<ss.str();
  }

  //register your products
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonL1Filter::~HLTMuonL1Filter()
{
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool
HLTMuonL1Filter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace std;
  using namespace edm;
  using namespace trigger;
  using namespace l1extra;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  auto_ptr<TriggerFilterObjectWithRefs> filterproduct(new TriggerFilterObjectWithRefs(path(), module()));

  // get hold of all muons
  Handle<L1MuonParticleCollection> allMuons;
  iEvent.getByLabel(candTag_, allMuons);

  // get hold of muons that fired the previous level
  Handle<TriggerFilterObjectWithRefs> previousLevelCands;
  iEvent.getByLabel(previousCandTag_, previousLevelCands);
  vector<L1MuonParticleRef> prevMuons;
  previousLevelCands->getObjects(TriggerL1Mu, prevMuons);
   
  // look at all muon candidates, check cuts and add to filter object
  int n = 0;
  for(size_t i = 0; i < allMuons->size(); i++){
    L1MuonParticleRef muon(allMuons, i);

    //check if triggered by the previous level
    if(find(prevMuons.begin(), prevMuons.end(), muon) == prevMuons.end()) continue;

    //check maxEta cut
    if(fabs(muon->eta()) > maxEta_) continue;

    //check pT cut
    if(muon->pt() < minPt_) continue;

    //check quality cut
    if(qualityBitMask_){
      int quality = muon->gmtMuonCand().empty() ? 0 : (1 << muon->gmtMuonCand().quality());
      if((quality & qualityBitMask_) == 0) continue;
    }

    //we have a good candidate
    n++;
    filterproduct->addObject(TriggerL1Mu,muon);
  }

  if(saveTag_) filterproduct->addCollectionTag(candTag_);

  // filter decision
  const bool accept(n >= minN_);

  // dump event for debugging
  if(edm::isDebugEnabled()){
    ostringstream ss;
    ss.precision(2);
    ss<<"L1mu#"<<'\t'<<"q*pt"<<'\t'<<'\t'<<"eta"<<'\t'<<"phi"<<'\t'<<"quality"<<'\t'<<"isPrev"<<'\t'<<"isFired"<<endl;
    ss<<"---------------------------------------------------------------"<<endl;

    vector<L1MuonParticleRef> firedMuons;
    filterproduct->getObjects(TriggerL1Mu, firedMuons);
    for(size_t i=0; i<allMuons->size(); i++){
      L1MuonParticleRef mu(allMuons, i);
      int quality = mu->gmtMuonCand().empty() ? 0 : mu->gmtMuonCand().quality();
      bool isPrev = find(prevMuons.begin(), prevMuons.end(), mu) != prevMuons.end();
      bool isFired = find(firedMuons.begin(), firedMuons.end(), mu) != firedMuons.end();
      ss<<i<<'\t'<<scientific<<mu->charge()*mu->pt()<<'\t'<<fixed<<mu->eta()<<'\t'<<mu->phi()<<'\t'<<quality<<'\t'<<isPrev<<'\t'<<isFired<<endl;
    }
    ss<<"---------------------------------------------------------------"<<endl;
    LogDebug("HLTMuonL1Filter")<<ss.str()<<"Decision of filter is "<<accept<<", number of muons passing = "<<filterproduct->l1muonSize();
  }

  // put filter object into the Event
  iEvent.put(filterproduct);

  return accept;
}


#include "HLTrigger/Muon/interface/HLTMuonL1RegionalFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "FWCore/Utilities/interface/EDMException.h"

HLTMuonL1RegionalFilter::HLTMuonL1RegionalFilter(const edm::ParameterSet& iConfig):
  candTag_( iConfig.getParameter<edm::InputTag>("CandTag") ),
  previousCandTag_( iConfig.getParameter<edm::InputTag>("PreviousCandTag") ),
  minN_( iConfig.getParameter<int>("MinN") ),
  saveTags_( iConfig.getParameter<bool>("saveTags") ) 
{
  using namespace std;
  using namespace edm;

  // read in the eta-range dependent parameters  
  const vector<ParameterSet> cuts = iConfig.getParameter<vector<ParameterSet> >("Cuts");
  size_t ranges = cuts.size();
  if(ranges==0){
    throw edm::Exception(errors::Configuration) << "Please provide at least one PSet in the Cuts VPSet!";
  }
  etaBoundaries_.reserve(ranges+1);
  minPts_.reserve(ranges);
  qualityBitMasks_.reserve(ranges);
  for(size_t i=0; i<ranges; i++){
    //set the eta range
    vector<double> etaRange = cuts[i].getParameter<vector<double> >("EtaRange");
    if(etaRange.size() != 2 || etaRange[0] >= etaRange[1]){
      throw edm::Exception(errors::Configuration) << "EtaRange must have two non-equal values in increasing order!";
    }
    if(i==0){
      etaBoundaries_.push_back( etaRange[0] );
    }else if(etaBoundaries_[i] != etaRange[0]){
      throw edm::Exception(errors::Configuration) << "EtaRanges must be disjoint without gaps and listed in increasing eta order!";
    }
    etaBoundaries_.push_back( etaRange[1] );

    //set the minPt
    minPts_.push_back( cuts[i].getParameter<double>("MinPt") );

    //set the quality bit masks
    qualityBitMasks_.push_back( 0 );
    vector<unsigned int> qualities = cuts[i].getParameter<vector<unsigned int> >("QualityBits");
    for(size_t j=0; j<qualities.size(); j++){
      if(7U < qualities[j]){ // qualities[j] >= 0, since qualities[j] is unsigned
        throw edm::Exception(errors::Configuration) << "QualityBits must be between 0 and 7 !";
      }
      qualityBitMasks_[i] |= 1 << qualities[j];
    }
  }

  // dump parameters for debugging
  if(edm::isDebugEnabled()){
    ostringstream ss;
    ss<<"Constructed with parameters:"<<endl;
    ss<<"    CandTag = "<<candTag_.encode()<<endl;
    ss<<"    PreviousCandTag = "<<previousCandTag_.encode()<<endl;
    ss<<"    EtaBoundaries = \t"<<etaBoundaries_[0];
    for(size_t i=1; i<etaBoundaries_.size(); i++){
      ss<<'\t'<<etaBoundaries_[i];
    }
    ss<<endl;
    ss<<"    MinPts =        \t    "<<minPts_[0];
    for(size_t i=1; i<minPts_.size(); i++){
      ss<<"\t    "<<minPts_[i];
    }
    ss<<endl;
    ss<<"    QualityBitMasks =  \t    "<<qualityBitMasks_[0];
    for(size_t i=1; i<qualityBitMasks_.size(); i++){
      ss<<"\t    "<<qualityBitMasks_[i];
    }
    ss<<endl;
    ss<<"    MinN = "<<minN_<<endl;
    ss<<"    saveTags= "<<saveTags_;
    LogDebug("HLTMuonL1RegionalFilter")<<ss.str();
  }

  //register the product
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonL1RegionalFilter::~HLTMuonL1RegionalFilter(){
}

bool HLTMuonL1RegionalFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
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
   
  // look at all mucands,  check cuts and add to filter object
  int n = 0;
  for (size_t i = 0; i < allMuons->size(); i++) {
    L1MuonParticleRef muon(allMuons, i);

    //check if triggered by the previous level
    if( find(prevMuons.begin(), prevMuons.end(), muon) == prevMuons.end() ) continue;

    //find eta region, continue otherwise
    float eta   =  muon->eta();
    int region = -1;
    for(size_t r=0; r<etaBoundaries_.size()-1; r++){
      if(etaBoundaries_[r]<=eta && eta<=etaBoundaries_[r+1]){
        region = r;
        break;
      }
    }
    if(region == -1) continue;

    //check pT cut
    if(muon->pt() < minPts_[region]) continue;

    //check quality cut
    if(qualityBitMasks_[region]){
      int quality = muon->gmtMuonCand().empty() ? 0 : (1 << muon->gmtMuonCand().quality());
      if((quality & qualityBitMasks_[region]) == 0) continue;
    }

    //we have a good candidate
    filterproduct->addObject(TriggerL1Mu, muon);

    n++;
  }

  if(saveTags_) filterproduct->addCollectionTag(candTag_);

  // filter decision
  const bool accept (n >= minN_);

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
      ss<<i<<'\t'<<scientific<<mu->charge()*mu->pt()<<fixed<<'\t'<<mu->eta()<<'\t'<<mu->phi()<<'\t'<<quality<<'\t'<<isPrev<<'\t'<<isFired<<endl;
    }
    ss<<"---------------------------------------------------------------"<<endl;
    LogDebug("HLTMuonL1RegionalFilter")<<ss.str()<<"Decision of filter is "<<accept<<", number of muons passing = "<<filterproduct->l1muonSize();
  }

  // put filter object into the Event
  iEvent.put(filterproduct);

  return accept;
}


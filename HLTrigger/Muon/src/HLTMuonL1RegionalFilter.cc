#include "HLTrigger/Muon/interface/HLTMuonL1RegionalFilter.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerRefsCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace std;
using namespace edm;
using namespace trigger;
using namespace l1extra;

HLTMuonL1RegionalFilter::HLTMuonL1RegionalFilter(const edm::ParameterSet& iConfig):
  candTag( iConfig.getParameter<edm::InputTag>("CandTag") ),
  previousCandTag( iConfig.getParameter<edm::InputTag>("PreviousCandTag") ),
  minN( iConfig.getParameter<int>("MinN") ),
  saveTag( iConfig.getUntrackedParameter<bool>("SaveTag",false) ) 
{
  const vector<ParameterSet> cuts = iConfig.getParameter<vector<ParameterSet> >("Cuts");
  size_t ranges = cuts.size();
  if(ranges==0){
    throw edm::Exception(errors::Configuration) << "Please provide at least one PSet in the Cuts VPSet!";
  }
  etaBoundaries.reserve(ranges+1);
  minPts.reserve(ranges);
  qualityBitMasks.reserve(ranges);
  for(size_t i=0; i<ranges; i++){
    //set the eta range
    vector<double> etaRange = cuts[i].getParameter<vector<double> >("EtaRange");
    if(etaRange.size() != 2 || etaRange[0] >= etaRange[1]){
      throw edm::Exception(errors::Configuration) << "EtaRange must have two non-equal values in increasing order!";
    }
    if(i==0){
      etaBoundaries.push_back( etaRange[0] );
    }else if(etaBoundaries[i] != etaRange[0]){
      throw edm::Exception(errors::Configuration) << "EtaRanges must be disjoint without gaps and listed in increasing eta order!";
    }
    etaBoundaries.push_back( etaRange[1] );

    //set the minPt
    minPts.push_back( cuts[i].getParameter<double>("MinPt") );

    //set the quality bit mask
    qualityBitMasks.push_back( 0 );
    vector<uint> qualities = cuts[i].getParameter<vector<uint> >("QualityBits");
    for(size_t j=0; j<qualities.size(); j++){
      if(qualities[j] > 7){
        throw edm::Exception(errors::Configuration) << "QualityBits must smaller than 8!";
      }
      qualityBitMasks[i] |= 1 << qualities[j];
    }
    if(qualityBitMasks[i] == 0){
      qualityBitMasks[i] = 255;
    }
  }

  LogDebug("HLTMuonL1RegionalFilter")<<dumpParameters();

  //register the product
  produces<trigger::TriggerFilterObjectWithRefs>();
}

HLTMuonL1RegionalFilter::~HLTMuonL1RegionalFilter(){
}

bool HLTMuonL1RegionalFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup){
  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // The filter object
  auto_ptr<TriggerFilterObjectWithRefs> filterproduct(new TriggerFilterObjectWithRefs(path(), module()));

  // get hold of all muons
  Handle<L1MuonParticleCollection> allMuons;
  iEvent.getByLabel(candTag, allMuons);

  // get hold of muons that fired the previous level
  Handle<TriggerFilterObjectWithRefs> previousLevelCands;
  iEvent.getByLabel(previousCandTag, previousLevelCands);
  vector<L1MuonParticleRef> prevMuons;
  previousLevelCands->getObjects(TriggerL1Mu, prevMuons);
   
  // look at all mucands,  check cuts and add to filter object
  int n = 0;
  for (size_t i = 0; i < allMuons->size(); i++) {
    L1MuonParticleRef muon(allMuons, i);

    //check if triggered by the previous level
    if( find(prevMuons.begin(), prevMuons.end(), muon) == prevMuons.end() )
      continue;

    //find eta region, continue otherwise
    float eta   =  muon->eta();
    int region = -1;
    for(size_t r=0; r<etaBoundaries.size()-1; r++){
      if(etaBoundaries[r]<=eta && eta<=etaBoundaries[r+1]){
        region = r;
        break;
      }
    }
    if(region == -1)
      continue;

    //check pT cut
    if(muon->pt() < minPts[region])
      continue;

    //check quality cut
    int quality = muon->gmtMuonCand().empty() ? 0 : (1 << muon->gmtMuonCand().quality());
    if((quality & qualityBitMasks[region]) == 0)
      continue;

    //we have a good candidate
    n++;
    filterproduct->addObject(TriggerL1Mu, muon);
  }

  if(saveTag)
    filterproduct->addCollectionTag(candTag);

  // filter decision
  const bool accept (n >= minN);

  LogDebug("HLTMuonL1RegionalFilter")<<dumpEvent(allMuons, prevMuons, &(*filterproduct))<<"Decision of filter is "<<accept<<", number of muons passing = "<<filterproduct->l1muonSize();

  // put filter object into the Event
  iEvent.put(filterproduct);

  return accept;
}

string HLTMuonL1RegionalFilter::dumpParameters(){
  stringstream ss;
  ss<<"Constructed with parameters:"<<endl;
  ss<<"    CandTag = "<<candTag.encode()<<endl;
  ss<<"    PreviousCandTag = "<<previousCandTag.encode()<<endl;
  ss<<"    EtaBoundaries = \t"<<etaBoundaries[0];
  for(size_t i=1; i<etaBoundaries.size(); i++){
    ss<<'\t'<<etaBoundaries[i];
  }
  ss<<endl;
  ss<<"    MinPts =        \t    "<<minPts[0];
  for(size_t i=1; i<minPts.size(); i++){
    ss<<"\t    "<<minPts[i];
  }
  ss<<endl;
  ss<<"    QualityBitMasks =  \t    "<<qualityBitMasks[0];
  for(size_t i=1; i<qualityBitMasks.size(); i++){
    ss<<"\t    "<<qualityBitMasks[i];
  }
  ss<<endl;
  ss<<"    MinN = "<<minN<<endl;
  ss<<"    SaveTag = "<<saveTag;
  return ss.str();
}

string HLTMuonL1RegionalFilter::dumpEvent(Handle<L1MuonParticleCollection>& allMuons, vector<L1MuonParticleRef>& prevMuons, TriggerFilterObjectWithRefs* filterproduct){
  stringstream ss;
  ss.precision(4);
  ss<<"muon#"<<'\t'<<"q*pt"<<'\t'<<"eta"<<'\t'<<"phi"<<'\t'<<"quality"<<'\t'<<"isPrev"<<'\t'<<"isFired"<<endl;
  ss<<"-------------------------------------------------------"<<endl;

  vector<L1MuonParticleRef> firedMuons;
  filterproduct->getObjects(TriggerL1Mu, firedMuons);
  for(size_t i=0; i<allMuons->size(); i++){
    L1MuonParticleRef mu(allMuons, i);
    int quality = mu->gmtMuonCand().empty() ? 0 : (1 << mu->gmtMuonCand().quality());
    bool isPrev = find(prevMuons.begin(), prevMuons.end(), mu) != prevMuons.end();
    bool isFired = find(firedMuons.begin(), firedMuons.end(), mu) != firedMuons.end();
    ss<<i<<'\t'<<mu->charge()*mu->pt()<<'\t'<<mu->eta()<<'\t'<<mu->phi()<<'\t'<<quality<<'\t'<<isPrev<<'\t'<<isFired<<endl;
  }
  ss<<"-------------------------------------------------------"<<endl;
  return ss.str();
}


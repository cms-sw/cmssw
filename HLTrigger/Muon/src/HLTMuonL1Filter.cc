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

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTCand.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "TMath.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

//
// constructors and destructor
//
HLTMuonL1Filter::HLTMuonL1Filter(const edm::ParameterSet& iConfig) : HLTFilter(iConfig),
  candTag_( iConfig.getParameter<edm::InputTag>("CandTag") ),
  candToken_( consumes<l1extra::L1MuonParticleCollection>(candTag_)),
  previousCandTag_( iConfig.getParameter<edm::InputTag>("PreviousCandTag") ),
  previousCandToken_( consumes<trigger::TriggerFilterObjectWithRefs>(previousCandTag_)),
  maxEta_( iConfig.getParameter<double>("MaxEta") ),
  minPt_( iConfig.getParameter<double>("MinPt") ),
  minN_( iConfig.getParameter<int>("MinN") ),
  excludeSingleSegmentCSC_( iConfig.getParameter<bool>("ExcludeSingleSegmentCSC") ),
  csctfTag_( iConfig.getParameter<edm::InputTag>("CSCTFtag") ),
  csctfToken_(excludeSingleSegmentCSC_ ? consumes<L1CSCTrackCollection>(csctfTag_) : edm::EDGetTokenT<L1CSCTrackCollection>{})
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
    ss<<"    ExcludeSingleSegmentCSC = "<<excludeSingleSegmentCSC_<<endl;
    ss<<"    CSCTFtag = "<<csctfTag_.encode()<<endl;
    ss<<"    saveTags= "<<saveTags();
    LogDebug("HLTMuonL1Filter")<<ss.str();
  }
}

HLTMuonL1Filter::~HLTMuonL1Filter()
{
}

void
HLTMuonL1Filter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);
  desc.add<edm::InputTag>("CandTag",edm::InputTag("hltL1extraParticles"));
  //  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag("hltL1sL1DoubleMuOpen"));
  desc.add<edm::InputTag>("PreviousCandTag",edm::InputTag(""));
  desc.add<double>("MaxEta",2.5);
  desc.add<double>("MinPt",0.0);
  desc.add<int>("MinN",1);
  desc.add<bool>("ExcludeSingleSegmentCSC",false);
  //  desc.add<edm::InputTag>("CSCTFtag",edm::InputTag("unused"));
  desc.add<edm::InputTag>("CSCTFtag",edm::InputTag("csctfDigis"));
  {
    std::vector<int> temp1;
    temp1.reserve(0);
    desc.add<std::vector<int> >("SelectQualities",temp1);
  }
  descriptions.add("hltMuonL1Filter",desc);
}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool HLTMuonL1Filter::hltFilter(edm::Event& iEvent, const edm::EventSetup& iSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) const {
  using namespace std;
  using namespace edm;
  using namespace trigger;
  using namespace l1extra;

  // All HLT filters must create and fill an HLT filter object,
  // recording any reconstructed physics objects satisfying (or not)
  // this HLT filter, and place it in the Event.

  // get hold of all muons
  Handle<L1MuonParticleCollection> allMuons;
  iEvent.getByToken(candToken_, allMuons);

  /// handle for CSCTFtracks
  const L1CSCTrackCollection * csctfTracks = nullptr;
  const L1MuTriggerScales * l1MuTriggerScales = nullptr;

  // get hold of CSCTF raw tracks
  if( excludeSingleSegmentCSC_ ) {
    edm::Handle<L1CSCTrackCollection> csctfTracksHandle;
    iEvent.getByToken(csctfToken_, csctfTracksHandle);
    csctfTracks = csctfTracksHandle.product();

    // read scales for every event (fast, no need to cache this)
    ESHandle<L1MuTriggerScales> scales;
    iSetup.get<L1MuTriggerScalesRcd>().get(scales);
    l1MuTriggerScales = scales.product();
  }

  // get hold of muons that fired the previous level
  Handle<TriggerFilterObjectWithRefs> previousLevelCands;
  iEvent.getByToken(previousCandToken_, previousLevelCands);
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

    // reject single-segment CSC objects if necessary
    if (excludeSingleSegmentCSC_ and isSingleSegmentCSC(muon, * csctfTracks, * l1MuTriggerScales)) continue;

    //we have a good candidate
    n++;
    filterproduct.addObject(TriggerL1Mu,muon);
  }

  if (saveTags()) filterproduct.addCollectionTag(candTag_);

  // filter decision
  const bool accept(n >= minN_);

  // dump event for debugging
  if(edm::isDebugEnabled()){
    ostringstream ss;
    ss.precision(2);
    ss<<"L1mu#"<<'\t'<<"q*pt"<<'\t'<<'\t'<<"eta"<<'\t'<<"phi"<<'\t'<<"quality"<<'\t'<<"isPrev"<<'\t'<<"isFired"<<'\t'<<"isSingleCSC"<<endl;
    ss<<"--------------------------------------------------------------------------"<<endl;

    vector<L1MuonParticleRef> firedMuons;
    filterproduct.getObjects(TriggerL1Mu, firedMuons);
    for(size_t i=0; i<allMuons->size(); i++){
      L1MuonParticleRef mu(allMuons, i);
      int quality = mu->gmtMuonCand().empty() ? 0 : mu->gmtMuonCand().quality();
      bool isPrev = find(prevMuons.begin(), prevMuons.end(), mu) != prevMuons.end();
      bool isFired = find(firedMuons.begin(), firedMuons.end(), mu) != firedMuons.end();
      bool isSingleCSC = excludeSingleSegmentCSC_ and isSingleSegmentCSC(mu, * csctfTracks, * l1MuTriggerScales);
      ss<<i<<'\t'<<scientific<<mu->charge()*mu->pt()<<'\t'<<fixed<<mu->eta()<<'\t'<<mu->phi()<<'\t'<<quality<<'\t'<<isPrev<<'\t'<<isFired<<'\t'<<isSingleCSC<<endl;
    }
    ss<<"--------------------------------------------------------------------------"<<endl;
    LogDebug("HLTMuonL1Filter")<<ss.str()<<"Decision of filter is "<<accept<<", number of muons passing = "<<filterproduct.l1muonSize();
  }

  return accept;
}

bool HLTMuonL1Filter::isSingleSegmentCSC(l1extra::L1MuonParticleRef const & muon, L1CSCTrackCollection const & csctfTracks, L1MuTriggerScales const & l1MuTriggerScales) const {
  // is the muon matching a csctf track?
  //bool matched   = false;     // unused
  // which csctf track mode?
  // -999: no matching
  //  1: bad phi road. Not good extrapolation, but still triggering
  // 11: singles
  // 15: halo
  // 2->10 and 12->14: coincidence trigger with good extrapolation
  int csctfMode = -999;

  // loop over the CSCTF tracks
  for(L1CSCTrackCollection::const_iterator trk = csctfTracks.begin(); trk < csctfTracks.end(); trk++){

    int trEndcap = (trk->first.endcap()==2 ? trk->first.endcap()-3 : trk->first.endcap());
    int trSector = 6*(trk->first.endcap()-1)+trk->first.sector();

    //... in radians
    // Type 2 is CSC
    float trEtaScale = l1MuTriggerScales.getRegionalEtaScale(2)->getCenter(trk->first.eta_packed());
    float trPhiScale = l1MuTriggerScales.getPhiScale()->getLowEdge(trk->first.localPhi());

    double trEta = trEtaScale * trEndcap;
    // there is no L1ExtraParticle below -2.375
    if(trEta<-2.4) trEta=-2.375;

    // CSCTF has 6 sectors
    // sector 1 starts at 15 degrees
    // trPhiScale is defined inside a sector
    float trPhi02PI = fmod(trPhiScale +
                           ((trSector-1)*TMath::Pi()/3) +
                           (TMath::Pi()/12) , 2*TMath::Pi());

    // L1 information are given from [-Pi,Pi]
    double trPhi = ( trPhi02PI<TMath::Pi()? trPhi02PI : trPhi02PI - 2*TMath::Pi() );
    /*
    std::cout << "\ntrEndcap="               << trEndcap                << std::endl;
    std::cout << "trSector="                 << trSector                << std::endl;
    std::cout << "trk->first.eta_packed()="  << trk->first.eta_packed() << std::endl;
    std::cout << "trk->first.localPhi()="    << trk->first.localPhi()   << std::endl;
    std::cout << "trEtaScale=" << trEtaScale << std::endl;
    std::cout << "trPhiScale=" << trPhiScale << std::endl;
    std::cout << "trEta="      << trEta      << std::endl;
    std::cout << "trPhi="      << trPhi      << std::endl;
    */
    if ( fabs (trEta-muon->eta()) < 0.03 &&
         fabs (trPhi-muon->phi()) < 0.001  ) {

      //matched = true;
      ptadd thePtAddress(trk->first.ptLUTAddress());
      csctfMode = thePtAddress.track_mode;
      //std::cout << "is matched -> trMode=" << csctfMode << std::endl;
    }
  }

  /*
  std::cout << " ===================================== " << std::endl;
  std::cout << " is matched? " << matched                << std::endl;
  std::cout << " is singles? " << (csctfMode==11 ? 1 :0) << std::endl;
  std::cout << " ===================================== " << std::endl;
  */

  // singles are mode 11 "CSCTF tracks"
  return csctfMode==11;
}

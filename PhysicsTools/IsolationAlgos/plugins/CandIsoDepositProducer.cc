#include "PhysicsTools/IsolationAlgos/plugins/CandIsoDepositProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"


#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace reco;
using namespace muonisolation;


/// constructor with config
CandIsoDepositProducer::CandIsoDepositProducer(const ParameterSet& par) :
  theConfig(par),
  theCandCollectionTag(par.getParameter<edm::InputTag>("src")),
  theDepositNames(std::vector<std::string>(1,"")),
  theMultipleDepositsFlag(par.getParameter<bool>("MultipleDepositsFlag")),
  theExtractor(0)
  {
  LogDebug("PhysicsTools|MuonIsolation")<<" CandIsoDepositProducer CTOR";

  edm::ParameterSet extractorPSet = theConfig.getParameter<edm::ParameterSet>("ExtractorPSet");
  std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
  theExtractor = IsoDepositExtractorFactory::get()->create( extractorName, extractorPSet);

  if (! theMultipleDepositsFlag) produces<reco::IsoDepositMap>();
  else {
    theDepositNames = extractorPSet.getParameter<std::vector<std::string> >("DepositInstanceLabels");
    if (theDepositNames.size() > 10) throw cms::Exception("Configuration Error") << "This module supports only up to 10 deposits"; 
    for (unsigned int iDep=0; iDep<theDepositNames.size(); ++iDep){
      produces<reco::IsoDepositMap>(theDepositNames[iDep]);
    }
  }

  std::string trackType = par.getParameter<std::string>("trackType");
  if      (trackType == "fake") theTrackType = FakeT;
  else if (trackType == "best") theTrackType = BestT;
  else if (trackType == "standAloneMuon") theTrackType = StandAloneMuonT;
  else if (trackType == "combinedMuon")   theTrackType = CombinedMuonT;
  else if (trackType == "trackerMuon")    theTrackType = TrackT;
  else if (trackType == "track") theTrackType = TrackT;
  else if (trackType == "gsf") theTrackType = GsfT;
  else if (trackType == "candidate") theTrackType = CandidateT;
  else throw cms::Exception("Error")  << "Track type " << trackType << " not valid.";
}

/// destructor
CandIsoDepositProducer::~CandIsoDepositProducer(){
  LogDebug("PhysicsTools/CandIsoDepositProducer")<<" CandIsoDepositProducer DTOR";
  delete theExtractor;
}

inline const reco::Track * CandIsoDepositProducer::extractTrack(const reco::Candidate &c, reco::Track *dummy) const {
    if (theTrackType == CandidateT) {
        return 0;
    } else if (theTrackType == FakeT) {
        *dummy = Track(10,10,c.vertex(),c.momentum(),c.charge(), reco::Track::CovarianceMatrix());
        return dummy;
    } else {
        const RecoCandidate *rc = dynamic_cast<const RecoCandidate *>(&c);
        if (rc == 0) throw cms::Exception("Error") << " Candidate is not RecoCandidate: can't get a real track from it!";
        switch (theTrackType) {
            case FakeT: break; // to avoid warning
            case CandidateT: break; // to avoid warning
            case BestT: return rc->bestTrack(); break;
            case StandAloneMuonT: return &*rc->standAloneMuon(); break;
            case CombinedMuonT:   return &*rc->combinedMuon(); break;
            case TrackT: return &*rc->track(); break;
            case GsfT: return static_cast<const Track*>(rc->gsfTrack().get()); break;
        }
        return 0;
    }
}

/// build deposits
void CandIsoDepositProducer::produce(Event& event, const EventSetup& eventSetup){
  static const std::string metname = "CandIsoDepositProducer";

  edm::Handle< edm::View<reco::Candidate> > hCands;
  event.getByLabel(theCandCollectionTag, hCands);
    
  unsigned int nDeps = theMultipleDepositsFlag ? theDepositNames.size() : 1;

  static const unsigned int MAX_DEPS=10; 
  std::auto_ptr<reco::IsoDepositMap> depMaps[MAX_DEPS];
  
  if (nDeps >10 ) LogError(metname)<<"Unable to handle more than 10 input deposits"; 
  for (unsigned int i =0;i<nDeps; ++i){ // check if nDeps > 10??
    depMaps[i] =  std::auto_ptr<reco::IsoDepositMap>(new reco::IsoDepositMap()); 
  } 
   
  //! OK, now we know how many deps for how many muons each we will create 
  //! might linearize this at some point (lazy) 
  //! do it in case some muons are there only 
  size_t nMuons = hCands->size();
  if (nMuons > 0){ 
    std::vector<std::vector<IsoDeposit> > deps2D(nDeps, std::vector<IsoDeposit>(nMuons)); 


    Track dummy;
    for (size_t i=0; i<  nMuons; ++i) {
      const Candidate &c = (*hCands)[i];
      const Track *track = extractTrack(c, &dummy);
      if ((theTrackType != CandidateT) && (!track)) {
        edm::LogWarning("CandIsoDepositProducer") << "Candidate #"<<i<<" has no bestTrack(), it will produce no deposit";
	reco::IsoDeposit emptyDep;
        for (size_t iDep=0;iDep<nDeps;++iDep) {
	  deps2D[iDep][i] = emptyDep; //! well, it is empty already by construction, but still
        }
        continue;
      }
      if (!theMultipleDepositsFlag){
	IsoDeposit dep = ( ( theTrackType == CandidateT )
			     ? theExtractor->deposit(event, eventSetup, c)
			     : theExtractor->deposit(event, eventSetup, *track) );
	deps2D[0][i] = dep;
      } else {
	std::vector<IsoDeposit> deps = ( ( theTrackType == CandidateT )
					   ? theExtractor->deposits(event, eventSetup, c)
					   : theExtractor->deposits(event, eventSetup, *track) );
	for (unsigned int iDep=0; iDep < nDeps; ++iDep){ 	deps2D[iDep][i] =  deps[iDep];  }
      }
    }//! for(i<nMuons)
    

    //! now fill in selectively 
    for (unsigned int iDep=0; iDep < nDeps; ++iDep){ 
      //!some debugging stuff 
      for (unsigned int iMu = 0; iMu< nMuons; ++iMu){ 
        LogTrace(metname)<<"Contents of "<<theDepositNames[iDep] 
                         <<" for a muon at index "<<iMu; 
        LogTrace(metname)<<deps2D[iDep][iMu].print(); 
      } 
 
      //! fill the maps here   
      reco::IsoDepositMap::Filler filler(*depMaps[iDep]);      
      filler.insert(hCands, deps2D[iDep].begin(), deps2D[iDep].end()); 
      filler.fill();
    }//! for(iDep<nDeps)
  }//! if (nMuons>0)

  for (unsigned int iMap = 0; iMap < nDeps; ++iMap) event.put(depMaps[iMap], theDepositNames[iMap]);
}

DEFINE_FWK_MODULE( CandIsoDepositProducer );

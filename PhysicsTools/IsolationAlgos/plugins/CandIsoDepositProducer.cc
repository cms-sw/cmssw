#include "PhysicsTools/IsolationAlgos/plugins/CandIsoDepositProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"


#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace std;
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
  theExtractor = MuIsoExtractorFactory::get()->create( extractorName, extractorPSet);

  if (! theMultipleDepositsFlag) produces<CandIsoDepositAssociationVector>();
  else {
    theDepositNames = extractorPSet.getParameter<std::vector<std::string> >("DepositInstanceLabels");
    if (theDepositNames.size() > 10) throw cms::Exception("Configuration Error") << "This module supports only up to 10 deposits"; 
    for (uint iDep=0; iDep<theDepositNames.size(); ++iDep){
      produces<CandIsoDepositAssociationVector>(theDepositNames[iDep]);
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
  edm::Handle< edm::View<reco::Candidate> > hCands;
  event.getByLabel(theCandCollectionTag, hCands);
  edm::RefToBaseProd<reco::Candidate> refProd(hCands);
    
  uint nDeps = theMultipleDepositsFlag ? theDepositNames.size() : 1;

  static const uint MAX_DEPS=10; 
  std::auto_ptr<CandIsoDepositAssociationVector> depMaps[MAX_DEPS];
  
  for (uint i =0;i<nDeps; ++i){ // check if nDeps > 10??
    depMaps[i] =  std::auto_ptr<CandIsoDepositAssociationVector>(new CandIsoDepositAssociationVector(refProd));
  }

  size_t nMuons = hCands->size();
  Track dummy;
  for (size_t i=0; i<  nMuons; ++i) {
    const Candidate &c = (*hCands)[i];
    const Track *track = extractTrack(c, &dummy);
    if ((theTrackType != CandidateT) && (!track)) {
        edm::LogWarning("CandIsoDepositProducer") << "Candidate #"<<i<<" has no bestTrack(), it will produce no deposit";
        MuIsoDeposit empty("dummy");
        for (size_t iDep=0;iDep<nDeps;++iDep) {
            depMaps[iDep]->setValue(i, empty);
        }
        continue;
    }
    if (!theMultipleDepositsFlag){
      MuIsoDeposit dep = ( ( theTrackType == CandidateT )
                                ? theExtractor->deposit(event, eventSetup, c)
                                : theExtractor->deposit(event, eventSetup, *track) );
      depMaps[0]->setValue(i, dep);
    } else {
      std::vector<MuIsoDeposit> deps = ( ( theTrackType == CandidateT )
                                            ? theExtractor->deposits(event, eventSetup, c)
                                            : theExtractor->deposits(event, eventSetup, *track) );
      for (uint iDep=0; iDep < nDeps; ++iDep){ 	depMaps[iDep]->setValue(i, deps[iDep]);  }
    }
  }


  for (uint iMap = 0; iMap < nDeps; ++iMap) event.put(depMaps[iMap], theDepositNames[iMap]);
}

DEFINE_FWK_MODULE( CandIsoDepositProducer );

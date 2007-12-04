#include "RecoMuon/MuonIsolationProducers/plugins/MuIsoDepositProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"


#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;


//! constructor with config
MuIsoDepositProducer::MuIsoDepositProducer(const ParameterSet& par) :
  theConfig(par),
  theDepositNames(std::vector<std::string>(1,std::string())),
  theExtractor(0)
{
  LogDebug("RecoMuon|MuonIsolation")<<" MuIsoDepositProducer CTOR";

  edm::ParameterSet ioPSet = par.getParameter<edm::ParameterSet>("IOPSet");

  theInputType = ioPSet.getParameter<std::string>("InputType");
  theOutputType = ioPSet.getParameter<std::string>("OutputType");
  theExtractForCandidate = ioPSet.getParameter<bool>("ExtractForCandidate");
  theMuonTrackRefType = ioPSet.getParameter<std::string>("MuonTrackRefType");
  theMuonCollectionTag = ioPSet.getParameter<edm::InputTag>("inputMuonCollection");
  theMultipleDepositsFlag = ioPSet.getParameter<bool>("MultipleDepositsFlag");
  

  
  if (theMultipleDepositsFlag){
    theDepositNames = par.getParameter<edm::ParameterSet>("ExtractorPSet")
      .getParameter<std::vector<std::string> >("DepositInstanceLabels");
  }
  
  if (theOutputType == "MapToMuons"){
    callProduces<MuIsoDepositAssociationMapToMuon >(theDepositNames);
  } else if (theOutputType == "MapToTracks"){
    callProduces<MuIsoDepositAssociationMap >(theDepositNames);
  }else if (theOutputType == "VectorToMuons"){
    callProduces<MuIsoDepositAssociationVectorToMuon >(theDepositNames);
  } else if (theOutputType == "VectorToTracks"){
    callProduces<MuIsoDepositAssociationVector >(theDepositNames);
  } else if (theOutputType == "VectorToCandidateView"){
    callProduces<MuIsoDepositAssociationVectorToCandidateView >(theDepositNames);
  } else {
    edm::LogError("MuonIsolation")<<" Unknown output type requested ";
  }

}

/// destructor
MuIsoDepositProducer::~MuIsoDepositProducer(){
  LogDebug("RecoMuon/MuIsoDepositProducer")<<" MuIsoDepositProducer DTOR";
  delete theExtractor;
}

/// build deposits
void MuIsoDepositProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|MuonIsolationProducers|MuIsoDepositProducer";

  LogDebug(metname)<<" Muon Deposit producing..."
		   <<" BEGINING OF EVENT " <<"================================";

  if (!theExtractor) {
    edm::ParameterSet extractorPSet = theConfig.getParameter<edm::ParameterSet>("ExtractorPSet");
    std::string extractorName = extractorPSet.getParameter<std::string>("ComponentName");
    theExtractor = MuIsoExtractorFactory::get()->create( extractorName, extractorPSet);
    LogDebug(metname)<<" Load extractor..."<<extractorName;
  }


  uint nDeps = theMultipleDepositsFlag ? theDepositNames.size() : 1;



  // Take the muon container
  LogTrace(metname)<<" Taking the muons: "<<theMuonCollectionTag;
  Handle<TrackCollection> muonTracks;
  Handle<MuonCollection> muons;
  Handle<View<Candidate> > candsView;

  uint nMuons = 99999;

  bool readFromRecoTrack = theInputType == "TrackCollection";
  bool readFromRecoMuon = theInputType == "MuonCollection";
  bool readFromCandidateView = theInputType == "CandidateView";

  if (readFromRecoMuon){
    event.getByLabel(theMuonCollectionTag,muons);
    nMuons = muons->size();
    LogDebug(metname) <<"Got Muons of size "<<nMuons;
    
  } 
  if (readFromRecoTrack){
    event.getByLabel(theMuonCollectionTag,muonTracks);
    nMuons = muonTracks->size();
    LogDebug(metname) <<"Got MuonTracks of size "<<nMuons;
  }
  if (readFromCandidateView || theExtractForCandidate 
      || theOutputType == "VectorToCandidateView"){
    event.getByLabel(theMuonCollectionTag,candsView);
    uint nCands = candsView->size();
    if (nCands != nMuons && nMuons != 99999) edm::LogError(metname)<<"Wrong muon-candidate count";
    LogDebug(metname)<< "Got candidate view with size "<<nCands;
    if (nMuons == 99999) nMuons = nCands;
  }

  static const uint MAX_DEPS=10;
  std::auto_ptr<MuIsoDepositAssociationMap> depMaps[MAX_DEPS];
  std::auto_ptr<MuIsoDepositAssociationMapToMuon> depMapsMus[MAX_DEPS];
  std::auto_ptr<MuIsoDepositAssociationVector> depVecs[MAX_DEPS];
  std::auto_ptr<MuIsoDepositAssociationVectorToMuon> depVecsMus[MAX_DEPS];
  std::auto_ptr<MuIsoDepositAssociationVectorToCandidateView> depVecsCand[MAX_DEPS];
  for (uint i =0;i<nDeps; ++i){
    if (theOutputType == "MapToMuons"){
      depMapsMus[i] =  std::auto_ptr<MuIsoDepositAssociationMapToMuon>(new MuIsoDepositAssociationMapToMuon());
    } else if (theOutputType == "MapToTracks"){
      depMaps[i] =  std::auto_ptr<MuIsoDepositAssociationMap>(new MuIsoDepositAssociationMap());
    } else if (theOutputType == "VectorToMuons"){
      MuonRefProd refMuons(muons);
      depVecsMus[i] =  std::auto_ptr<MuIsoDepositAssociationVectorToMuon>(new MuIsoDepositAssociationVectorToMuon(refMuons));
    } else if (theOutputType == "VectorToTracks"){
      TrackRefProd refTracks(muonTracks);
      depVecs[i] =  std::auto_ptr<MuIsoDepositAssociationVector>(new MuIsoDepositAssociationVector(refTracks));
    } else if (theOutputType == "VectorToCandidateView"){
      CandidateBaseRefProd refBCands(candsView);
      depVecsCand[i] =  std::auto_ptr<MuIsoDepositAssociationVectorToCandidateView>(new MuIsoDepositAssociationVectorToCandidateView(refBCands));
    }
  }


  for (uint i=0; i<  nMuons; ++i) {
    TrackRef mu;
    if (readFromRecoMuon){
      if (theMuonTrackRefType == "track"){
	mu = (*muons)[i].track();
      } else if (theMuonTrackRefType == "standAloneMuon"){
	mu = (*muons)[i].standAloneMuon();
      } else if (theMuonTrackRefType == "combinedMuon"){
	mu = (*muons)[i].combinedMuon();
      } else if (theMuonTrackRefType == "bestGlbTrkSta"){
	if (!(*muons)[i].combinedMuon().isNull()){
	  mu = (*muons)[i].combinedMuon();
	} else if (!(*muons)[i].track().isNull()){
	  mu = (*muons)[i].track();
        } else {
	  mu = (*muons)[i].standAloneMuon();
	}
      }else {
	edm::LogWarning(metname)<<"Wrong track type is supplied: breaking";
	break;
      }
    } else if (readFromRecoTrack){
      mu = TrackRef(muonTracks, i);
    }
    std::vector<MuIsoDeposit> deps(1);
    if (! theMultipleDepositsFlag){
      if (theExtractForCandidate) deps[0] = theExtractor->deposit(event, eventSetup, (*candsView)[i]);
      else deps[0] = theExtractor->deposit(event, eventSetup, *mu);

    } else {
      if (theExtractForCandidate) deps = theExtractor->deposits(event, eventSetup, (*candsView)[i]);
      else deps = theExtractor->deposits(event, eventSetup, *mu);

    }

    //! now fill in selectively
    for (uint iDep=0; iDep < nDeps; ++iDep){
      LogTrace(metname)<<deps[iDep].print();
      if (theOutputType == "MapToMuons"){
	depMapsMus[iDep]->insert(MuonRef(muons,i), deps[iDep]);
      } else if (theOutputType == "MapToTracks"){
	depMaps[iDep]->insert(mu, deps[iDep]);
      } else if (theOutputType == "VectorToMuons"){
	depVecsMus[iDep]->setValue(i, deps[iDep]);
      } else if (theOutputType == "VectorToTracks"){
	depVecs[iDep]->setValue(i, deps[iDep]);
      } else if (theOutputType == "VectorToCandidateView"){
	depVecsCand[iDep]->setValue(i, deps[iDep]);
      }
    }
  }


  for (uint iMap = 0; iMap < nDeps; ++iMap){
    if (theOutputType == "MapToMuons"){
      event.put(depMapsMus[iMap], theDepositNames[iMap]);
    } else if (theOutputType == "MapToTracks"){
      event.put(depMaps[iMap], theDepositNames[iMap]);
    }else if (theOutputType == "VectorToMuons"){
      event.put(depVecsMus[iMap], theDepositNames[iMap]);
    } else if (theOutputType == "VectorToTracks"){
      event.put(depVecs[iMap], theDepositNames[iMap]);
    } else if (theOutputType == "VectorToCandidateView"){
      event.put(depVecsCand[iMap], theDepositNames[iMap]);
    } else {
      edm::LogError("MuonIsolation")<<" Unknown output type requested ";
    }
    
  }
  LogTrace(metname) <<" END OF EVENT " <<"================================";
}


template <typename T>
void MuIsoDepositProducer::callProduces(const std::vector<std::string>& instLabels){
  for (uint i = 0; i < instLabels.size(); ++i){
    std::string alias = theConfig.getParameter<std::string>("@module_label");
    if (instLabels[i] != "") alias += "_" + instLabels[i];
    produces<T>(instLabels[i]).setBranchAlias(alias);
  }
}


#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuIsoDepositProducer);

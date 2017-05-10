////////////////////////////////////////////////////////////////////////////////
//
// FastjetJetProducer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetProducers/plugins/FastjetJetProducer.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"


#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "fastjet/SISConePlugin.hh"
#include "fastjet/CMSIterativeConePlugin.hh"
#include "fastjet/ATLASConePlugin.hh"
#include "fastjet/CDFMidPointPlugin.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "RecoJets/JetAlgorithms/interface/CMSBoostedTauSeedingAlgorithm.h"

#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
//#include <fstream>

using namespace std;



////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FastjetJetProducer::FastjetJetProducer(const edm::ParameterSet& iConfig)
  : VirtualJetProducer( iConfig ),
    useMassDropTagger_(false),
    useFiltering_(false),
    useTrimming_(false),
    usePruning_(false),
    useCMSBoostedTauSeedingAlgorithm_(false),
    muCut_(-1.0),
    yCut_(-1.0),
    rFilt_(-1.0),
    nFilt_(-1),
    trimPtFracMin_(-1.0),
    zCut_(-1.0),
    RcutFactor_(-1.0)
{

  if ( iConfig.exists("UseOnlyVertexTracks") )
    useOnlyVertexTracks_ = iConfig.getParameter<bool>("UseOnlyVertexTracks");
  else 
    useOnlyVertexTracks_ = false;
  
  if ( iConfig.exists("UseOnlyOnePV") )
    useOnlyOnePV_        = iConfig.getParameter<bool>("UseOnlyOnePV");
  else
    useOnlyOnePV_ = false;

  if ( iConfig.exists("DzTrVtxMax") )
    dzTrVtxMax_          = iConfig.getParameter<double>("DzTrVtxMax");
  else
    dzTrVtxMax_ = 999999.;
  if ( iConfig.exists("DxyTrVtxMax") )
    dxyTrVtxMax_          = iConfig.getParameter<double>("DxyTrVtxMax");
  else
    dxyTrVtxMax_ = 999999.;
  if ( iConfig.exists("MinVtxNdof") )
    minVtxNdof_ = iConfig.getParameter<int>("MinVtxNdof");
  else
    minVtxNdof_ = 5;
  if ( iConfig.exists("MaxVtxZ") )
    maxVtxZ_ = iConfig.getParameter<double>("MaxVtxZ");
  else
    maxVtxZ_ = 15;


  if ( iConfig.exists("useFiltering") ||
       iConfig.exists("useTrimming") ||
       iConfig.exists("usePruning") ||
       iConfig.exists("useMassDropTagger") ||
       iConfig.exists("useCMSBoostedTauSeedingAlgorithm") ) {
    useMassDropTagger_=false;
    useFiltering_=false;
    useTrimming_=false;
    usePruning_=false;
    useCMSBoostedTauSeedingAlgorithm_=false;
    rFilt_=-1.0;
    nFilt_=-1;
    trimPtFracMin_=-1.0;
    zCut_=-1.0;
    RcutFactor_=-1.0;
    muCut_=-1.0;
    yCut_=-1.0;
    subjetPtMin_ = -1.0;
    muMin_ = -1.0;
    muMax_ = -1.0;
    yMin_ = -1.0;
    yMax_ = -1.0;
    dRMin_ = -1.0;
    dRMax_ = -1.0;
    maxDepth_ = -1;
    useExplicitGhosts_ = true;

    if ( iConfig.exists("useMassDropTagger") ) {
      useMassDropTagger_ = true;
      muCut_ = iConfig.getParameter<double>("muCut");
      yCut_ = iConfig.getParameter<double>("yCut");
    }

    if ( iConfig.exists("useFiltering") ) {
      useFiltering_ = true;
      rFilt_ = iConfig.getParameter<double>("rFilt");
      nFilt_ = iConfig.getParameter<int>("nFilt");
    }
  
    if ( iConfig.exists("useTrimming") ) {
      useTrimming_ = true;
      rFilt_ = iConfig.getParameter<double>("rFilt");
      trimPtFracMin_ = iConfig.getParameter<double>("trimPtFracMin");
    }

    if ( iConfig.exists("usePruning") ) {
      usePruning_ = true;
      zCut_ = iConfig.getParameter<double>("zcut");
      RcutFactor_ = iConfig.getParameter<double>("rcut_factor");
      nFilt_ = iConfig.getParameter<int>("nFilt");
    }

    if ( iConfig.exists("useCMSBoostedTauSeedingAlgorithm") ) {
      useCMSBoostedTauSeedingAlgorithm_ = iConfig.getParameter<bool>("useCMSBoostedTauSeedingAlgorithm");
      subjetPtMin_ = iConfig.getParameter<double>("subjetPtMin");
      muMin_ = iConfig.getParameter<double>("muMin");
      muMax_ = iConfig.getParameter<double>("muMax");
      yMin_ = iConfig.getParameter<double>("yMin");
      yMax_ = iConfig.getParameter<double>("yMax");
      dRMin_ = iConfig.getParameter<double>("dRMin");
      dRMax_ = iConfig.getParameter<double>("dRMax");
      maxDepth_ = iConfig.getParameter<int>("maxDepth");
    }

  }

}


//______________________________________________________________________________
FastjetJetProducer::~FastjetJetProducer()
{
} 


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void FastjetJetProducer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{

  // for everything but track jets
  if (!makeTrackJet(jetTypeE)) {
 
     // use the default production from one collection
     VirtualJetProducer::produce( iEvent, iSetup );

  } else { // produce trackjets from tracks grouped per primary vertex

    produceTrackJets(iEvent, iSetup);
  
  }

}


void FastjetJetProducer::produceTrackJets( edm::Event & iEvent, const edm::EventSetup & iSetup )
{

    // read in the track candidates
    edm::Handle<edm::View<reco::RecoChargedRefCandidate> > inputsHandle;
    iEvent.getByLabel(src_, inputsHandle);
    // make collection with pointers so we can play around with it
    std::vector<edm::Ptr<reco::RecoChargedRefCandidate> > allInputs;
    std::vector<edm::Ptr<reco::Candidate> > origInputs;
    for (size_t i = 0; i < inputsHandle->size(); ++i) {
      allInputs.push_back(inputsHandle->ptrAt(i));
      origInputs.push_back(inputsHandle->ptrAt(i));
    }

    // read in the PV collection
    edm::Handle<reco::VertexCollection> pvCollection;
    iEvent.getByLabel(srcPVs_, pvCollection);
    // define the overall output jet container
    std::auto_ptr<std::vector<reco::TrackJet> > jets(new std::vector<reco::TrackJet>() );

    // loop over the good vertices, clustering for each vertex separately
    for (reco::VertexCollection::const_iterator itVtx = pvCollection->begin(); itVtx != pvCollection->end(); ++itVtx) {
      if (itVtx->isFake() || itVtx->ndof() < minVtxNdof_ || fabs(itVtx->z()) > maxVtxZ_) continue;

      // clear the intermediate containers
      inputs_.clear();
      fjInputs_.clear();
      fjJets_.clear();

      // if only vertex-associated tracks should be used
      if (useOnlyVertexTracks_) {
        // loop over the tracks associated to the vertex
        for (reco::Vertex::trackRef_iterator itTr = itVtx->tracks_begin(); itTr != itVtx->tracks_end(); ++itTr) {
          // whether a match was found in the track candidate input
          bool found = false;
          // loop over input track candidates
          for (std::vector<edm::Ptr<reco::RecoChargedRefCandidate> >::iterator itIn = allInputs.begin(); itIn != allInputs.end(); ++itIn) {
            // match the input track candidate to the track from the vertex
            reco::TrackRef trref(itTr->castTo<reco::TrackRef>());
            // check if the tracks match
            if ((*itIn)->track() == trref) {
              found = true;
              // add this track candidate to the input for clustering
              inputs_.push_back(*itIn);
              // erase the track candidate from the total list of input, so we don't reuse it later
              allInputs.erase(itIn);
              // found the candidate track corresponding to the vertex track, so stop the loop
              break;
            } // end if match found
          } // end loop over input tracks
          // give an info message in case no match is found (can happen if candidates are subset of tracks used for clustering)
          if (!found) edm::LogInfo("FastjetTrackJetProducer") << "Ignoring a track at vertex which is not in input track collection!";
        } // end loop over tracks associated to vertex
      // if all inpt track candidates should be used
      } else {
        // loop over input track candidates
        for (std::vector<edm::Ptr<reco::RecoChargedRefCandidate> >::iterator itIn = allInputs.begin(); itIn != allInputs.end(); ++itIn) {
          // check if the track is close enough to the vertex
          float dz = (*itIn)->track()->dz(itVtx->position());
          float dxy = (*itIn)->track()->dxy(itVtx->position());
          if (fabs(dz) > dzTrVtxMax_) continue;
          if (fabs(dxy) > dxyTrVtxMax_) continue;
          bool closervtx = false;
          // now loop over the good vertices a second time
          for (reco::VertexCollection::const_iterator itVtx2 = pvCollection->begin(); itVtx2 != pvCollection->end(); ++itVtx2) {
            if (itVtx->isFake() || itVtx->ndof() < minVtxNdof_ || fabs(itVtx->z()) > maxVtxZ_) continue;
            // and check this track is closer to any other vertex (if more than 1 vertex considered)
            if (!useOnlyOnePV_ &&
                itVtx != itVtx2 &&
                fabs((*itIn)->track()->dz(itVtx2->position())) < fabs(dz)) {
              closervtx = true;
              break; // 1 closer vertex makes the track already not matched, so break
            }
          }
          // don't add this track if another vertex is found closer
          if (closervtx) continue;
          // add this track candidate to the input for clustering
          inputs_.push_back(*itIn);
          // erase the track candidate from the total list of input, so we don't reuse it later
          allInputs.erase(itIn);
          // take a step back in the loop since we just erased
          --itIn;
        }
      }

      // convert candidates in inputs_ to fastjet::PseudoJets in fjInputs_
      fjInputs_.reserve(inputs_.size());
      inputTowers();
      LogDebug("FastjetTrackJetProducer") << "Inputted towers\n";

      // run algorithm, using fjInputs_, modifying fjJets_ and allocating fjClusterSeq_
      runAlgorithm(iEvent, iSetup);
      LogDebug("FastjetTrackJetProducer") << "Ran algorithm\n";

      // convert our jets and add to the overall jet vector
      for (unsigned int ijet=0;ijet<fjJets_.size();++ijet) {
        // get the constituents from fastjet
        std::vector<fastjet::PseudoJet> fjConstituents = sorted_by_pt(fjClusterSeq_->constituents(fjJets_[ijet]));
        // convert them to CandidatePtr vector
        std::vector<reco::CandidatePtr> constituents = getConstituents(fjConstituents);
        // fill the trackjet
        reco::TrackJet jet;
        // write the specifics to the jet (simultaneously sets 4-vector, vertex).
        writeSpecific( jet,
                       reco::Particle::LorentzVector(fjJets_[ijet].px(), fjJets_[ijet].py(), fjJets_[ijet].pz(), fjJets_[ijet].E()),
                       vertex_, constituents, iSetup);
        jet.setJetArea(0);
        jet.setPileup(0);
        jet.setPrimaryVertex(edm::Ref<reco::VertexCollection>(pvCollection, (int) (itVtx-pvCollection->begin())));
        jet.setVertex(itVtx->position());
        jets->push_back(jet);
      }

      if (useOnlyOnePV_) break; // stop vertex loop if only one vertex asked for
    } // end loop over vertices

    // put the jets in the collection
    LogDebug("FastjetTrackJetProducer") << "Put " << jets->size() << " jets in the event.\n";
    iEvent.put(jets);

}


//______________________________________________________________________________
void FastjetJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // run algorithm
  /*
  fjInputs_.clear();
  double px, py , pz, E;
  string line;
  std::ifstream fin("dump3.txt");
  while (getline(fin, line)){
    if (line == "#END") break;
    if (line.substr(0,1) == "#") {continue;}
    istringstream istr(line);
    istr >> px >> py >> pz >> E;
    // create a fastjet::PseudoJet with these components and put it onto
    // back of the input_particles vector
    fastjet::PseudoJet j(px,py,pz,E);
    //if ( fabs(j.rap()) < inputEtaMax )
    fjInputs_.push_back(fastjet::PseudoJet(px,py,pz,E)); 
  }
  fin.close();
  */

  if ( !doAreaFastjet_ && !doRhoFastjet_) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, *fjJetDefinition_ ) );
  } else if (voronoiRfact_ <= 0) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceArea( fjInputs_, *fjJetDefinition_ , *fjAreaDefinition_ ) );
  } else {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceVoronoiArea( fjInputs_, *fjJetDefinition_ , fastjet::VoronoiAreaSpec(voronoiRfact_) ) );
  }

  if ( !(useMassDropTagger_ || useCMSBoostedTauSeedingAlgorithm_ || useTrimming_ || useFiltering_ || usePruning_) ) {
    fjJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));
  } else {
    fjJets_.clear();
    std::vector<fastjet::PseudoJet> tempJets = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));

    fastjet::MassDropTagger md_tagger( muCut_, yCut_ );
    fastjet::contrib::CMSBoostedTauSeedingAlgorithm tau_tagger( subjetPtMin_, muMin_, muMax_, yMin_, yMax_, dRMin_, dRMax_, maxDepth_, verbosity_ );
    fastjet::Filter trimmer( fastjet::Filter(fastjet::JetDefinition(fastjet::kt_algorithm, rFilt_), fastjet::SelectorPtFractionMin(trimPtFracMin_)));
    fastjet::Filter filter( fastjet::Filter(fastjet::JetDefinition(fastjet::cambridge_algorithm, rFilt_), fastjet::SelectorNHardest(nFilt_)));
    fastjet::Pruner pruner(fastjet::cambridge_algorithm, zCut_, RcutFactor_);

    std::vector<fastjet::Transformer const *> transformers;

    if ( useMassDropTagger_ ) {
      transformers.push_back(&md_tagger);
    }
    if ( useCMSBoostedTauSeedingAlgorithm_ ) {
      transformers.push_back(&tau_tagger);
    }
    if ( useTrimming_ ) {
      transformers.push_back(&trimmer);
    } 
    if ( useFiltering_ ) {
      transformers.push_back(&filter);
    } 
    if ( usePruning_ ) {
      transformers.push_back(&pruner);
    }

    for ( std::vector<fastjet::PseudoJet>::const_iterator ijet = tempJets.begin(),
	    ijetEnd = tempJets.end(); ijet != ijetEnd; ++ijet ) {

      fastjet::PseudoJet transformedJet = *ijet;
      bool passed = true;
      for ( std::vector<fastjet::Transformer const *>::const_iterator itransf = transformers.begin(),
	      itransfEnd = transformers.end(); itransf != itransfEnd; ++itransf ) {
	if ( transformedJet != 0 ) {
	  transformedJet = (**itransf)(transformedJet);
	} else {
	  passed=false;
	}
      }
      if ( passed ) {
	fjJets_.push_back( transformedJet );
      }
    }
  }

}



////////////////////////////////////////////////////////////////////////////////
// define as cmssw plugin
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(FastjetJetProducer);


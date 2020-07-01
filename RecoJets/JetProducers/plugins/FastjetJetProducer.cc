////////////////////////////////////////////////////////////////////////////////
//
// FastjetJetProducer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetProducers/plugins/FastjetJetProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "fastjet/SISConePlugin.hh"
#include "fastjet/CMSIterativeConePlugin.hh"
#include "fastjet/ATLASConePlugin.hh"
#include "fastjet/CDFMidPointPlugin.hh"
#include "fastjet/tools/Filter.hh"
#include "fastjet/tools/Pruner.hh"
#include "fastjet/tools/MassDropTagger.hh"
#include "fastjet/contrib/SoftDrop.hh"
#include "fastjet/tools/JetMedianBackgroundEstimator.hh"
#include "fastjet/tools/GridMedianBackgroundEstimator.hh"
#include "fastjet/tools/Subtractor.hh"
#include "fastjet/contrib/ConstituentSubtractor.hh"
#include "RecoJets/JetAlgorithms/interface/CMSBoostedTauSeedingAlgorithm.h"

#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>
//#include <fstream>

using namespace std;
using namespace edm;

////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FastjetJetProducer::FastjetJetProducer(const edm::ParameterSet& iConfig) : VirtualJetProducer(iConfig) {
  useOnlyVertexTracks_ = iConfig.getParameter<bool>("UseOnlyVertexTracks");
  useOnlyOnePV_ = iConfig.getParameter<bool>("UseOnlyOnePV");
  dzTrVtxMax_ = iConfig.getParameter<double>("DzTrVtxMax");
  dxyTrVtxMax_ = iConfig.getParameter<double>("DxyTrVtxMax");
  minVtxNdof_ = iConfig.getParameter<int>("MinVtxNdof");
  maxVtxZ_ = iConfig.getParameter<double>("MaxVtxZ");

  useMassDropTagger_ = iConfig.getParameter<bool>("useMassDropTagger");
  muCut_ = iConfig.getParameter<double>("muCut");
  yCut_ = iConfig.getParameter<double>("yCut");

  useFiltering_ = iConfig.getParameter<bool>("useFiltering");
  rFilt_ = iConfig.getParameter<double>("rFilt");
  nFilt_ = iConfig.getParameter<int>("nFilt");
  useDynamicFiltering_ = iConfig.getParameter<bool>("useDynamicFiltering");
  if (useDynamicFiltering_)
    rFiltDynamic_ = std::make_shared<DynamicRfilt>(rFilt_, rFiltFactor_);
  rFiltFactor_ = iConfig.getParameter<double>("rFiltFactor");

  useTrimming_ = iConfig.getParameter<bool>("useTrimming");
  trimPtFracMin_ = iConfig.getParameter<double>("trimPtFracMin");

  usePruning_ = iConfig.getParameter<bool>("usePruning");
  zCut_ = iConfig.getParameter<double>("zcut");
  RcutFactor_ = iConfig.getParameter<double>("rcut_factor");
  nFilt_ = iConfig.getParameter<int>("nFilt");
  useKtPruning_ = iConfig.getParameter<bool>("useKtPruning");

  useCMSBoostedTauSeedingAlgorithm_ = iConfig.getParameter<bool>("useCMSBoostedTauSeedingAlgorithm");
  subjetPtMin_ = iConfig.getParameter<double>("subjetPtMin");
  muMin_ = iConfig.getParameter<double>("muMin");
  muMax_ = iConfig.getParameter<double>("muMax");
  yMin_ = iConfig.getParameter<double>("yMin");
  yMax_ = iConfig.getParameter<double>("yMax");
  dRMin_ = iConfig.getParameter<double>("dRMin");
  dRMax_ = iConfig.getParameter<double>("dRMax");
  maxDepth_ = iConfig.getParameter<int>("maxDepth");

  useConstituentSubtraction_ = iConfig.getParameter<bool>("useConstituentSubtraction");
  csRho_EtaMax_ = iConfig.getParameter<double>("csRho_EtaMax");
  csRParam_ = iConfig.getParameter<double>("csRParam");

  useSoftDrop_ = iConfig.getParameter<bool>("useSoftDrop");
  zCut_ = iConfig.getParameter<double>("zcut");
  beta_ = iConfig.getParameter<double>("beta");
  R0_ = iConfig.getParameter<double>("R0");

  correctShape_ = iConfig.getParameter<bool>("correctShape");
  gridMaxRapidity_ = iConfig.getParameter<double>("gridMaxRapidity");
  gridSpacing_ = iConfig.getParameter<double>("gridSpacing");

  input_chrefcand_token_ =
      consumes<edm::View<reco::RecoChargedRefCandidate>>(iConfig.getParameter<edm::InputTag>("src"));

  if (useFiltering_ || useTrimming_ || usePruning_ || useMassDropTagger_ || useCMSBoostedTauSeedingAlgorithm_ ||
      useConstituentSubtraction_ || useSoftDrop_ || correctShape_)
    useExplicitGhosts_ = true;

  ////// adding exceptions

  if ((useMassDropTagger_) && ((muCut_ == -1) || (yCut_ == -1)))
    throw cms::Exception("useMassDropTagger")
        << "Parameters muCut and/or yCut for Mass Drop are not defined." << std::endl;

  if ((useFiltering_) && ((rFilt_ == -1) || (nFilt_ == -1))) {
    throw cms::Exception("useFiltering") << "Parameters rFilt and/or nFilt for Filtering are not defined." << std::endl;
    if ((useDynamicFiltering_) && (rFiltFactor_ == -1))
      throw cms::Exception("useDynamicFiltering")
          << "Parameters rFiltFactor for DynamicFiltering is not defined." << std::endl;
  }

  if ((useTrimming_) && ((rFilt_ == -1) || (trimPtFracMin_ == -1)))
    throw cms::Exception("useTrimming") << "Parameters rFilt and/or trimPtFracMin for Trimming are not defined."
                                        << std::endl;

  if ((usePruning_) && ((zCut_ == -1) || (RcutFactor_ == -1) || (nFilt_ == -1)))
    throw cms::Exception("usePruning") << "Parameters zCut and/or RcutFactor and/or nFilt for Pruning are not defined."
                                       << std::endl;

  if ((useCMSBoostedTauSeedingAlgorithm_) &&
      ((subjetPtMin_ == -1) || (maxDepth_ == -1) || (muMin_ == -1) || (muMax_ == -1) || (yMin_ == -1) ||
       (yMax_ == -1) || (dRMin_ == -1) || (dRMax_ == -1)))
    throw cms::Exception("useCMSBoostedTauSeedingAlgorithm")
        << "Parameters subjetPtMin, muMin, muMax, yMin, yMax, dRmin, dRmax, maxDepth for CMSBoostedTauSeedingAlgorithm "
           "are not defined."
        << std::endl;

  if (useConstituentSubtraction_ && (fjAreaDefinition_.get() == nullptr))
    throw cms::Exception("AreaMustBeSet")
        << "Logic error. The area definition must be set if you use constituent subtraction." << std::endl;

  if ((useConstituentSubtraction_) && ((csRho_EtaMax_ == -1) || (csRParam_ == -1)))
    throw cms::Exception("useConstituentSubtraction")
        << "Parameters csRho_EtaMax and/or csRParam for ConstituentSubtraction are not defined." << std::endl;

  if (useSoftDrop_ && usePruning_)
    throw cms::Exception("PruningAndSoftDrop")
        << "Logic error. Soft drop is a generalized pruning, do not run them together." << std::endl;

  if ((useSoftDrop_) && ((zCut_ == -1) || (beta_ == -1) || (R0_ == -1)))
    throw cms::Exception("useSoftDrop") << "Parameters zCut and/or beta and/or R0 for SoftDrop are not defined."
                                        << std::endl;

  if ((correctShape_) && ((gridMaxRapidity_ == -1) || (gridSpacing_ == -1)))
    throw cms::Exception("correctShape")
        << "Parameters gridMaxRapidity and/or gridSpacing for SoftDrop are not defined." << std::endl;
}

//______________________________________________________________________________
FastjetJetProducer::~FastjetJetProducer() {}

////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

void FastjetJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // for everything but track jets
  if (!makeTrackJet(jetTypeE)) {
    // use the default production from one collection
    VirtualJetProducer::produce(iEvent, iSetup);

  } else {  // produce trackjets from tracks grouped per primary vertex

    produceTrackJets(iEvent, iSetup);
  }

  // fjClusterSeq_ retains quite a lot of memory - about 1 to 7Mb at 200 pileup
  // depending on the exact configuration; and there are 24 FastjetJetProducers in the
  // sequence so this adds up to about 60 Mb. It's allocated every time runAlgorithm
  // is called, so safe to delete here.
  fjClusterSeq_.reset();
}

void FastjetJetProducer::produceTrackJets(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // read in the track candidates
  edm::Handle<edm::View<reco::RecoChargedRefCandidate>> inputsHandle;
  iEvent.getByToken(input_chrefcand_token_, inputsHandle);

  // make collection with pointers so we can play around with it
  std::vector<edm::Ptr<reco::RecoChargedRefCandidate>> allInputs;
  std::vector<edm::Ptr<reco::Candidate>> origInputs;
  for (size_t i = 0; i < inputsHandle->size(); ++i) {
    allInputs.push_back(inputsHandle->ptrAt(i));
    origInputs.push_back(inputsHandle->ptrAt(i));
  }

  // read in the PV collection
  edm::Handle<reco::VertexCollection> pvCollection;
  iEvent.getByToken(input_vertex_token_, pvCollection);
  // define the overall output jet container
  auto jets = std::make_unique<std::vector<reco::TrackJet>>();

  // loop over the good vertices, clustering for each vertex separately
  for (reco::VertexCollection::const_iterator itVtx = pvCollection->begin(); itVtx != pvCollection->end(); ++itVtx) {
    if (itVtx->isFake() || itVtx->ndof() < minVtxNdof_ || fabs(itVtx->z()) > maxVtxZ_)
      continue;

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
        for (std::vector<edm::Ptr<reco::RecoChargedRefCandidate>>::iterator itIn = allInputs.begin();
             itIn != allInputs.end();
             ++itIn) {
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
          }  // end if match found
        }    // end loop over input tracks
        // give an info message in case no match is found (can happen if candidates are subset of tracks used for clustering)
        if (!found)
          edm::LogInfo("FastjetTrackJetProducer")
              << "Ignoring a track at vertex which is not in input track collection!";
      }  // end loop over tracks associated to vertex
      // if all inpt track candidates should be used
    } else {
      // loop over input track candidates
      for (std::vector<edm::Ptr<reco::RecoChargedRefCandidate>>::iterator itIn = allInputs.begin();
           itIn != allInputs.end();
           ++itIn) {
        // check if the track is close enough to the vertex
        float dz = (*itIn)->track()->dz(itVtx->position());
        float dxy = (*itIn)->track()->dxy(itVtx->position());
        if (fabs(dz) > dzTrVtxMax_)
          continue;
        if (fabs(dxy) > dxyTrVtxMax_)
          continue;
        bool closervtx = false;
        // now loop over the good vertices a second time
        for (reco::VertexCollection::const_iterator itVtx2 = pvCollection->begin(); itVtx2 != pvCollection->end();
             ++itVtx2) {
          if (itVtx->isFake() || itVtx->ndof() < minVtxNdof_ || fabs(itVtx->z()) > maxVtxZ_)
            continue;
          // and check this track is closer to any other vertex (if more than 1 vertex considered)
          if (!useOnlyOnePV_ && itVtx != itVtx2 && fabs((*itIn)->track()->dz(itVtx2->position())) < fabs(dz)) {
            closervtx = true;
            break;  // 1 closer vertex makes the track already not matched, so break
          }
        }
        // don't add this track if another vertex is found closer
        if (closervtx)
          continue;
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
    for (unsigned int ijet = 0; ijet < fjJets_.size(); ++ijet) {
      // get the constituents from fastjet
      std::vector<fastjet::PseudoJet> fjConstituents = sorted_by_pt(fjClusterSeq_->constituents(fjJets_[ijet]));
      // convert them to CandidatePtr vector
      std::vector<reco::CandidatePtr> constituents = getConstituents(fjConstituents);
      // fill the trackjet
      reco::TrackJet jet;
      // write the specifics to the jet (simultaneously sets 4-vector, vertex).
      writeSpecific(
          jet,
          reco::Particle::LorentzVector(fjJets_[ijet].px(), fjJets_[ijet].py(), fjJets_[ijet].pz(), fjJets_[ijet].E()),
          vertex_,
          constituents,
          iSetup);
      jet.setJetArea(0);
      jet.setPileup(0);
      jet.setPrimaryVertex(edm::Ref<reco::VertexCollection>(pvCollection, (int)(itVtx - pvCollection->begin())));
      jet.setVertex(itVtx->position());
      jets->push_back(jet);
    }

    if (useOnlyOnePV_)
      break;  // stop vertex loop if only one vertex asked for
  }           // end loop over vertices

  // put the jets in the collection
  LogDebug("FastjetTrackJetProducer") << "Put " << jets->size() << " jets in the event.\n";
  iEvent.put(std::move(jets));

  // Clear the work vectors so that memory is free for other modules.
  // Use the trick of swapping with an empty vector so that the memory
  // is actually given back rather than silently kept.
  decltype(fjInputs_)().swap(fjInputs_);
  decltype(fjJets_)().swap(fjJets_);
  decltype(inputs_)().swap(inputs_);
}

//______________________________________________________________________________
void FastjetJetProducer::runAlgorithm(edm::Event& iEvent, edm::EventSetup const& iSetup) {
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

  if (!doAreaFastjet_ && !doRhoFastjet_) {
    fjClusterSeq_ = std::make_shared<fastjet::ClusterSequence>(fjInputs_, *fjJetDefinition_);
  } else if (voronoiRfact_ <= 0) {
    fjClusterSeq_ =
        ClusterSequencePtr(new fastjet::ClusterSequenceArea(fjInputs_, *fjJetDefinition_, *fjAreaDefinition_));
  } else {
    fjClusterSeq_ = ClusterSequencePtr(
        new fastjet::ClusterSequenceVoronoiArea(fjInputs_, *fjJetDefinition_, fastjet::VoronoiAreaSpec(voronoiRfact_)));
  }

  if (!(useMassDropTagger_ || useCMSBoostedTauSeedingAlgorithm_ || useTrimming_ || useFiltering_ || usePruning_ ||
        useSoftDrop_ || useConstituentSubtraction_)) {
    fjJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));
  } else {
    fjJets_.clear();

    transformer_coll transformers;

    std::vector<fastjet::PseudoJet> tempJets = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));

    unique_ptr<fastjet::JetMedianBackgroundEstimator> bge_rho;
    if (useConstituentSubtraction_) {
      fastjet::Selector rho_range = fastjet::SelectorAbsRapMax(csRho_EtaMax_);
      bge_rho = std::make_unique<fastjet::JetMedianBackgroundEstimator>(
          rho_range, fastjet::JetDefinition(fastjet::kt_algorithm, csRParam_), *fjAreaDefinition_);
      bge_rho->set_particles(fjInputs_);
      fastjet::contrib::ConstituentSubtractor* constituentSubtractor =
          new fastjet::contrib::ConstituentSubtractor(bge_rho.get());

      transformers.push_back(transformer_ptr(constituentSubtractor));
    };
    if (useMassDropTagger_) {
      fastjet::MassDropTagger* md_tagger = new fastjet::MassDropTagger(muCut_, yCut_);
      transformers.push_back(transformer_ptr(md_tagger));
    }
    if (useCMSBoostedTauSeedingAlgorithm_) {
      fastjet::contrib::CMSBoostedTauSeedingAlgorithm* tau_tagger = new fastjet::contrib::CMSBoostedTauSeedingAlgorithm(
          subjetPtMin_, muMin_, muMax_, yMin_, yMax_, dRMin_, dRMax_, maxDepth_, verbosity_);
      transformers.push_back(transformer_ptr(tau_tagger));
    }
    if (useTrimming_) {
      fastjet::Filter* trimmer = new fastjet::Filter(fastjet::JetDefinition(fastjet::kt_algorithm, rFilt_),
                                                     fastjet::SelectorPtFractionMin(trimPtFracMin_));
      transformers.push_back(transformer_ptr(trimmer));
    }
    if ((useFiltering_) && (!useDynamicFiltering_)) {
      fastjet::Filter* filter = new fastjet::Filter(fastjet::JetDefinition(fastjet::cambridge_algorithm, rFilt_),
                                                    fastjet::SelectorNHardest(nFilt_));
      transformers.push_back(transformer_ptr(filter));
    }

    if ((usePruning_) && (!useKtPruning_)) {
      fastjet::Pruner* pruner = new fastjet::Pruner(fastjet::cambridge_algorithm, zCut_, RcutFactor_);
      transformers.push_back(transformer_ptr(pruner));
    }

    if (useDynamicFiltering_) {
      fastjet::Filter* filter =
          new fastjet::Filter(fastjet::Filter(&*rFiltDynamic_, fastjet::SelectorNHardest(nFilt_)));
      transformers.push_back(transformer_ptr(filter));
    }

    if (useKtPruning_) {
      fastjet::Pruner* pruner = new fastjet::Pruner(fastjet::kt_algorithm, zCut_, RcutFactor_);
      transformers.push_back(transformer_ptr(pruner));
    }

    if (useSoftDrop_) {
      fastjet::contrib::SoftDrop* sd = new fastjet::contrib::SoftDrop(beta_, zCut_, R0_);
      transformers.push_back(transformer_ptr(sd));
    }

    unique_ptr<fastjet::Subtractor> subtractor;
    unique_ptr<fastjet::GridMedianBackgroundEstimator> bge_rho_grid;
    if (correctShape_) {
      bge_rho_grid = std::make_unique<fastjet::GridMedianBackgroundEstimator>(
          gridMaxRapidity_, gridSpacing_);
      bge_rho_grid->set_particles(fjInputs_);
      subtractor = std::make_unique<fastjet::Subtractor>(bge_rho_grid.get());
      subtractor->set_use_rho_m();
      //subtractor->use_common_bge_for_rho_and_rhom(true);
    }

    for (std::vector<fastjet::PseudoJet>::const_iterator ijet = tempJets.begin(), ijetEnd = tempJets.end();
         ijet != ijetEnd;
         ++ijet) {
      fastjet::PseudoJet transformedJet = *ijet;
      bool passed = true;
      for (transformer_coll::const_iterator itransf = transformers.begin(), itransfEnd = transformers.end();
           itransf != itransfEnd;
           ++itransf) {
        if (transformedJet != 0) {
          transformedJet = (**itransf)(transformedJet);
        } else {
          passed = false;
        }
      }

      if (correctShape_) {
        transformedJet = (*subtractor)(transformedJet);
      }

      if (passed) {
        fjJets_.push_back(transformedJet);
      }
    }
  }
}

//______________________________________________________________________________
void FastjetJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription descFastjetJetProducer;
  ////// From FastjetJetProducer
  fillDescriptionsFromFastJetProducer(descFastjetJetProducer);
  ///// From VirtualJetProducer
  VirtualJetProducer::fillDescriptionsFromVirtualJetProducer(descFastjetJetProducer);
  ////
  descFastjetJetProducer.add<string>("jetCollInstanceName", "");
  descFastjetJetProducer.add<bool>("sumRecHits", false);

  /////////////////////
  descriptions.add("FastjetJetProducer", descFastjetJetProducer);
}

void FastjetJetProducer::fillDescriptionsFromFastJetProducer(edm::ParameterSetDescription& desc) {
  desc.add<bool>("useMassDropTagger", false);
  desc.add<bool>("useFiltering", false);
  desc.add<bool>("useDynamicFiltering", false);
  desc.add<bool>("useTrimming", false);
  desc.add<bool>("usePruning", false);
  desc.add<bool>("useCMSBoostedTauSeedingAlgorithm", false);
  desc.add<bool>("useKtPruning", false);
  desc.add<bool>("useConstituentSubtraction", false);
  desc.add<bool>("useSoftDrop", false);
  desc.add<bool>("correctShape", false);
  desc.add<bool>("UseOnlyVertexTracks", false);
  desc.add<bool>("UseOnlyOnePV", false);
  desc.add<double>("muCut", -1.0);
  desc.add<double>("yCut", -1.0);
  desc.add<double>("rFilt", -1.0);
  desc.add<double>("rFiltFactor", -1.0);
  desc.add<double>("trimPtFracMin", -1.0);
  desc.add<double>("zcut", -1.0);
  desc.add<double>("rcut_factor", -1.0);
  desc.add<double>("csRho_EtaMax", -1.0);
  desc.add<double>("csRParam", -1.0);
  desc.add<double>("beta", -1.0);
  desc.add<double>("R0", -1.0);
  desc.add<double>("gridMaxRapidity", -1.0);  // For fixed-grid rho
  desc.add<double>("gridSpacing", -1.0);      // For fixed-grid rho
  desc.add<double>("DzTrVtxMax", 999999.);
  desc.add<double>("DxyTrVtxMax", 999999.);
  desc.add<double>("MaxVtxZ", 15.0);
  desc.add<double>("subjetPtMin", -1.0);
  desc.add<double>("muMin", -1.0);
  desc.add<double>("muMax", -1.0);
  desc.add<double>("yMin", -1.0);
  desc.add<double>("yMax", -1.0);
  desc.add<double>("dRMin", -1.0);
  desc.add<double>("dRMax", -1.0);
  desc.add<int>("maxDepth", -1);
  desc.add<int>("nFilt", -1);
  desc.add<int>("MinVtxNdof", 5);
}

////////////////////////////////////////////////////////////////////////////////
// define as cmssw plugin
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(FastjetJetProducer);

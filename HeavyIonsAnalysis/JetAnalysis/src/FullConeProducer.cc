#ifndef RecoHIJets_JetProducers_plugins_FullConeProducer_h
#define RecoHIJets_JetProducers_plugins_FullConeProducer_h

#include "MyVirtualJetProducer.h"

class FullConeProducer : public MyVirtualJetProducer
{

public:
  //
  // construction/destruction
  //
  explicit FullConeProducer(const edm::ParameterSet& iConfig);
  virtual ~FullConeProducer();

protected:

  //
  // member functions
  //

  virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );
  virtual void output(  edm::Event & iEvent, edm::EventSetup const& iSetup );
  template< typename T >
  void writeBkgJets( edm::Event & iEvent, edm::EventSetup const& iSetup );

private:

  // trackjet clustering parameters
  bool useOnlyVertexTracks_;
  bool useOnlyOnePV_;
  float dzTrVtxMax_;

  int nFill_;
  float etaMax_;
  bool avoidNegative_;

  const CaloGeometry *geo;
};


#endif
////////////////////////////////////////////////////////////////////////////////
//
// FullConeProducer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

//#include "FullConeProducer.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

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
#include "DataFormats/Math/interface/deltaR.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "fastjet/SISConePlugin.hh"
#include "fastjet/CMSIterativeConePlugin.hh"
#include "fastjet/ATLASConePlugin.hh"
#include "fastjet/CDFMidPointPlugin.hh"

#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;
using namespace edm;

static const double pi = 3.14159265358979;

////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
FullConeProducer::FullConeProducer(const edm::ParameterSet& iConfig)
  : MyVirtualJetProducer( iConfig ),
    nFill_(5),
    etaMax_(3),
    geo(0)
{
  avoidNegative_  = iConfig.getParameter<bool>("avoidNegative");

  if ( iConfig.exists("UseOnlyVertexTracks") )
    useOnlyVertexTracks_ = iConfig.getParameter<bool>("UseOnlyVertexTracks");
  else
    useOnlyVertexTracks_ = false;

  if ( iConfig.exists("UseOnlyOnePV") )
    useOnlyOnePV_        = iConfig.getParameter<bool>("UseOnlyOnePV");
  else
    useOnlyOnePV_ = false;

  if ( iConfig.exists("DrTrVtxMax") )
    dzTrVtxMax_          = iConfig.getParameter<double>("DzTrVtxMax");
  else
    dzTrVtxMax_ = false;

  produces<std::vector<bool> >("directions");

}


//______________________________________________________________________________
FullConeProducer::~FullConeProducer()
{
}

void FullConeProducer::output(edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // Write jets and constitutents. Will use fjJets_, inputs_
  // and fjClusterSeq_
  switch( jetTypeE ) {
  case JetType::CaloJet :
    writeBkgJets<reco::CaloJet>( iEvent, iSetup);
    break;
  case JetType::PFJet :
    writeBkgJets<reco::PFJet>( iEvent, iSetup);
    break;
  case JetType::GenJet :
    writeBkgJets<reco::GenJet>( iEvent, iSetup);
    break;
  case JetType::TrackJet :
    writeBkgJets<reco::TrackJet>( iEvent, iSetup);
    break;
  case JetType::BasicJet :
    writeBkgJets<reco::BasicJet>( iEvent, iSetup);
    break;
  default:
    edm::LogError("InvalidInput") << " invalid jet type in MyVirtualJetProducer\n";
    break;
  };

}

template< typename T >
void FullConeProducer::writeBkgJets( edm::Event & iEvent, edm::EventSetup const& iSetup )
{
  // produce output jet collection

  using namespace reco;

  if(!geo){
    edm::ESHandle<CaloGeometry> pGeo;
    iSetup.get<CaloGeometryRecord>().get(pGeo);
    geo = pGeo.product();
  }

  std::vector<fastjet::PseudoJet> fjFakeJets_;
  std::vector<std::vector<reco::CandidatePtr> > constituents_;
  vector<double> et;
  vector<double> pileUp;
  std::auto_ptr<std::vector<bool> > directions(new std::vector<bool>());

  nFill_ = fjJets_.size();
  directions->reserve(nFill_);
  et.reserve(nFill_);
  pileUp.reserve(nFill_);

  fjFakeJets_.reserve(nFill_);
  constituents_.reserve(nFill_);

  constituents_.reserve(nFill_);
  for(int ijet = 0; ijet < nFill_; ++ijet){
    vector<reco::CandidatePtr> vec;
    constituents_.push_back(vec);
    directions->push_back(true);
  }

  for(int ijet = 0; ijet < nFill_; ++ijet){
    et[ijet] = 0;
    pileUp[ijet] = 0;
  }

  for (vector<fastjet::PseudoJet>::const_iterator input_object = fjInputs_.begin (),
	 fjInputsEnd = fjInputs_.end();
       input_object != fjInputsEnd; ++input_object) {

    const reco::CandidatePtr & tower=inputs_[input_object->user_index()];
    const CaloTower* ctc = dynamic_cast<const CaloTower*>(tower.get());
    int ieta = ctc->id().ieta();
    int iphi = ctc->id().iphi();
    CaloTowerDetId id(ieta,iphi);
    const GlobalPoint& hitpoint = geo->getPosition(id);
    double towEta = hitpoint.eta();
    double towPhi = hitpoint.phi();

    for (unsigned int ijet = 0; ijet < fjJets_.size(); ++ijet){
      fastjet::PseudoJet & jet = fjJets_[ijet];
      if(reco::deltaR(towEta,towPhi,jet.eta(),jet.phi()) > rParam_) continue;
      constituents_[ijet].push_back(tower);

      double towet = tower->et();
      double putow = subtractor_->getPileUpAtTower(tower);
      double etadd = towet - putow;
      if(avoidNegative_ && etadd < 0.) etadd = 0;
      et[ijet] += etadd;
      pileUp[ijet] += towet - etadd;
    }
  }

  //cout<<"Start filling jets"<<endl;

  for(int ir = 0; ir < nFill_; ++ir){
    if(et[ir] < 0){
      //cout<<"Flipping vector"<<endl;
      (*directions)[ir] = false;
      et[ir] = -et[ir];
    }else{
      //cout<<"Keep vector same"<<endl;
      (*directions)[ir] = true;
    }
    //cout<<"Lorentz"<<endl;

    math::PtEtaPhiMLorentzVector p(et[ir],fjJets_[ir].eta(),fjJets_[ir].phi(),0);
    fastjet::PseudoJet jet(p.px(),p.py(),p.pz(),p.energy());
    fjFakeJets_.push_back(jet);
  }

  std::auto_ptr<std::vector<T> > jets(new std::vector<T>() );
  jets->reserve(fjFakeJets_.size());

  for (unsigned int ijet=0;ijet<fjFakeJets_.size();++ijet) {
    // allocate this jet
    T jet;
    // get the fastjet jet
    const fastjet::PseudoJet& fjJet = fjFakeJets_[ijet];

    // convert them to CandidatePtr vector
    std::vector<CandidatePtr> constituents =
      constituents_[ijet];

    writeSpecific(jet,
		  Particle::LorentzVector(fjJet.px(),
					  fjJet.py(),
					  fjJet.pz(),
					  fjJet.E()),
		  vertex_,
		  constituents, iSetup);

    // calcuate the jet area
    double jetArea=0.0;
    jet.setJetArea (jetArea);
    if(doPUOffsetCorr_){
      jet.setPileup(pileUp[ijet]);
    }else{
      jet.setPileup (0.0);
    }

    // add to the list
    jets->push_back(jet);
  }

  // put the jets in the collection
  iEvent.put(jets);
  iEvent.put(directions,"directions");
}

void FullConeProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup){
  if ( !doAreaFastjet_ && !doRhoFastjet_) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, *fjJetDefinition_ ) );
  } else {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceArea( fjInputs_, *fjJetDefinition_ , *fjActiveArea_ ) );
  }
  fjJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));

}





////////////////////////////////////////////////////////////////////////////////
// define as cmssw plugin
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(FullConeProducer);

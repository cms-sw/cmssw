#ifndef RecoJets_JetProducers_plugins_JetAlgorithmAnalyzer_h
#define RecoJets_JetProducers_plugins_JetAlgorithmAnalyzer_h

//#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"
#include "MyVirtualJetProducer.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"
#include "TNtuple.h"
#include "TH2D.h"
#include <vector>
#include <iostream>
using namespace std;

static const int nSteps = 8;
static const double PI = 3.141592653589;

class JetAlgorithmAnalyzer : public MyVirtualJetProducer
{

public:
  //
  // construction/destruction
  //
  explicit JetAlgorithmAnalyzer(const edm::ParameterSet& iConfig);
  virtual ~JetAlgorithmAnalyzer();

protected:

  //
  // member functions
  //

  virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup );
  virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup );
  virtual void output(  edm::Event & iEvent, edm::EventSetup const& iSetup );
  template< typename T >
  void writeBkgJets( edm::Event & iEvent, edm::EventSetup const& iSetup );

  void fillNtuple(int output, const  std::vector<fastjet::PseudoJet>& jets, int step);
  void fillTowerNtuple(const  std::vector<fastjet::PseudoJet>& jets, int step);
  void fillJetNtuple(const  std::vector<fastjet::PseudoJet>& jets, int step);
  void fillBkgNtuple(const PileUpSubtractor* subtractor, int step);

private:

  // trackjet clustering parameters
  bool useOnlyVertexTracks_;
  bool useOnlyOnePV_;
  float dzTrVtxMax_;

  int evtPlaneIndex_;
  double phi0_;
  int nFill_;
  float etaMax_;
  int iev_;
  bool avoidNegative_;
  double cone_;

  bool doAnalysis_;

  bool backToBack_;
  bool doMC_;
  bool doRecoEvtPlane_;
  bool doRandomCones_;
  bool doFullCone_;

  bool sumRecHits_;

  double hf_;
  double sumET_;
  int bin_;

  int centBin_;
  edm::InputTag centTag_;
  edm::InputTag epTag_;
  edm::InputTag PatJetSrc_;

  TNtuple* ntTowers;
  TNtuple* ntJetTowers;

  TNtuple* ntTowersFromEvtPlane;

  TNtuple* ntJets;
  TNtuple* ntPU;
  TNtuple* ntRandom;
  TNtuple* ntRandom0;
  TNtuple* ntRandom1;

  edm::Handle<pat::JetCollection> patjets;

  std::vector<TH2D*> hTowers;
  std::vector<TH2D*> hJetTowers;

  std::vector<TH2D*> hTowersFromEvtPlane;

  std::vector<TH2D*> hJets;
  std::vector<TH1D*> hPU;
  std::vector<TH1D*> hMean;
  std::vector<TH1D*> hSigma;

  std::vector<TH2D*> hRandom;


  TH2D* hPTieta;
  TH1D* hMeanieta;
  TH1D* hRMSieta;
  TH1D* hPUieta;

  const CaloGeometry *geo;
  edm::Service<TFileService> f;
};


#endif
////////////////////////////////////////////////////////////////////////////////
//
// JetAlgorithmAnalyzer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

//#include "RecoJets/JetProducers/plugins/JetAlgorithmAnalyzer.h"
//#include "JetAlgorithmAnalyzer.h"

#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

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

#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "fastjet/SISConePlugin.hh"
#include "fastjet/CMSIterativeConePlugin.hh"
#include "fastjet/ATLASConePlugin.hh"
#include "fastjet/CDFMidPointPlugin.hh"

#include "CLHEP/Random/RandomEngine.h"

#include <iostream>
#include <memory>
#include <algorithm>
#include <limits>
#include <cmath>

using namespace std;
using namespace edm;

static const double pi = 3.14159265358979;

CLHEP::HepRandomEngine* randomEngine;
edm::Service<RandomNumberGenerator> rng;


////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
JetAlgorithmAnalyzer::JetAlgorithmAnalyzer(const edm::ParameterSet& iConfig)
  : MyVirtualJetProducer( iConfig ),
    phi0_(0),
    nFill_(5),
    etaMax_(3),
    iev_(0),
    cone_(1),
    geo(0)
{

  doAreaFastjet_ = false;
  doRhoFastjet_ = false;

  puSubtractorName_  =  iConfig.getParameter<string> ("subtractorName");
  subtractor_ =  boost::shared_ptr<PileUpSubtractor>(PileUpSubtractorFactory::get()->create( puSubtractorName_, iConfig, consumesCollector()));


  doAnalysis_  = iConfig.getUntrackedParameter<bool>("doAnalysis",true);
  doRandomCones_  = iConfig.getUntrackedParameter<bool>("doRandomCones",true);

  backToBack_ = iConfig.getUntrackedParameter<bool>("doBackToBack",false);
  if(backToBack_) nFill_ = 2;

  doFullCone_  = iConfig.getUntrackedParameter<bool>("doFullCone",true);

  doMC_  = iConfig.getUntrackedParameter<bool>("doMC",true);
  doRecoEvtPlane_  = iConfig.getUntrackedParameter<bool>("doRecoEvtPlane",true);
  evtPlaneIndex_ = iConfig.getUntrackedParameter<int>("evtPlaneIndex",31);

  sumRecHits_  = iConfig.getParameter<bool>("sumRecHits");

  epTag_  = iConfig.getParameter<InputTag>("evtPlaneTag");
  PatJetSrc_ = iConfig.getUntrackedParameter<edm::InputTag>("patJetSrc",edm::InputTag("icPu5patJets"));

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

  produces<reco::CaloJetCollection>("randomCones");
  produces<std::vector<bool> >("directions");

  if(backToBack_){
    ntRandom0 = f->make<TNtuple>("ntRandom0","Algorithm Analysis Background","eta:phi:phiRel:et:had:em:pu:mean:rms:bin:hf:sumET:event:dR:matchedJetPt:evtJetPt");
    ntRandom1 = f->make<TNtuple>("ntRandom1","Algorithm Analysis Background","eta:phi:phiRel:et:had:em:pu:mean:rms:bin:hf:sumET:event:dR:matchedJetPt:evtJetPt");
  }else{
    ntRandom = f->make<TNtuple>("ntRandom","Algorithm Analysis Background","eta:phi:phiRel:et:had:em:pu:mean:rms:bin:hf:sumET:event:dR:matchedJetPt:evtJetPt");
  }

  if(doAnalysis_){
    ntTowers = f->make<TNtuple>("ntTowers","Algorithm Analysis Towers","eta:phi:et:step:event");
    ntTowersFromEvtPlane = f->make<TNtuple>("ntTowersFromEvtPlane","Algorithm Analysis Towers","eta:phi:et:step:event");
    ntJetTowers = f->make<TNtuple>("ntTowers","Algorithm Analysis Towers","eta:phi:et:step:event");

    ntJets = f->make<TNtuple>("ntJets","Algorithm Analysis Jets","eta:phi:et:step:event");
    ntPU = f->make<TNtuple>("ntPU","Algorithm Analysis Background","eta:mean:sigma:step:event");
    ntuple = f->make<TNtuple>("nt","debug","ieta:eta:iphi:phi:pt:em:had");

    hPTieta = f->make<TH2D>("hPTieta","hPTieta",23,-11.5,11.5,200,0,10);
    hRMSieta = f->make<TH1D>("hRMSieta","hPTieta",23,-11.5,11.5);
    hMeanieta = f->make<TH1D>("hMeanieta","hPTieta",23,-11.5,11.5);
    hPUieta = f->make<TH1D>("hPUieta","hPTieta",23,-11.5,11.5);

    for(int i = 0; i < nSteps; ++i){
      hTowers.push_back(f->make<TH2D>(Form("hTowers_step%d",i),"",200,-5.5,5.5,200,-0.02,6.3));
      hJets.push_back(f->make<TH2D>(Form("hJets_step%d",i),"",200,-5.5,5.5,200,-0.02,6.3));
      hTowersFromEvtPlane.push_back(f->make<TH2D>(Form("hTowersFromEvtPlane_step%d",i),"",200,-5.5,5.5,200,-0.02,6.3));

      hJetTowers.push_back(f->make<TH2D>(Form("hJetTowers_step%d",i),"",200,-5.5,5.5,200,-0.02,6.3));

      hPU.push_back(f->make<TH1D>(Form("hPU_step%d",i),"",200,-5.5,5.5));
      hMean.push_back(f->make<TH1D>(Form("hMean_step%d",i),"",200,-5.5,5.5));
      hSigma.push_back(f->make<TH1D>(Form("hSigma_step%d",i),"",200,-5.5,5.5));

    }

  }

}


//______________________________________________________________________________
JetAlgorithmAnalyzer::~JetAlgorithmAnalyzer()
{
}

void JetAlgorithmAnalyzer::fillNtuple(int output, const  std::vector<fastjet::PseudoJet>& jets, int step){
  if(!doAnalysis_) return;

  TNtuple* nt;
  TH2D* h;

  if(output == 1){
    nt = ntJets;
    h = hJets[step];
  }
  if(output == 0){
    nt = ntTowers;
    h = hTowers[step];
  }
  if(output == 2){
    nt = ntTowersFromEvtPlane;
    h = hTowersFromEvtPlane[step];
  }
  if(output == 3){
    nt = ntJetTowers;
    h = hJetTowers[step];
  }
  else{
    nt = 0;
    h = 0;
  }

  //bool printDebug = 0;

  double totet = 0;
  int ntow = 0;
  //int nj = jets.size();

  //if(printDebug){
  //   cout<<"step : "<<step<<endl;
  //   cout<<"Size of input : "<<nj<<endl;
  //}
  for(unsigned int i = 0; i < jets.size(); ++i){
    const fastjet::PseudoJet& jet = jets[i];

    double pt = jet.perp();
    int ieta = -99;
    if(output != 1){
      reco::CandidatePtr const & itow =  inputs_[ jet.user_index() ];
      pt =  itow->et();
      ieta = subtractor_->ieta(itow);
    }

    if(output == 0 && step == 6){
      hPTieta->Fill(ieta,pt);
    }

    double phi = jet.phi();
    if(output == 2){
      phi = phi - phi0_;
      if(phi < 0) phi += 2*PI;
      if(phi > 2*PI) phi -= 2*PI;
    }

    double eta = jet.eta();
    if(eta > 0 && eta < 0.08){
      //     if(fabs(eta) < 1.){
      totet += pt;
      ntow++;
    }

    nt->Fill(jet.eta(),phi,pt,step,iev_);
    h->Fill(jet.eta(),phi,pt);
  }
  // if(printDebug && 0){
  // cout<<"-----------------------------"<<endl;
  // cout<<"STEP             = "<<step<<endl;
  // cout<<"Total ET         = "<<totet<<endl;
  // cout<<"Towers counted   = "<<ntow<<endl;
  // cout<<"Average tower ET = "<<totet/ntow<<endl;
  // cout<<"-----------------------------"<<endl;
  // }
}


void JetAlgorithmAnalyzer::fillTowerNtuple(const  std::vector<fastjet::PseudoJet>& jets, int step){
  fillNtuple(0,jets,step);
  fillNtuple(2,jets,step);
}

void JetAlgorithmAnalyzer::fillJetNtuple(const  std::vector<fastjet::PseudoJet>& jets, int step){
  fillNtuple(1,jets,step);

  for(unsigned int i = 0; i < jets.size(); ++i){
    const fastjet::PseudoJet& jet = jets[i];
    std::vector<fastjet::PseudoJet> fjConstituents = sorted_by_pt(fjClusterSeq_->constituents(jet));




    fillNtuple(3,fjConstituents,step);
  }
}

void JetAlgorithmAnalyzer::fillBkgNtuple(const PileUpSubtractor* subtractor, int step){
  if(!doAnalysis_) return;
  CaloTowerCollection col;
  for(int ieta = -29; ieta < 30; ++ieta){
    if(ieta == 0) continue;
    CaloTowerDetId id(ieta,1);
    const GlobalPoint& hitpoint = geo->getPosition(id);
    //  cout<<"iETA "<<ieta<<endl;
    double eta = hitpoint.eta();
    //  cout<<"eta "<<eta<<endl;
    math::PtEtaPhiMLorentzVector p4(1,eta,1.,1.);
    GlobalPoint pos(1,1,1);
    CaloTower c(id,1.,1.,1.,1,1, p4, pos,pos);
    col.push_back(c);
    reco::CandidatePtr ptr(&col,col.size() - 1);
    double mean = subtractor->getMeanAtTower(ptr);
    double sigma = subtractor->getSigmaAtTower(ptr);
    double pu = subtractor->getPileUpAtTower(ptr);
    ntPU->Fill(eta,mean,sigma,step,iev_);
    hPU[step]->Fill(eta,pu);
    hMean[step]->Fill(eta,mean);
    hSigma[step]->Fill(eta,sigma);

    if(step == 7){
      hPUieta->Fill(ieta,pu);
      hMeanieta->Fill(ieta,mean);
      hRMSieta->Fill(ieta,sigma);
    }
  }
}

void JetAlgorithmAnalyzer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  phi0_ = 0;

  if(!geo){
    edm::ESHandle<CaloGeometry> pGeo;
    iSetup.get<CaloGeometryRecord>().get(pGeo);
    geo = pGeo.product();
  }

  iEvent.getByLabel(PatJetSrc_,patjets);

  //   cout<<("VirtualJetProducer") << "Entered produce\n";
  //determine signal vertex
  vertex_=reco::Jet::Point(0,0,0);
  if (makeCaloJet(jetTypeE)&&doPVCorrection_) {
    //     cout<<("VirtualJetProducer") << "Adding PV info\n";
    edm::Handle<reco::VertexCollection> pvCollection;
    iEvent.getByLabel(srcPVs_,pvCollection);
    if (pvCollection->size()>0) vertex_=pvCollection->begin()->position();
  }

  // For Pileup subtraction using offset correction:
  // set up geometry map
  if ( doPUOffsetCorr_ ) {
    //      cout<<"setting up ... "<<endl;
    subtractor_->setupGeometryMap(iEvent, iSetup);
  }

  // clear data
  //   cout<<("VirtualJetProducer") << "Clear data\n";
  fjInputs_.clear();
  fjJets_.clear();
  inputs_.clear();

  if(doRecoEvtPlane_){
    Handle<reco::EvtPlaneCollection> planes;
    iEvent.getByLabel(epTag_,planes);
    phi0_  = (*planes)[evtPlaneIndex_].angle();
  }


  // get inputs and convert them to the fastjet format (fastjet::PeudoJet)
  edm::Handle<reco::CandidateView> inputsHandle;
  iEvent.getByLabel(src_,inputsHandle);
  for (size_t i = 0; i < inputsHandle->size(); ++i) {
    inputs_.push_back(inputsHandle->ptrAt(i));
  }
  //   cout<<("VirtualJetProducer") << "Got inputs\n";

  // Convert candidates to fastjet::PseudoJets.
  // Also correct to Primary Vertex. Will modify fjInputs_
  // and use inputs_
  fjInputs_.reserve(inputs_.size());
  inputTowers();
  //   cout<<("VirtualJetProducer") << "Inputted towers\n";

  fillTowerNtuple(fjInputs_,0);
  fillBkgNtuple(subtractor_.get(),0);

  // For Pileup subtraction using offset correction:
  // Subtract pedestal.

  if ( doPUOffsetCorr_ ) {
    subtractor_->setDefinition(fjJetDefinition_);
    subtractor_->reset(inputs_,fjInputs_,fjJets_);
    subtractor_->calculatePedestal(fjInputs_);

    fillTowerNtuple(fjInputs_,1);
    fillBkgNtuple(subtractor_.get(),1);
    subtractor_->subtractPedestal(fjInputs_);

    fillTowerNtuple(fjInputs_,2);
    fillBkgNtuple(subtractor_.get(),2);

    //      cout<<("VirtualJetProducer") << "Subtracted pedestal\n";
  }

  // Run algorithm. Will modify fjJets_ and allocate fjClusterSeq_.
  // This will use fjInputs_
  runAlgorithm( iEvent, iSetup );

  fillTowerNtuple(fjInputs_,3);
  fillBkgNtuple(subtractor_.get(),3);
  //   fillJetNtuple(fjJets_,3);

///   if ( doPUOffsetCorr_ ) {
///      subtractor_->setAlgorithm(fjClusterSeq_);
///   }

//   cout<<("VirtualJetProducer") << "Ran algorithm\n";

  // For Pileup subtraction using offset correction:
  // Now we find jets and need to recalculate their energy,
  // mark towers participated in jet,
  // remove occupied towers from the list and recalculate mean and sigma
  // put the initial towers collection to the jet,
  // and subtract from initial towers in jet recalculated mean and sigma of towers
  if ( doPUOffsetCorr_ ) {
    vector<fastjet::PseudoJet> orphanInput;
    subtractor_->calculateOrphanInput(orphanInput);
    fillTowerNtuple(orphanInput,4);
    fillBkgNtuple(subtractor_.get(),4);
    //      fillJetNtuple(fjJets_,4);

    //only the id's of the orphan input are used, not their energy
    subtractor_->calculatePedestal(orphanInput);
    fillTowerNtuple(orphanInput,5);
    fillBkgNtuple(subtractor_.get(),5);
    //      fillJetNtuple(fjJets_,5);

    subtractor_->offsetCorrectJets();
    fillTowerNtuple(orphanInput,6);
    fillBkgNtuple(subtractor_.get(),6);
    //      fillJetNtuple(fjJets_,6);

  }

  // Write the output jets.
  // This will (by default) call the member function template
  // "writeJets", but can be overridden.
  // this will use inputs_
  output( iEvent, iSetup );
  fillBkgNtuple(subtractor_.get(),7);
  //  fillJetNtuple(fjJets_,7);

  //   cout<<("VirtualJetProducer") << "Wrote jets\n";

  ++iev_;

  doAnalysis_ = false;
  return;
}

void JetAlgorithmAnalyzer::output(edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // Write jets and constitutents. Will use fjJets_, inputs_
  // and fjClusterSeq_

  //  cout<<"output running "<<endl;
  //  return;

  switch( jetTypeE ) {
  case JetType::CaloJet :
    //      writeJets<reco::CaloJet>( iEvent, iSetup);
    writeBkgJets<reco::CaloJet>( iEvent, iSetup);
    break;
  case JetType::PFJet :
    //     writeJets<reco::PFJet>( iEvent, iSetup);
    writeBkgJets<reco::PFJet>( iEvent, iSetup);
    break;
  case JetType::GenJet :
    //     writeJets<reco::GenJet>( iEvent, iSetup);
    writeBkgJets<reco::GenJet>( iEvent, iSetup);
    break;
  case JetType::TrackJet :
    //     writeJets<reco::TrackJet>( iEvent, iSetup);
    writeBkgJets<reco::TrackJet>( iEvent, iSetup);
    break;
  case JetType::BasicJet :
    //     writeJets<reco::BasicJet>( iEvent, iSetup);
    writeBkgJets<reco::BasicJet>( iEvent, iSetup);
    break;
  default:
    edm::LogError("InvalidInput") << " invalid jet type in VirtualJetProducer\n";
    break;
  };

}

template< typename T >
void JetAlgorithmAnalyzer::writeBkgJets( edm::Event & iEvent, edm::EventSetup const& iSetup )
{
  // produce output jet collection

  //  cout<<"Started the Random Cones"<<endl;


  using namespace reco;

  std::vector<fastjet::PseudoJet> fjFakeJets_;
  std::vector<std::vector<reco::CandidatePtr> > constituents_;
  vector<double> phiRandom;
  vector<double> etaRandom;
  vector<double> et;
  vector<double> had;
  vector<double> em;
  vector<double> pileUp;
  vector<double> mean;
  vector<double> rms;
  vector<double> rawJetPt;
  vector<double> dr;
  vector<bool> matched;


  std::auto_ptr<std::vector<bool> > directions(new std::vector<bool>());
  directions->reserve(nFill_);

  phiRandom.reserve(nFill_);
  etaRandom.reserve(nFill_);
  et.reserve(nFill_);
  pileUp.reserve(nFill_);
  rms.reserve(nFill_);
  mean.reserve(nFill_);
  em.reserve(nFill_);
  had.reserve(nFill_);
  matched.reserve(nFill_);
  rawJetPt.reserve(nFill_);
  dr.reserve(nFill_);

  double evtJetPt = 0;
  randomEngine = &(rng->getEngine(iEvent.streamID()));

  for(unsigned int j = 0 ; j < patjets->size(); ++j){
    const pat::Jet& jet = (*patjets)[j];
    if(jet.pt() > evtJetPt){
      evtJetPt = jet.pt();
    }
  }

  fjFakeJets_.reserve(nFill_);
  constituents_.reserve(nFill_);

  for(int ijet = 0; ijet < nFill_; ++ijet){
    vector<reco::CandidatePtr> vec;
    constituents_.push_back(vec);
    directions->push_back(true);
  }

  for(int ijet = 0; ijet < nFill_; ++ijet){
    phiRandom[ijet] = 2*pi*randomEngine->flat() - pi;
    etaRandom[ijet] = 2*etaMax_*randomEngine->flat() - etaMax_;
    if(ijet>0 && backToBack_) phiRandom[ijet] = - phiRandom[0];
    et[ijet] = 0;
    had[ijet] = 0;
    em[ijet] = 0;
    pileUp[ijet] = 0;
    mean[ijet]=0;
    rms[ijet]=0;
    rawJetPt[ijet]=-99;
    dr[ijet]=-99;
    matched[ijet]=false;
  }


  for(unsigned int iy = 0; iy < inputs_.size(); ++iy){

    const reco::CandidatePtr & tower=inputs_[iy];
    const CaloTower* ctc = dynamic_cast<const CaloTower*>(tower.get());

    int ieta = ctc->id().ieta();
    int iphi = ctc->id().iphi();
    CaloTowerDetId id(ieta,iphi);
    const GlobalPoint& hitpoint = geo->getPosition(id);
    double towEta = hitpoint.eta();
    double towPhi = hitpoint.phi();

    for(int ir = 0; ir < nFill_; ++ir){
      if(reco::deltaR(towEta,towPhi,etaRandom[ir],phiRandom[ir]) > rParam_) continue;

      constituents_[ir].push_back(tower);

      if(sumRecHits_){
	const GlobalPoint& pos=geo->getPosition(ctc->id());
	//double energy = ctc->emEnergy() + ctc->hadEnergy();
	double ang = sin(pos.theta());
	// towet = energy*ang;
	em[ir] += ctc->emEnergy()*ang;
	had[ir] += ctc->hadEnergy()*ang;
      }

      for(unsigned int j = 0 ; j < patjets->size(); ++j){
	const pat::Jet& jet = (*patjets)[j];
	double thisdr = reco::deltaR(etaRandom[ir],phiRandom[ir],jet.eta(),jet.phi());
	if(thisdr < cone_  && jet.pt() > rawJetPt[ir]){
	  dr[ir] = thisdr;
	  rawJetPt[ir] = jet.pt();
	  matched[ir] = true;
	}
      }

      double towet = tower->et();
      double putow = subtractor_->getPileUpAtTower(tower);
      double etadd = towet - putow;
      if(avoidNegative_ && etadd < 0.) etadd = 0;
      et[ir] += etadd;
      pileUp[ir] += towet - etadd;
      mean[ir] += subtractor_->getMeanAtTower(tower);
      rms[ir] += subtractor_->getSigmaAtTower(tower);

    }
  }
  //   cout<<"Start filling jets"<<endl;

  if(backToBack_){
    int ir = 0;
    double phiRel = reco::deltaPhi(phiRandom[ir],phi0_);
    ///float entry0[100] = {etaRandom[ir],phiRandom[ir],phiRel,et[ir],had[ir],em[ir],pileUp[ir],mean[ir],rms[ir],bin_,hf_,sumET_,iev_,dr[ir],rawJetPt[ir],evtJetPt};
    Float_t entry0[100];
    entry0[0]=etaRandom[ir];
    entry0[1]=phiRandom[ir];
    entry0[2]=phiRel;
    entry0[3]=et[ir];
    entry0[4]=had[ir];
    entry0[5]=em[ir];
    entry0[6]=pileUp[ir];
    entry0[7]=mean[ir];
    entry0[8]=rms[ir];
    entry0[9]=bin_;
    entry0[10]=hf_;
    entry0[11]=sumET_;
    entry0[12]=iev_;
    entry0[13]=dr[ir];
    entry0[14]=rawJetPt[ir];
    entry0[15]=evtJetPt;
    //Float_t entry0[100] = {etaRandom[ir],phiRandom[ir],phiRel,et[ir],had[ir],em[ir],pileUp[ir],mean[ir],rms[ir],bin_,hf_,sumET_,iev_,dr[ir],rawJetPt[ir],evtJetPt};
    ntRandom0->Fill(entry0);
    ir = 1;
    phiRel = reco::deltaPhi(phiRandom[ir],phi0_);
    ///float entry1[100] = {etaRandom[ir],phiRandom[ir],phiRel,et[ir],had[ir],em[ir],pileUp[ir],mean[ir],rms[ir],bin_,hf_,sumET_,iev_,dr[ir],rawJetPt[ir],evtJetPt};
    Float_t entry1[100];
    entry1[0]=etaRandom[ir];
    entry1[1]=phiRandom[ir];
    entry1[2]=phiRel;
    entry1[3]=et[ir];
    entry1[4]=had[ir];
    entry1[5]=em[ir];
    entry1[6]=pileUp[ir];
    entry1[7]=mean[ir];
    entry1[8]=rms[ir];
    entry1[9]=bin_;
    entry1[10]=hf_;
    entry1[11]=sumET_;
    entry1[12]=iev_;
    entry1[13]=dr[ir];
    entry1[14]=rawJetPt[ir];
    entry1[15]=evtJetPt;
    //Float_t entry1[100] = {etaRandom[ir],phiRandom[ir],phiRel,et[ir],had[ir],em[ir],pileUp[ir],mean[ir],rms[ir],bin_,hf_,sumET_,iev_,dr[ir],rawJetPt[ir],evtJetPt};
    ntRandom1->Fill(entry1);
  }

  for(int ir = 0; ir < nFill_; ++ir){
    double phiRel = reco::deltaPhi(phiRandom[ir],phi0_);
    //float entry[100] = {etaRandom[ir],phiRandom[ir],phiRel,et[ir],had[ir],em[ir],pileUp[ir],mean[ir],rms[ir],bin_,hf_,sumET_,iev_,dr[ir],rawJetPt[ir],evtJetPt};
    Float_t entry[100];
    entry[0]=etaRandom[ir];
    entry[1]=phiRandom[ir];
    entry[2]=phiRel;
    entry[3]=et[ir];
    entry[4]=had[ir];
    entry[5]=em[ir];
    entry[6]=pileUp[ir];
    entry[7]=mean[ir];
    entry[8]=rms[ir];
    entry[9]=bin_;
    entry[10]=hf_;
    entry[11]=sumET_;
    entry[12]=iev_;
    entry[13]=dr[ir];
    entry[14]=rawJetPt[ir];
    entry[15]=evtJetPt;
    //Float_t entry[100] = {etaRandom[ir],phiRandom[ir],phiRel,et[ir],had[ir],em[ir],pileUp[ir],mean[ir],rms[ir],bin_,hf_,sumET_,iev_,dr[ir],rawJetPt[ir],evtJetPt};
    if(!backToBack_)ntRandom->Fill(entry);
    if(et[ir] < 0){
      //	 cout<<"Flipping vector"<<endl;
      (*directions)[ir] = false;
      et[ir] = -et[ir];
    }else{
      //         cout<<"Keep vector same"<<endl;
      (*directions)[ir] = true;
    }
    // cout<<"Lorentz"<<endl;

    math::PtEtaPhiMLorentzVector p(et[ir],etaRandom[ir],phiRandom[ir],0);
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
  //   iEvent.put(jets,"randomCones");
  //   iEvent.put(directions,"directions");
}

//void JetAlgorithmAnalyzer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup, std::vector<fastjet::PseudoJet>& input )
void JetAlgorithmAnalyzer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup)
{
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

DEFINE_FWK_MODULE(JetAlgorithmAnalyzer);

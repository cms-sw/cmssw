////////////////////////////////////////////////////////////////////////////////
//
// VirtualJetProducer
// ------------------
//
//            04/21/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "RecoJets/JetProducers/plugins/VirtualJetProducer.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"
#include "RecoJets/JetProducers/interface/BackgroundEstimator.h"
#include "RecoJets/JetProducers/interface/VirtualJetProducerHelper.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/PFClusterJetCollection.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

//#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

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


namespace reco {
  namespace helper {
    struct GreaterByPtPseudoJet {
      bool operator()( const fastjet::PseudoJet & t1, const fastjet::PseudoJet & t2 ) const {
        return t1.perp() > t2.perp();
      }
    };

  }
}

//______________________________________________________________________________
const char *VirtualJetProducer::JetType::names[] = {
  "BasicJet","GenJet","CaloJet","PFJet","TrackJet","PFClusterJet"
};


//______________________________________________________________________________
VirtualJetProducer::JetType::Type
VirtualJetProducer::JetType::byName(const string &name)
{
  const char **pos = std::find(names, names + LastJetType, name);
  if (pos == names + LastJetType) {
    std::string errorMessage="Requested jetType not supported: "+name+"\n"; 
    throw cms::Exception("Configuration",errorMessage);
  }
  return (Type)(pos-names);
}


void VirtualJetProducer::makeProduces( std::string alias, std::string tag ) 
{


  if ( writeCompound_ ) {
    produces<reco::BasicJetCollection>();
  }

  if (makeCaloJet(jetTypeE)) {
    produces<reco::CaloJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makePFJet(jetTypeE)) {
    produces<reco::PFJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makeGenJet(jetTypeE)) {
    produces<reco::GenJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makeTrackJet(jetTypeE)) {
    produces<reco::TrackJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makePFClusterJet(jetTypeE)) {
    produces<reco::PFClusterJetCollection>(tag).setBranchAlias(alias);
  }
  else if (makeBasicJet(jetTypeE)) {
    produces<reco::BasicJetCollection>(tag).setBranchAlias(alias);
  }
}

////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
VirtualJetProducer::VirtualJetProducer(const edm::ParameterSet& iConfig)
  : moduleLabel_   (iConfig.getParameter<string>       ("@module_label"))
  , src_           (iConfig.getParameter<edm::InputTag>("src"))
  , srcPVs_        (iConfig.getParameter<edm::InputTag>("srcPVs"))
  , jetType_       (iConfig.getParameter<string>       ("jetType"))
  , jetAlgorithm_  (iConfig.getParameter<string>       ("jetAlgorithm"))
  , rParam_        (iConfig.getParameter<double>       ("rParam"))
  , inputEtMin_    (iConfig.getParameter<double>       ("inputEtMin"))
  , inputEMin_     (iConfig.getParameter<double>       ("inputEMin"))
  , jetPtMin_      (iConfig.getParameter<double>       ("jetPtMin"))
  , doPVCorrection_(iConfig.getParameter<bool>         ("doPVCorrection"))
  , restrictInputs_(false)
  , maxInputs_(99999999)
  , doAreaFastjet_ (iConfig.getParameter<bool>         ("doAreaFastjet"))
  , useExplicitGhosts_(false)
  , doAreaDiskApprox_       (false)
  , doRhoFastjet_  (iConfig.getParameter<bool>         ("doRhoFastjet"))
  , voronoiRfact_           (-9)
  , doPUOffsetCorr_(iConfig.getParameter<bool>         ("doPUOffsetCorr"))
  , puWidth_(0)
  , nExclude_(0)
  , jetCollInstanceName_ ("")
  , writeCompound_ ( false )
{
  anomalousTowerDef_ = std::auto_ptr<AnomalousTower>(new AnomalousTower(iConfig));

  //
  // additional parameters to think about:
  // - overlap threshold (set to 0.75 for the time being)
  // - p parameter for generalized kT (set to -2 for the time being)
  // - fastjet PU subtraction parameters (not yet considered)
  //
  if (jetAlgorithm_=="SISCone") {
    fjPlugin_ = PluginPtr( new fastjet::SISConePlugin(rParam_,0.75,0,0.0,false,
                                                      fastjet::SISConePlugin::SM_pttilde) );
    fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(&*fjPlugin_) );
  }
  else if (jetAlgorithm_=="IterativeCone") {
    fjPlugin_ = PluginPtr(new fastjet::CMSIterativeConePlugin(rParam_,1.0));
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(&*fjPlugin_));
  }
  else if (jetAlgorithm_=="CDFMidPoint") {
    fjPlugin_ = PluginPtr(new fastjet::CDFMidPointPlugin(rParam_,0.75));
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(&*fjPlugin_));
  }
  else if (jetAlgorithm_=="ATLASCone") {
    fjPlugin_ = PluginPtr(new fastjet::ATLASConePlugin(rParam_));
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(&*fjPlugin_));
  }
  else if (jetAlgorithm_=="Kt")
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(fastjet::kt_algorithm,rParam_));
  else if (jetAlgorithm_=="CambridgeAachen")
    fjJetDefinition_= JetDefPtr(new fastjet::JetDefinition(fastjet::cambridge_algorithm,
                                                           rParam_) );
  else if (jetAlgorithm_=="AntiKt")
    fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::antikt_algorithm,rParam_) );
  else if (jetAlgorithm_=="GeneralizedKt")
    fjJetDefinition_= JetDefPtr( new fastjet::JetDefinition(fastjet::genkt_algorithm,
                                                            rParam_,-2) );
  else
    throw cms::Exception("Invalid jetAlgorithm")
      <<"Jet algorithm for VirtualJetProducer is invalid, Abort!\n";
  
  jetTypeE=JetType::byName(jetType_);

  if ( iConfig.exists("jetCollInstanceName") ) {
    jetCollInstanceName_ = iConfig.getParameter<string>("jetCollInstanceName");
  }

  if ( doPUOffsetCorr_ ) {
    if ( jetTypeE != JetType::CaloJet && jetTypeE != JetType::BasicJet) {
        throw cms::Exception("InvalidInput") << "Can only offset correct jets of type CaloJet or BasicJet";
     }
     
     if(iConfig.exists("subtractorName")) puSubtractorName_  =  iConfig.getParameter<string> ("subtractorName");
     else puSubtractorName_ = std::string();
     
     if(puSubtractorName_.empty()){
       edm::LogWarning("VirtualJetProducer") << "Pile Up correction on; however, pile up type is not specified. Using default... \n";
       subtractor_ =  boost::shared_ptr<PileUpSubtractor>(new PileUpSubtractor(iConfig));
     }else{
       subtractor_ =  boost::shared_ptr<PileUpSubtractor>(PileUpSubtractorFactory::get()->create( puSubtractorName_, iConfig));
     }
  }

  // use explicit ghosts in the fastjet clustering sequence?
  if ( iConfig.exists("useExplicitGhosts") ) {
    useExplicitGhosts_ = iConfig.getParameter<bool>("useExplicitGhosts");
  }

  // do approximate disk-based area calculation => warn if conflicting request
  if (iConfig.exists("doAreaDiskApprox")) {
    doAreaDiskApprox_ = iConfig.getParameter<bool>("doAreaDiskApprox");
    if (doAreaDiskApprox_ && doAreaFastjet_)
      throw cms::Exception("Conflicting area calculations") << "Both the calculation of jet area via fastjet and via an analytical disk approximation have been requested. Please decide on one.\n";

  }
  // turn off jet collection output for speed
  // Voronoi-based area calculation allows for an empirical scale factor
  if (iConfig.exists("voronoiRfact"))
    voronoiRfact_     = iConfig.getParameter<double>("voronoiRfact");


  // do fasjet area / rho calcluation? => accept corresponding parameters
  if ( doAreaFastjet_ || doRhoFastjet_ ) {
    // Eta range of jets to be considered for Rho calculation
    // Should be at most (jet acceptance - jet radius)
    double rhoEtaMax=iConfig.getParameter<double>("Rho_EtaMax");
    // default Ghost_EtaMax should be 5
    double ghostEtaMax = iConfig.getParameter<double>("Ghost_EtaMax");
    // default Active_Area_Repeats 1
    int    activeAreaRepeats = iConfig.getParameter<int> ("Active_Area_Repeats");
    // default GhostArea 0.01
    double ghostArea = iConfig.getParameter<double> ("GhostArea");
    if (voronoiRfact_ <= 0) {
      fjActiveArea_     = ActiveAreaSpecPtr(new fastjet::GhostedAreaSpec(ghostEtaMax,activeAreaRepeats,ghostArea));
      fjActiveArea_->set_fj2_placement(true);
      if ( ! useExplicitGhosts_ ) {
	fjAreaDefinition_ = AreaDefinitionPtr( new fastjet::AreaDefinition(fastjet::active_area, *fjActiveArea_ ) );
      } else {
	fjAreaDefinition_ = AreaDefinitionPtr( new fastjet::AreaDefinition(fastjet::active_area_explicit_ghosts, *fjActiveArea_ ) );
      }
    }
    fjRangeDef_ = RangeDefPtr( new fastjet::RangeDefinition(rhoEtaMax) );
  } 

  // restrict inputs to first "maxInputs" towers?
  if ( iConfig.exists("restrictInputs") ) {
    restrictInputs_ = iConfig.getParameter<bool>("restrictInputs");
    maxInputs_      = iConfig.getParameter<unsigned int>("maxInputs");
  }
 

  string alias=iConfig.getUntrackedParameter<string>("alias",moduleLabel_);


  // Check to see if we are writing compound jets for substructure
  // and jet grooming
  if ( iConfig.exists("writeCompound") ) {
    writeCompound_ = iConfig.getParameter<bool>("writeCompound");
  }

  // make the "produces" statements
  makeProduces( alias, jetCollInstanceName_ );

  doFastJetNonUniform_ = false;
  if(iConfig.exists("doFastJetNonUniform")) doFastJetNonUniform_ = iConfig.getParameter<bool>   ("doFastJetNonUniform");
  if(doFastJetNonUniform_){
    puCenters_ = iConfig.getParameter<std::vector<double> >("puCenters");
    puWidth_ = iConfig.getParameter<double>("puWidth");
    nExclude_ = iConfig.getParameter<unsigned int>("nExclude");
  }

  useDeterministicSeed_ = false;
  minSeed_ = 0;
  if ( iConfig.exists("useDeterministicSeed") ) {
    useDeterministicSeed_ = iConfig.getParameter<bool>("useDeterministicSeed");
    minSeed_ =              iConfig.getParameter<unsigned int>("minSeed");
  }


  produces<std::vector<double> >("rhos");
  produces<std::vector<double> >("sigmas");
  produces<double>("rho");
  produces<double>("sigma");

  
}

//______________________________________________________________________________
VirtualJetProducer::~VirtualJetProducer()
{
} 


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
void VirtualJetProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup)
{

  // If requested, set the fastjet random seed to a deterministic function
  // of the run/lumi/event. 
  // NOTE!!! The fastjet random number sequence is a global singleton.
  // Thus, we have to create an object and get access to the global singleton
  // in order to change it. 
  if ( useDeterministicSeed_ ) {
    fastjet::GhostedAreaSpec gas;
    std::vector<int> seeds(2);
    seeds[0] = std::max(iEvent.id().run(),minSeed_ + 3) + 3 * iEvent.id().event();
    seeds[1] = std::max(iEvent.id().run(),minSeed_ + 5) + 5 * iEvent.id().event();
    gas.set_random_status(seeds);
  }

  LogDebug("VirtualJetProducer") << "Entered produce\n";
  //determine signal vertex2
  vertex_=reco::Jet::Point(0,0,0);
  if (makeCaloJet(jetTypeE)&&doPVCorrection_) {
    LogDebug("VirtualJetProducer") << "Adding PV info\n";
    edm::Handle<reco::VertexCollection> pvCollection;
    iEvent.getByLabel(srcPVs_,pvCollection);
    if (pvCollection->size()>0) vertex_=pvCollection->begin()->position();
  }

  // For Pileup subtraction using offset correction:
  // set up geometry map
  if ( doPUOffsetCorr_ ) {
     subtractor_->setupGeometryMap(iEvent, iSetup);
  }

  // clear data
  LogDebug("VirtualJetProducer") << "Clear data\n";
  fjInputs_.clear();
  fjJets_.clear();
  inputs_.clear();  
  
  // get inputs and convert them to the fastjet format (fastjet::PeudoJet)
  edm::Handle<reco::CandidateView> inputsHandle;
  edm::Handle< std::vector<edm::FwdPtr<reco::PFCandidate> > > pfinputsHandleAsFwdPtr; 
  
  bool isView = iEvent.getByLabel(src_,inputsHandle);
  if ( isView ) {
    for (size_t i = 0; i < inputsHandle->size(); ++i) {
      inputs_.push_back(inputsHandle->ptrAt(i));
    }
  } else {
    iEvent.getByLabel(src_,pfinputsHandleAsFwdPtr);
    for (size_t i = 0; i < pfinputsHandleAsFwdPtr->size(); ++i) {
      if ( (*pfinputsHandleAsFwdPtr)[i].ptr().isAvailable() ) {
	inputs_.push_back( (*pfinputsHandleAsFwdPtr)[i].ptr() );
      }
      else if ( (*pfinputsHandleAsFwdPtr)[i].backPtr().isAvailable() ) {
	inputs_.push_back( (*pfinputsHandleAsFwdPtr)[i].backPtr() );
      }
    }
  }
  LogDebug("VirtualJetProducer") << "Got inputs\n";
  
  // Convert candidates to fastjet::PseudoJets.
  // Also correct to Primary Vertex. Will modify fjInputs_
  // and use inputs_
  fjInputs_.reserve(inputs_.size());
  inputTowers();
  LogDebug("VirtualJetProducer") << "Inputted towers\n";

  // For Pileup subtraction using offset correction:
  // Subtract pedestal. 
  if ( doPUOffsetCorr_ ) {
     subtractor_->setDefinition(fjJetDefinition_);
     subtractor_->reset(inputs_,fjInputs_,fjJets_);
     subtractor_->calculatePedestal(fjInputs_); 
     subtractor_->subtractPedestal(fjInputs_);    
     LogDebug("VirtualJetProducer") << "Subtracted pedestal\n";
  }
  // Run algorithm. Will modify fjJets_ and allocate fjClusterSeq_. 
  // This will use fjInputs_
  runAlgorithm( iEvent, iSetup );

  // if ( doPUOffsetCorr_ ) {
  //    subtractor_->setAlgorithm(fjClusterSeq_);
  // }

  LogDebug("VirtualJetProducer") << "Ran algorithm\n";
  // For Pileup subtraction using offset correction:
  // Now we find jets and need to recalculate their energy,
  // mark towers participated in jet,
  // remove occupied towers from the list and recalculate mean and sigma
  // put the initial towers collection to the jet,   
  // and subtract from initial towers in jet recalculated mean and sigma of towers 
  if ( doPUOffsetCorr_ ) {
    LogDebug("VirtualJetProducer") << "Do PUOffsetCorr\n";
    vector<fastjet::PseudoJet> orphanInput;
    subtractor_->calculateOrphanInput(orphanInput);
    subtractor_->calculatePedestal(orphanInput);
    subtractor_->offsetCorrectJets();
  }
  // Write the output jets.
  // This will (by default) call the member function template
  // "writeJets", but can be overridden. 
  // this will use inputs_
  output( iEvent, iSetup );
  LogDebug("VirtualJetProducer") << "Wrote jets\n";
  
  return;
}

//______________________________________________________________________________
  
void VirtualJetProducer::inputTowers( )
{
  std::vector<edm::Ptr<reco::Candidate> >::const_iterator inBegin = inputs_.begin(),
    inEnd = inputs_.end(), i = inBegin;
  for (; i != inEnd; ++i ) {
    reco::CandidatePtr input = *i;
    if (edm::isNotFinite(input->pt()))           continue;
    if (input->et()    <inputEtMin_)  continue;
    if (input->energy()<inputEMin_)   continue;
    if (isAnomalousTower(input))      continue;
    if (input->pt() == 0) {
      edm::LogError("NullTransverseMomentum") << "dropping input candidate with pt=0";
      continue;
    }
    if (makeCaloJet(jetTypeE)&&doPVCorrection_) {
      const CaloTower* tower=dynamic_cast<const CaloTower*>(input.get());
      math::PtEtaPhiMLorentzVector ct(tower->p4(vertex_));
      fjInputs_.push_back(fastjet::PseudoJet(ct.px(),ct.py(),ct.pz(),ct.energy()));
      //std::cout << "tower:" << *tower << '\n';
    }
    else {
      /*
      if(makePFJet(jetTypeE)) {
	reco::PFCandidate* pfc = (reco::PFCandidate*)input.get();
	std::cout << "PF cand:" << *pfc << '\n';
      }
      */
      fjInputs_.push_back(fastjet::PseudoJet(input->px(),input->py(),input->pz(),
					     input->energy()));
    }
    fjInputs_.back().set_user_index(i - inBegin);
  }

  if ( restrictInputs_ && fjInputs_.size() > maxInputs_ ) {
    reco::helper::GreaterByPtPseudoJet   pTComparator;
    std::sort(fjInputs_.begin(), fjInputs_.end(), pTComparator);
    fjInputs_.resize(maxInputs_);
    edm::LogWarning("JetRecoTooManyEntries") << "Too many inputs in the event, limiting to first " << maxInputs_ << ". Output is suspect.";
  }
}

//______________________________________________________________________________
bool VirtualJetProducer::isAnomalousTower(reco::CandidatePtr input)
{
  if (!makeCaloJet(jetTypeE)) 
      return false;
  else
      return (*anomalousTowerDef_)(*input);
}

//------------------------------------------------------------------------------
// This is pure virtual. 
//______________________________________________________________________________
// void VirtualJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup,
//                                        std::vector<edm::Ptr<reco::Candidate> > const & inputs_);

//______________________________________________________________________________
void VirtualJetProducer::copyConstituents(const vector<fastjet::PseudoJet>& fjConstituents,
                                          reco::Jet* jet)
{
  for (unsigned int i=0;i<fjConstituents.size();++i) { 
    int index = fjConstituents[i].user_index();
    if ( index >= 0 && static_cast<unsigned int>(index) < inputs_.size() )
      jet->addDaughter(inputs_[index]);
  }
}


//______________________________________________________________________________
vector<reco::CandidatePtr>
VirtualJetProducer::getConstituents(const vector<fastjet::PseudoJet>&fjConstituents)
{
  vector<reco::CandidatePtr> result;
  for (unsigned int i=0;i<fjConstituents.size();i++) {
    int index = fjConstituents[i].user_index();
    if ( index >= 0 && static_cast<unsigned int>(index) < inputs_.size() ) {
      reco::CandidatePtr candidate = inputs_[index];
      result.push_back(candidate);
    }
  }
  return result;
}


//_____________________________________________________________________________

void VirtualJetProducer::output(edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // Write jets and constitutents. Will use fjJets_, inputs_
  // and fjClusterSeq_

  if ( !writeCompound_ ) {
    switch( jetTypeE ) {
    case JetType::CaloJet :
      writeJets<reco::CaloJet>( iEvent, iSetup);
      break;
    case JetType::PFJet :
      writeJets<reco::PFJet>( iEvent, iSetup);
      break;
    case JetType::GenJet :
      writeJets<reco::GenJet>( iEvent, iSetup);
      break;
    case JetType::TrackJet :
      writeJets<reco::TrackJet>( iEvent, iSetup);
      break;
    case JetType::PFClusterJet :
      writeJets<reco::PFClusterJet>( iEvent, iSetup);
      break;
    case JetType::BasicJet :
      writeJets<reco::BasicJet>( iEvent, iSetup);
      break;
    default:
      throw cms::Exception("InvalidInput") << "invalid jet type in VirtualJetProducer\n";
      break;
    };
  } else {
    // Write jets and constitutents.
    switch( jetTypeE ) {
    case JetType::CaloJet :
      writeCompoundJets<reco::CaloJet>( iEvent, iSetup );
      break;
    case JetType::PFJet :
      writeCompoundJets<reco::PFJet>( iEvent, iSetup );
      break;
    case JetType::GenJet :
      writeCompoundJets<reco::GenJet>( iEvent, iSetup );
      break;
    case JetType::BasicJet :
      writeCompoundJets<reco::BasicJet>( iEvent, iSetup );
      break;
    default:
      throw cms::Exception("InvalidInput") << "invalid jet type in CompoundJetProducer\n";
      break;
    };
  }
  
}

template< typename T >
void VirtualJetProducer::writeJets( edm::Event & iEvent, edm::EventSetup const& iSetup )
{
  if (doRhoFastjet_) {
    // declare jet collection without the two jets, 
    // for unbiased background estimation.
    std::vector<fastjet::PseudoJet> fjexcluded_jets;
    fjexcluded_jets=fjJets_;
    
    if(fjexcluded_jets.size()>2) fjexcluded_jets.resize(nExclude_);
    
    if(doFastJetNonUniform_){
      std::auto_ptr<std::vector<double> > rhos(new std::vector<double>);
      std::auto_ptr<std::vector<double> > sigmas(new std::vector<double>);
      int nEta = puCenters_.size();
      rhos->reserve(nEta);
      sigmas->reserve(nEta);
      fastjet::ClusterSequenceAreaBase const* clusterSequenceWithArea =
        dynamic_cast<fastjet::ClusterSequenceAreaBase const *> ( &*fjClusterSeq_ );

      
      for(int ie = 0; ie < nEta; ++ie){
        double eta = puCenters_[ie];
        double etamin=eta-puWidth_;
        double etamax=eta+puWidth_;
        fastjet::RangeDefinition range_rho(etamin,etamax);
        fastjet::BackgroundEstimator bkgestim(*clusterSequenceWithArea,range_rho);
        bkgestim.set_excluded_jets(fjexcluded_jets);
        rhos->push_back(bkgestim.rho());
        sigmas->push_back(bkgestim.sigma());
      }
      iEvent.put(rhos,"rhos");
      iEvent.put(sigmas,"sigmas");
    }else{
      std::auto_ptr<double> rho(new double(0.0));
      std::auto_ptr<double> sigma(new double(0.0));
      double mean_area = 0;
      
      fastjet::ClusterSequenceAreaBase const* clusterSequenceWithArea =
        dynamic_cast<fastjet::ClusterSequenceAreaBase const *> ( &*fjClusterSeq_ );
      /*
	const double nemptyjets = clusterSequenceWithArea->n_empty_jets(*fjRangeDef_);
	if(( nemptyjets  < -15 ) || ( nemptyjets > fjRangeDef_->area()+ 15)) {
	edm::LogWarning("StrangeNEmtpyJets") << "n_empty_jets is : " << clusterSequenceWithArea->n_empty_jets(*fjRangeDef_) << " with range " << fjRangeDef_->description() << ".";
	}
      */
      clusterSequenceWithArea->get_median_rho_and_sigma(*fjRangeDef_,false,*rho,*sigma,mean_area);
      if((*rho < 0)|| (edm::isNotFinite(*rho))) {
	edm::LogError("BadRho") << "rho value is " << *rho << " area:" << mean_area << " and n_empty_jets: " << clusterSequenceWithArea->n_empty_jets(*fjRangeDef_) << " with range " << fjRangeDef_->description()
				<<". Setting rho to rezo.";
	*rho = 0;
      }
      iEvent.put(rho,"rho");
      iEvent.put(sigma,"sigma");
    }
  } // doRhoFastjet_
  
  // produce output jet collection
  
  using namespace reco;
  
  std::auto_ptr<std::vector<T> > jets(new std::vector<T>() );
  jets->reserve(fjJets_.size());
  
  // Distance between jet centers -- for disk-based area calculation
  std::vector<std::vector<double> >   rij(fjJets_.size());

  for (unsigned int ijet=0;ijet<fjJets_.size();++ijet) {
    // allocate this jet
    T jet;
    // get the fastjet jet
    const fastjet::PseudoJet& fjJet = fjJets_[ijet];
    // get the constituents from fastjet
    std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(fjJet.constituents());
    // convert them to CandidatePtr vector
    std::vector<CandidatePtr> constituents =
      getConstituents(fjConstituents);


    // calcuate the jet area
    double jetArea=0.0;
    if ( doAreaFastjet_ && fjJet.has_area() ) {
      jetArea = fjJet.area();
    }
    else if ( doAreaDiskApprox_ ) {
      // Here it is assumed that fjJets_ is in decreasing order of pT, 
      // which should happen in FastjetJetProducer::runAlgorithm() 
      jetArea   = M_PI;
      if (ijet) {
        std::vector<double>&  distance  = rij[ijet];
        distance.resize(ijet);
        for (unsigned jJet = 0; jJet < ijet; ++jJet) {
          distance[jJet]      = reco::deltaR(fjJets_[ijet],fjJets_[jJet]) / rParam_;
          jetArea            -= reco::helper::VirtualJetProducerHelper::intersection(distance[jJet]);
          for (unsigned kJet = 0; kJet < jJet; ++kJet) {
            jetArea          += reco::helper::VirtualJetProducerHelper::intersection(distance[jJet], distance[kJet], rij[jJet][kJet]);
          } // end loop over harder jets
        } // end loop over harder jets
      }
      jetArea  *= rParam_;
      jetArea  *= rParam_;
    }  
    // write the specifics to the jet (simultaneously sets 4-vector, vertex).
    // These are overridden functions that will call the appropriate
    // specific allocator. 
    writeSpecific(jet,
                  Particle::LorentzVector(fjJet.px(),
                                          fjJet.py(),
                                          fjJet.pz(),
                                          fjJet.E()),
                  vertex_, 
                  constituents, iSetup);

    jet.setJetArea (jetArea);
    
    if(doPUOffsetCorr_){
      jet.setPileup(subtractor_->getPileUpEnergy(ijet));
    }else{
      jet.setPileup (0.0);
    }
    
    // add to the list
    jets->push_back(jet);        
  }
  // put the jets in the collection
  iEvent.put(jets,jetCollInstanceName_);
  

}



/// function template to write out the outputs
template< class T>
void VirtualJetProducer::writeCompoundJets(  edm::Event & iEvent, edm::EventSetup const& iSetup)
{

  // get a list of output jets
  std::auto_ptr<reco::BasicJetCollection>  jetCollection( new reco::BasicJetCollection() );
  // get a list of output subjets
  std::auto_ptr<std::vector<T> >  subjetCollection( new std::vector<T>() );

  // This will store the handle for the subjets after we write them
  edm::OrphanHandle< std::vector<T> > subjetHandleAfterPut;
  // this is the mapping of subjet to hard jet
  std::vector< std::vector<int> > indices;
  // this is the list of hardjet 4-momenta
  std::vector<math::XYZTLorentzVector> p4_hardJets;
  // this is the hardjet areas
  std::vector<double> area_hardJets;


  // Loop over the hard jets
  std::vector<fastjet::PseudoJet>::const_iterator it = fjJets_.begin(),
    iEnd = fjJets_.end(),
    iBegin = fjJets_.begin();
  indices.resize( fjJets_.size() );
  for ( ; it != iEnd; ++it ) {
    fastjet::PseudoJet const & localJet = *it;
    unsigned int jetIndex = it - iBegin;
    // Get the 4-vector for the hard jet
    p4_hardJets.push_back( math::XYZTLorentzVector(localJet.px(), localJet.py(), localJet.pz(), localJet.e() ));
    double localJetArea = 0.0;
    if ( doAreaFastjet_ && localJet.has_area() ) {
      localJetArea = localJet.area();
    }
    area_hardJets.push_back( localJetArea );

    // create the subjet list
    std::vector<fastjet::PseudoJet> constituents;
    if ( it->has_pieces() ) {
      constituents = it->pieces();
    } else {
      constituents=it->constituents();
    }

    
    std::vector<fastjet::PseudoJet>::const_iterator itSubJetBegin = constituents.begin(),
      itSubJet = itSubJetBegin, itSubJetEnd = constituents.end();
    for (; itSubJet != itSubJetEnd; ++itSubJet ){

      fastjet::PseudoJet const & subjet = *itSubJet;      

      math::XYZTLorentzVector p4Subjet(subjet.px(), subjet.py(), subjet.pz(), subjet.e() );
      reco::Particle::Point point(0,0,0);

      // This will hold ptr's to the subjets
      std::vector<reco::CandidatePtr> subjetConstituents;

      // Get the transient subjet constituents from fastjet
      std::vector<fastjet::PseudoJet> subjetFastjetConstituents = subjet.constituents();
      std::vector<reco::CandidatePtr> constituents =
	getConstituents(subjetFastjetConstituents );    

      indices[jetIndex].push_back( subjetCollection->size() );

      // Add the concrete subjet type to the subjet list to write to event record
      T jet;
      reco::writeSpecific( jet, p4Subjet, point, constituents, iSetup);
      double subjetArea = 0.0;
      if ( doAreaFastjet_ && itSubJet->has_area() ){
	subjetArea = itSubJet->area();
      }
      jet.setJetArea( subjetArea );
      subjetCollection->push_back( jet );

    }
  }
  // put subjets into event record
  subjetHandleAfterPut = iEvent.put( subjetCollection, jetCollInstanceName_ );
  
  
  // Now create the hard jets with ptr's to the subjets as constituents
  std::vector<math::XYZTLorentzVector>::const_iterator ip4 = p4_hardJets.begin(),
    ip4Begin = p4_hardJets.begin(),
    ip4End = p4_hardJets.end();

  for ( ; ip4 != ip4End; ++ip4 ) {
    int p4_index = ip4 - ip4Begin;
    std::vector<int> & ind = indices[p4_index];
    std::vector<reco::CandidatePtr> i_hardJetConstituents;
    // Add the subjets to the hard jet
    for( std::vector<int>::const_iterator isub = ind.begin();
	 isub != ind.end(); ++isub ) {
      reco::CandidatePtr candPtr( subjetHandleAfterPut, *isub, false );
      i_hardJetConstituents.push_back( candPtr );
    }   
    reco::Particle::Point point(0,0,0);
    reco::BasicJet toput( *ip4, point, i_hardJetConstituents);
    toput.setJetArea( area_hardJets[ip4 - ip4Begin] );
    jetCollection->push_back( toput );
  }
  
  // put hard jets into event record
  iEvent.put( jetCollection);

}

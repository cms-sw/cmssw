#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetProducers/plugins/CSJetProducer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"

#include "fastjet/contrib/ConstituentSubtractor.hh"

using namespace std;
using namespace reco;
using namespace edm;
using namespace cms;

CSJetProducer::CSJetProducer(edm::ParameterSet const& conf):
  VirtualJetProducer( conf ),
  csRParam_(-1.0),
  csAlpha_(0.)
{
  //get eta range, rho and rhom map
  etaToken_ = consumes<std::vector<double>>(conf.getParameter<edm::InputTag>( "etaMap" ));
  rhoToken_ = consumes<std::vector<double>>(conf.getParameter<edm::InputTag>( "rho" ));
  rhomToken_ = consumes<std::vector<double>>(conf.getParameter<edm::InputTag>( "rhom" ));
  csRParam_ = conf.getParameter<double>("csRParam");
  csAlpha_ = conf.getParameter<double>("csAlpha");
}

void CSJetProducer::produce( edm::Event & iEvent, const edm::EventSetup & iSetup )
{
  // use the default production from one collection
  VirtualJetProducer::produce( iEvent, iSetup );
  //use runAlgorithm of this class

  //Delete allocated memory. It is allocated every time runAlgorithm is called
  fjClusterSeq_.reset();
}

//______________________________________________________________________________
void CSJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup)
{
  // run algorithm
  if ( !doAreaFastjet_ && !doRhoFastjet_) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, *fjJetDefinition_ ) );
  } else if (voronoiRfact_ <= 0) {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceArea( fjInputs_, *fjJetDefinition_ , *fjAreaDefinition_ ) );
  } else {
    fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequenceVoronoiArea( fjInputs_, *fjJetDefinition_ , fastjet::VoronoiAreaSpec(voronoiRfact_) ) );
  }

  fjJets_.clear();
  std::vector<fastjet::PseudoJet> tempJets = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));

  //Get local rho and rhom map
  edm::Handle<std::vector<double>> etaRanges;
  edm::Handle<std::vector<double>> rhoRanges;
  edm::Handle<std::vector<double>> rhomRanges;
  
  iEvent.getByToken(etaToken_, etaRanges);
  iEvent.getByToken(rhoToken_, rhoRanges);
  iEvent.getByToken(rhomToken_, rhomRanges);

  //Check if size of eta and background density vectors is the same
  unsigned int bkgVecSize = etaRanges->size();
  if(bkgVecSize<1) { throw cms::Exception("WrongBkgInput") << "Producer needs vector with background estimates\n"; }
  if(bkgVecSize != (rhoRanges->size()+1) || bkgVecSize != (rhomRanges->size()+1) ) {
    throw cms::Exception("WrongBkgInput") << "Size of etaRanges (" << bkgVecSize << ") and rhoRanges (" << rhoRanges->size() << ") and/or rhomRanges (" << rhomRanges->size() << ") vectors inconsistent\n";
  }

  
  //Allow the background densities to change within the jet

  for(fastjet::PseudoJet& ijet : tempJets ) {
  
    //----------------------------------------------------------------------
    // sift ghosts and particles in the input jet
    std::vector<fastjet::PseudoJet> particles, ghosts;
    fastjet::SelectorIsPureGhost().sift(ijet.constituents(), ghosts, particles);
    unsigned long nParticles=particles.size();
    if(nParticles==0) continue; //don't subtract ghost jets
    
    //assign rho and rhom to ghosts according to local eta-dependent map
    double rho  = 1e-6;
    double rhom = 1e-6;
    for(fastjet::PseudoJet& ighost : ghosts) {
      
      if(ighost.eta()<=etaRanges->at(0) || bkgVecSize==1) {
        rho = rhoRanges->at(0);
        rhom = rhomRanges->at(0);
      } else if(ighost.eta()>=etaRanges->at(bkgVecSize-1)) {
        rho = rhoRanges->at(bkgVecSize-2);
        rhom = rhomRanges->at(bkgVecSize-2);
      } else {
        for(unsigned int ie = 0; ie<(bkgVecSize-1); ie++) {
          if(ighost.eta()>=etaRanges->at(ie) && ighost.eta()<etaRanges->at(ie+1)) {
            rho = rhoRanges->at(ie);
            rhom = rhomRanges->at(ie);
            break;
          }
        }
      }
      double pt = rho*ighost.area();
      double mass_squared=std::pow(rhom*ighost.area()+pt,2)-std::pow(pt,2);
      double mass=0;
      if (mass_squared>0) mass=sqrt(mass_squared);
      ighost.reset_momentum_PtYPhiM(pt,ighost.rap(),ighost.phi(),mass);
    }

    //----------------------------------------------------------------------
    //from here use official fastjet contrib package
    
    fastjet::contrib::ConstituentSubtractor subtractor;
    subtractor.set_distance_type(fastjet::contrib::ConstituentSubtractor::deltaR); // distance in eta-phi plane
    subtractor.set_max_distance(csRParam_); // free parameter for the maximal allowed distance between particle i and ghost k
    subtractor.set_alpha(csAlpha_);  // free parameter for the distance measure (the exponent of particle pt). Note that in older versions of the package alpha was multiplied by two but in newer versions this is not the case anymore
    subtractor.set_do_mass_subtraction(true);

    std::vector<fastjet::PseudoJet> subtracted_particles = subtractor.do_subtraction(particles,ghosts);

    //Create subtracted jets
    fastjet::PseudoJet subtracted_jet=join(subtracted_particles);
    if(subtracted_jet.perp()>0.)
      fjJets_.push_back( subtracted_jet );
  }
  fjJets_ = fastjet::sorted_by_pt(fjJets_); 
}

void CSJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription descCSJetProducer;
  ////// From CSJetProducer
  fillDescriptionsFromCSJetProducer(descCSJetProducer);
  ///// From VirtualJetProducer
  descCSJetProducer.add<string>("jetCollInstanceName", ""    );
  VirtualJetProducer::fillDescriptionsFromVirtualJetProducer(descCSJetProducer);
  descCSJetProducer.add<bool> ("sumRecHits", false);
  
  /////////////////////
  descriptions.add("CSJetProducer",descCSJetProducer);
  
}

void CSJetProducer::fillDescriptionsFromCSJetProducer(edm::ParameterSetDescription& desc) {

  desc.add<double>("csRParam",-1.);
  desc.add<double>("csAlpha",2.);

  desc.add<edm::InputTag>("etaMap",edm::InputTag("hiFJRhoProducer","mapEtaEdges") );
  desc.add<edm::InputTag>("rho",edm::InputTag("hiFJRhoProducer","mapToRho") );
  desc.add<edm::InputTag>("rhom",edm::InputTag("hiFJRhoProducer","mapToRhoM") );

}

////////////////////////////////////////////////////////////////////////////////
// define as cmssw plugin
////////////////////////////////////////////////////////////////////////////////

DEFINE_FWK_MODULE(CSJetProducer);

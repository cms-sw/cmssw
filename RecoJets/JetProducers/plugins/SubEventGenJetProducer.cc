
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetProducers/plugins/SubEventGenJetProducer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

using namespace std;
using namespace reco;
using namespace edm;
using namespace cms;

namespace {
  const bool debug = false;

}

namespace {
   bool checkHydro(const reco::GenParticle * p){
      const Candidate* m1 = p->mother();
      while(m1){
	 int pdg = abs(m1->pdgId());
	 int st = m1->status();
	 LogDebug("SubEventMothers")<<"Pdg ID : "<<pdg<<endl;
	 if(st == 3 || pdg < 9 || pdg == 21){
	    LogDebug("SubEventMothers")<<"Sub-Collision  Found! Pdg ID : "<<pdg<<endl;
	    return false;
	 }
         const Candidate* m = m1->mother();
	 m1 = m;
	 if(!m1) LogDebug("SubEventMothers")<<"No Mother, particle is : "<<pdg<<" with status "<<st<<endl;
      }
      //      return true;
      return true; // Debugging - to be changed
   }
}

SubEventGenJetProducer::SubEventGenJetProducer(edm::ParameterSet const& conf):
  VirtualJetProducer( conf )
{
   //   mapSrc_ = conf.getParameter<edm::InputTag>( "srcMap");
   ignoreHydro_ = conf.getUntrackedParameter<bool>("ignoreHydro", true);
   produces<reco::BasicJetCollection>();
  // the subjet collections are set through the config file in the "jetCollInstanceName" field.
}


void SubEventGenJetProducer::inputTowers( ) 
{
   std::vector<edm::Ptr<reco::Candidate> >::const_iterator inBegin = inputs_.begin(),
      inEnd = inputs_.end(), i = inBegin;
   for (; i != inEnd; ++i ) {
      reco::CandidatePtr input = inputs_[i - inBegin];
      if (edm::isNotFinite(input->pt()))           continue;
      if (input->et()    <inputEtMin_)  continue;
      if (input->energy()<inputEMin_)   continue;
      if (isAnomalousTower(input))      continue;

      edm::Ptr<reco::Candidate> p = inputs_[i - inBegin];
      const GenParticle * pref = dynamic_cast<const GenParticle *>(p.get());
      int subevent = pref->collisionId();
      LogDebug("SubEventContainers")<<"SubEvent is : "<<subevent<<endl;
      LogDebug("SubEventContainers")<<"SubSize is : "<<subInputs_.size()<<endl;

      if(subevent >= (int)subInputs_.size()){ 
	 hydroTag_.resize(subevent+1, -1);
         subInputs_.resize(subevent+1);
      LogDebug("SubEventContainers")<<"SubSize is : "<<subInputs_.size()<<endl;
      LogDebug("SubEventContainers")<<"HydroTagSize is : "<<hydroTag_.size()<<endl;
      }

      LogDebug("SubEventContainers")<<"HydroTag is : "<<hydroTag_[subevent]<<endl;
      if(hydroTag_[subevent] != 0) hydroTag_[subevent] = (int)checkHydro(pref);

      subInputs_[subevent].push_back(fastjet::PseudoJet(input->px(),input->py(),input->pz(),
						input->energy()));

      subInputs_[subevent].back().set_user_index(i - inBegin);

   }

}

void SubEventGenJetProducer::produce(edm::Event& iEvent,const edm::EventSetup& iSetup){
   LogDebug("VirtualJetProducer") << "Entered produce\n";

   fjJets_.clear();
   subInputs_.clear();
   nSubParticles_.clear();
   hydroTag_.clear();
   inputs_.clear();

   // get inputs and convert them to the fastjet format (fastjet::PeudoJet)
   edm::Handle<reco::CandidateView> inputsHandle;
   iEvent.getByLabel(src_,inputsHandle);
   for (size_t i = 0; i < inputsHandle->size(); ++i) {
     inputs_.push_back(inputsHandle->ptrAt(i));
   }
   LogDebug("VirtualJetProducer") << "Got inputs\n";

   inputTowers();
   // Convert candidates to fastjet::PseudoJets.
   // Also correct to Primary Vertex. Will modify fjInputs_
   // and use inputs_

   ////////////////

   std::auto_ptr<std::vector<GenJet> > jets(new std::vector<GenJet>() );
   subJets_ = jets.get();

   LogDebug("VirtualJetProducer") << "Inputted towers\n";

   size_t nsub = subInputs_.size();

   for(size_t isub = 0; isub < nsub; ++isub){
      if(ignoreHydro_ && hydroTag_[isub]) continue;
      fjJets_.clear();
      fjInputs_.clear();
      fjInputs_ = subInputs_[isub];
      runAlgorithm( iEvent, iSetup );
   }

   //Finalize
   LogDebug("SubEventJetProducer") << "Wrote jets\n";

   iEvent.put(jets);  
   return;
}

void SubEventGenJetProducer::runAlgorithm( edm::Event & iEvent, edm::EventSetup const& iSetup)
{
   // run algorithm
   fjJets_.clear();

   fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, *fjJetDefinition_ ) );
   fjJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(jetPtMin_));
   
   using namespace reco;

   for (unsigned int ijet=0;ijet<fjJets_.size();++ijet) {
      
      GenJet jet;
      const fastjet::PseudoJet& fjJet = fjJets_[ijet];

      std::vector<fastjet::PseudoJet> fjConstituents =
	 sorted_by_pt(fjClusterSeq_->constituents(fjJet));

      std::vector<CandidatePtr> constituents =
	 getConstituents(fjConstituents);

    double px=fjJet.px();
    double py=fjJet.py();
    double pz=fjJet.pz();
    double E=fjJet.E();
    double jetArea=0.0;
    double pu=0.;

    writeSpecific( jet,
		   Particle::LorentzVector(px, py, pz, E),
		   vertex_,
		   constituents, iSetup);
    
    jet.setJetArea (jetArea);
    jet.setPileup (pu);
    
    subJets_->push_back(jet);
   }   
}

DEFINE_FWK_MODULE(SubEventGenJetProducer);



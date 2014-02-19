#include "RecoHI/HiJetAlgos/interface/VoronoiSubtractor.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
using namespace std;

bool VoronoiSubtractor::match(const fastjet::PseudoJet cand1, const fastjet::PseudoJet cand2){
   return (cand1.delta_R(cand2) < rParam_);
}


VoronoiSubtractor::VoronoiSubtractor(const edm::ParameterSet& iConfig) :
   PileUpSubtractor(iConfig),
   srcCand_(iConfig.getParameter<edm::InputTag>("src")),
   srcVor_(iConfig.getParameter<edm::InputTag>("bkg")),
   dropZeroTowers_(iConfig.getParameter<bool>("dropZeros")),
   addNegativesFromCone_(iConfig.getParameter<bool>("addNegativesFromCone")),
   infinitesimalPt_(iConfig.getParameter<double>("infinitesimalPt")),
   rParam_(iConfig.getParameter<double>("rParam"))
{

}



void VoronoiSubtractor::setupGeometryMap(edm::Event& iEvent,const edm::EventSetup& iSetup)
{

   LogDebug("VoronoiSubtractor")<<"The subtractor retrieving Voronoi background...\n";
   geo_ = 0;
   droppedCandidates_.clear();
   jetOffset_.clear();

   iEvent.getByLabel(srcCand_,candidates_);
   iEvent.getByLabel(srcVor_,backgrounds_);

}

void VoronoiSubtractor::offsetCorrectJets()
{

  LogDebug("VoronoiSubtractor")<<"finalizing the output...\n";

  jetOffset_.reserve(fjJets_->size());

  for (unsigned int ijet = 0;ijet <fjJets_->size();++ijet) {
     const fastjet::PseudoJet& fjJet = (*fjJets_)[ijet];

     LogDebug("VoronoiSubtractor")<<"fjJets_ "<<ijet<<"   pt : "<<fjJet.pt()<<" --- eta : "<<fjJet.eta()<<" --- phi : "<<fjJet.phi()<<endl;

     fastjet::PseudoJet subtracted;
     fastjet::PseudoJet unsubtracted;
     fastjet::PseudoJet unsubtractedDropped;
     jetOffset_[ijet] = 0;

     // Loop over constituents to determine the momentum
     // before and after subtraction

     std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(fjJet.constituents());
     for (unsigned int i=0;i<fjConstituents.size();++i) { 
	unsigned int index = fjConstituents[i].user_index();

	reco::CandidateViewRef ref(candidates_,index);
	const reco::VoronoiBackground& voronoi = (*backgrounds_)[ref];

	fastjet::PseudoJet candidate(ref->px(),ref->py(),ref->pz(),ref->energy());
	double orpt = candidate.perp();
	unsubtracted += candidate;
	candidate.reset_PtYPhiM(voronoi.pt(),ref->rapidity(),ref->phi(),ref->mass());
	LogDebug("VoronoiSubtractor")<<"cadidate "<<index<<" --- original pt : "<<orpt<<"  ---  voronoi pt : "<<voronoi.pt()<<" --- ref pt : "<<ref->pt()<<endl;
	subtracted += candidate;
     }

     // Loop over dropped candidates to see whether there is any of them
     // that would belong to this jet:

     for(unsigned int i=0; i < droppedCandidates_.size(); ++i){
	reco::CandidateViewRef ref(candidates_,droppedCandidates_[i]);
	fastjet::PseudoJet dropcand(ref->px(),ref->py(),ref->pz(),ref->energy());

	if(match(fjJet,dropcand)){
	   unsubtractedDropped += dropcand;
	}
     }

     LogDebug("VoronoiSubtractor")<<"fjJets_ "<<ijet<<"   unsubtracted : "<<unsubtracted.pt()<<endl;
     LogDebug("VoronoiSubtractor")<<"fjJets_ "<<ijet<<"   subtracted : "<<subtracted.pt()<<endl;
     LogDebug("VoronoiSubtractor")<<"fjJets_ "<<ijet<<"   dropped : "<<unsubtractedDropped.pt()<<endl;

     jetOffset_[ijet]  = unsubtracted.pt() - fjJet.pt();

  }










}

void VoronoiSubtractor::subtractPedestal(vector<fastjet::PseudoJet> & coll)
{ 

   LogDebug("VoronoiSubtractor")<<"The subtractor subtracting pedestals...\n";
   vector<fastjet::PseudoJet> newcoll;

   for (vector<fastjet::PseudoJet>::iterator input_object = coll.begin (),
	   fjInputsEnd = coll.end(); 
	input_object != fjInputsEnd; ++input_object) {
    
      reco::CandidateViewRef ref(candidates_,input_object->user_index());
      const reco::VoronoiBackground& voronoi = (*backgrounds_)[ref];

      double ptold = input_object->pt();
      double ptnew = voronoi.pt();

      LogDebug("VoronoiSubtractor")<<"pt old : "<<ptold<<" ;   pt new : "<<ptnew<<"  E : "<<input_object->e()<<" rap : "<<input_object->rapidity()<<"  phi : "<<input_object->phi()<<" MASS : "<<input_object->m()<<endl;
      
      float mScale = ptnew/ptold; 
      double candidatePtMin_ = 0;

      // If the pt of the candidate is equal to or below 0, 
      // it is removed from the input collection,
      // so that the jet clustering algorithm can function properly.
      // However, we need to keep track of these candidates
      // in order to determine how much energy has been subtracted in total.

      if(ptnew <= 0.){
	 mScale = 0.;
	 droppedCandidates_.push_back(input_object->user_index());
      }
    
      int index = input_object->user_index();
      double mass = input_object->m();
      if(mass < 0){
	 mass = 0;
      }

      //      LogDebug("VoronoiSubtractor")<<"candidate "<<int(input_object-coll.begin())<<" mScale : "<<mScale<<endl;

      double energy = sqrt(input_object->px()*input_object->px()*mScale*mScale+
			   input_object->py()*input_object->py()*mScale*mScale+
			   input_object->pz()*input_object->pz()*mScale*mScale+
			   mass*mass
			   );

      fastjet::PseudoJet ps(input_object->px()*mScale, input_object->py()*mScale,
			    input_object->pz()*mScale, energy);

      ps.set_user_index(index);

      //      LogDebug("VoronoiSubtractor")<<"New momentum : "<<ps.pt()<<"   rap : "<<ps.rap()<<"   phi : "<<ps.phi()<<" MASS : "<<ps.m()<<endl;

      if(ptnew > candidatePtMin_ || !dropZeroTowers_) newcoll.push_back(ps);
   }

   coll = newcoll;
}

void VoronoiSubtractor::calculatePedestal( vector<fastjet::PseudoJet> const & coll )
{
   LogDebug("VoronoiSubtractor")<<"do nothing...\n";
}


void VoronoiSubtractor::calculateOrphanInput(vector<fastjet::PseudoJet> & orphanInput)
{
   LogDebug("VoronoiSubtractor")<<"do nothing...\n";
}

/*
void VoronoiSubtractor::reset(std::vector<edm::Ptr<reco::Candidate> >& input,
			     std::vector<fastjet::PseudoJet>& towers,
			     std::vector<fastjet::PseudoJet>& output){
  

}

*/


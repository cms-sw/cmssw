#include "CommonTools/RecoAlgos/interface/PrimaryVertexSorting.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include <fastjet/internal/base.hh>
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"
#include "fastjet/PseudoJet.hh"

using namespace fastjet;
using namespace std;


float PrimaryVertexSorting::score(const reco::Vertex & pv,const  std::vector<const reco::Candidate *> & cands, bool useMet ) const {
  typedef math::XYZTLorentzVector LorentzVector;
  float sumPt2=0;
  float sumEt=0;
  LorentzVector met;
  std::vector<fastjet::PseudoJet> fjInputs_;  
  fjInputs_.clear();
  for (size_t i = 0 ; i < cands.size(); i++) {
    const reco::Candidate * c= cands[i];
    int absId=abs(c->pdgId());
    if(absId==13 or absId == 11) {
      float pt =c->pt();
      float ptErr = (c->bestTrack() != 0)?c->bestTrack()->ptError() :0;
      if(pt > ptErr) pt-=ptErr; else pt=0;
      sumPt2+=pt*pt;
      met+=c->p4();
      sumEt+=c->pt();
    } else {
      fjInputs_.push_back(fastjet::PseudoJet(c->px(),c->py(),c->pz(),c->p4().E()));
      fjInputs_.back().set_user_index(i);
    }
  }
  fastjet::ClusterSequence sequence( fjInputs_, JetDefinition(antikt_algorithm, 0.4));
  auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets(0));
  for (const auto & pj : jets) {
    float sumPtErr2=0;
    std::vector<fastjet::PseudoJet> constituents = pj.constituents();
    for (unsigned j = 0; j < constituents.size(); j++) {
	const reco::Candidate * c = cands[constituents[j].user_index()];
        float ptErr = (c->bestTrack() != 0)?c->bestTrack()->ptError() :0;
	sumPtErr2+=ptErr*ptErr;
    }
    auto p4 = LorentzVector( pj.px(), pj.py(), pj.pz(), pj.e() ) ;
    if(p4.pt()*p4.pt() > sumPtErr2) sumPt2+=(p4.pt()*p4.pt()-sumPtErr2)*0.8*0.8;
    met+=p4;
    sumEt+=p4.pt();
  }
  float metAbove = met.pt() - 2*sqrt(sumEt);
  if(metAbove > 0 and useMet) {
    sumPt2+=metAbove*metAbove;
  }
  return sumPt2;
}



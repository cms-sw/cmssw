#include "DQM/Physics/interface/TopDQMHelpers.h"


Calculate::Calculate(int maxNJets, double wMass): 
  failed_(false), maxNJets_(maxNJets), wMass_(wMass), massWBoson_(-1.), massTopQuark_(-1.), massBTopQuark_(-1.)
{
}

double
Calculate::massWBoson(const std::vector<reco::Jet>& jets)
{
  if(!failed_&& massWBoson_<0) operator()(jets); return massWBoson_;
}

double 
Calculate::massTopQuark(const std::vector<reco::Jet>& jets)
{ 
  if(!failed_&& massTopQuark_<0) operator()(jets); return massTopQuark_; 
}


double 
Calculate::massBTopQuark(const std::vector<reco::Jet>& jets, std::vector<double> VbtagWP, double btagWP_)
{ 
  if(!failed_&& massBTopQuark_<0) operator2(jets, VbtagWP, btagWP_); return massBTopQuark_; 
}


void
Calculate::operator()(const std::vector<reco::Jet>& jets)
{
  if(maxNJets_<0) maxNJets_=jets.size();
  failed_= jets.size()<(unsigned int) maxNJets_;
  if( failed_){ return; }

  // associate those jets with maximum pt of the vectorial 
  // sum to the hadronic decay chain
  double maxPt=-1.;
  std::vector<int> maxPtIndices;
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  maxPtIndices.push_back(-1);
  
  for(int idx=0; idx<maxNJets_; ++idx){
    for(int jdx=0; jdx<maxNJets_; ++jdx){ if(jdx<=idx) continue;
      for(int kdx=0; kdx<maxNJets_; ++kdx){ if(kdx==idx || kdx==jdx) continue;
	reco::Particle::LorentzVector sum = jets[idx].p4()+jets[jdx].p4()+jets[kdx].p4();
	if( maxPt<0. || maxPt<sum.pt() ){
	  maxPt=sum.pt();
	  maxPtIndices.clear();
	  maxPtIndices.push_back(idx);
	  maxPtIndices.push_back(jdx);
	  maxPtIndices.push_back(kdx);
	}
      }
    }
  }
  massTopQuark_= (jets[maxPtIndices[0]].p4()+
		  jets[maxPtIndices[1]].p4()+
		  jets[maxPtIndices[2]].p4()).mass();
              
  // associate those jets that get closest to the W mass
  // with their invariant mass to the W boson
  double wDist =-1.;
  std::vector<int> wMassIndices;
  wMassIndices.push_back(-1);
  wMassIndices.push_back(-1);
  for(unsigned idx=0; idx<maxPtIndices.size(); ++idx){  
    for(unsigned jdx=0; jdx<maxPtIndices.size(); ++jdx){  
      if( jdx==idx || maxPtIndices[idx]>maxPtIndices[jdx] ) continue;
	reco::Particle::LorentzVector sum = jets[maxPtIndices[idx]].p4()+jets[maxPtIndices[jdx]].p4();
	if( wDist<0. || wDist>fabs(sum.mass()-wMass_) ){
	  wDist=fabs(sum.mass()-wMass_);
	  wMassIndices.clear();
	  wMassIndices.push_back(maxPtIndices[idx]);
	  wMassIndices.push_back(maxPtIndices[jdx]);
	}
    }
  }
  massWBoson_= (jets[wMassIndices[0]].p4()+
		jets[wMassIndices[1]].p4()).mass();
}


void
Calculate::operator2(const std::vector<reco::Jet>& jets, std::vector<double> bjet, double btagWP)
{
  if(maxNJets_<0) maxNJets_=jets.size();
  failed_= jets.size()<(unsigned int) maxNJets_;
  if( failed_){ return; }
  if (jets.size() != bjet.size()){return;}

  // associate those jets with maximum pt of the vectorial 
  // sum to the hadronic decay chain. Require ONLY 1 btagged jet
  double maxBPt=-1.;
  std::vector<int> maxBPtIndices;
  maxBPtIndices.push_back(-1);
  maxBPtIndices.push_back(-1);
  maxBPtIndices.push_back(-1);
  for(int idx=0; idx<maxNJets_; ++idx){
    for(int jdx=0; jdx<maxNJets_; ++jdx){ if(jdx<=idx) continue;
        for(int kdx=0; kdx<maxNJets_; ++kdx){if(kdx==idx || kdx==jdx) continue;
        //require only 1b-jet
            if ((bjet[idx]>  btagWP && bjet[jdx]<= btagWP && bjet[kdx]<= btagWP) ||
                (bjet[idx]<= btagWP && bjet[jdx]>  btagWP && bjet[kdx]<= btagWP) ||
                (bjet[idx]<= btagWP && bjet[jdx]<= btagWP && bjet[kdx]>  btagWP) ){
                    reco::Particle::LorentzVector sum = jets[idx].p4()+jets[jdx].p4()+jets[kdx].p4();
                    if( maxBPt<0. || maxBPt<sum.pt() ){
                        maxBPt=sum.pt();
                        maxBPtIndices.clear();
                        maxBPtIndices.push_back(idx);
                        maxBPtIndices.push_back(jdx);
                        maxBPtIndices.push_back(kdx);
                    }
            }
        }
    }
  }
  if (maxBPtIndices[0]<0 || maxBPtIndices[1]<0 || maxBPtIndices[2]<0) return;
  massBTopQuark_= (jets[maxBPtIndices[0]].p4()+
                   jets[maxBPtIndices[1]].p4()+
                   jets[maxBPtIndices[2]].p4()).mass();
}
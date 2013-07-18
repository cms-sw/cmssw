#include "DataFormats/TauReco/interface/PFTau3ProngSummary.h"
#include "TMatrixT.h"
#include "TMatrixTSym.h"
#include "RecoVertex/VertexTools/interface/VertexDistance3D.h"
#include "TVectorT.h"
using namespace reco;

PFTau3ProngSummary::PFTau3ProngSummary(reco::PFTauTransverseImpactParameterRef TIP,TLorentzVector a1,double vertex_chi2,double vertex_ndf){
  TIP_=TIP;
  for(unsigned int i=0;i<nsolutions;i++){
    has3ProngSolution_.push_back(false);
    solution_Chi2_.push_back(0);
    thetaGJsig_.push_back(0);
    tau_p4_.push_back(TLorentzVector(0,0,0,0));
    daughter_PDGID_.push_back(std::vector<int>());
    daughter_charge_.push_back(std::vector<int>());
    daughter_p4_.push_back(std::vector<TLorentzVector>());
  }
  a1_=a1;
  sv_=TVector3(TIP_->secondaryVertex()->x(),TIP_->secondaryVertex()->y(),TIP_->secondaryVertex()->z());
  svcov_=TIP_->secondaryVertexCov();
  vertex_chi2_=vertex_chi2;
  vertex_ndf_=vertex_ndf;
}

PFTau3ProngSummary::PFTau3ProngSummary(reco::PFTauTransverseImpactParameterRef TIP,TLorentzVector a1,double vertex_chi2,double vertex_ndf,TVector3 sv,CovMatrix svcov){
  TIP_=TIP;
  for(unsigned int i=0;i<nsolutions;i++){
    has3ProngSolution_.push_back(false);
    solution_Chi2_.push_back(0);
    thetaGJsig_.push_back(0);
    tau_p4_.push_back(TLorentzVector(0,0,0,0));
    daughter_PDGID_.push_back(std::vector<int>());
    daughter_charge_.push_back(std::vector<int>());
    daughter_p4_.push_back(std::vector<TLorentzVector>());
  }
  a1_=a1;
  sv_=sv;
  svcov_=svcov;
  vertex_chi2_=vertex_chi2;
  vertex_ndf_=vertex_ndf;
}


PFTau3ProngSummary::PFTau3ProngSummary(){
  for(unsigned int i=0;i<nsolutions;i++){
    has3ProngSolution_.push_back(false);
    solution_Chi2_.push_back(0);
    thetaGJsig_.push_back(0);
    tau_p4_.push_back(TLorentzVector(0,0,0,0));
    daughter_PDGID_.push_back(std::vector<int>());
    daughter_charge_.push_back(std::vector<int>());
    daughter_p4_.push_back(std::vector<TLorentzVector>());
  }
}

PFTau3ProngSummary* PFTau3ProngSummary::clone() const{
  return new PFTau3ProngSummary(*this);
}


bool PFTau3ProngSummary::AddSolution(unsigned int solution,TLorentzVector tau, std::vector<TLorentzVector> daughter_p4,
			 std::vector<int> daughter_charge,std::vector<int> daughter_PDGID,
			 bool has3ProngSolution,double solutionChi2,double thetaGJsig){
  if(solution<nsolutions){
    has3ProngSolution_.at(solution)=true;
    solution_Chi2_.at(solution)=solutionChi2;
    thetaGJsig_.at(solution)=thetaGJsig;
    tau_p4_.at(solution)=tau;
    daughter_PDGID_.at(solution)=daughter_PDGID;
    daughter_charge_.at(solution)=daughter_charge;
    daughter_p4_.at(solution)=daughter_p4;
    return true;
  }
  return false;
}


double PFTau3ProngSummary::M_12()const{
  for(unsigned int i=0;i<has3ProngSolution_.size();i++){
    if(has3ProngSolution_.at(i)==true){
      int charge=Tau_Charge();
      TLorentzVector LV;
      for(unsigned int j=0;j<daughter_p4_.at(i).size();j++){
	if(daughter_charge_.at(i).at(j)==charge)LV+=daughter_p4_.at(i).at(j);
      }
      return LV.M();
    }
  }
  return 0.0;
}
double PFTau3ProngSummary::M_13()const{
  for(unsigned int i=0;i<has3ProngSolution_.size();i++){
    if(has3ProngSolution_.at(i)==true){
      int charge=Tau_Charge();
      TLorentzVector LV_opp;
      for(unsigned int j=0;j<daughter_p4_.at(i).size();j++){
        if(daughter_charge_.at(i).at(j)==-1*charge)LV_opp=daughter_p4_.at(i).at(j);
      }
      TLorentzVector LV_pair;
      bool found(false);
      for(unsigned int j=0;j<daughter_p4_.at(i).size();j++){
        if(daughter_charge_.at(i).at(j)==charge){
	  TLorentzVector LV=daughter_p4_.at(i).at(j);
	  LV+=LV_opp;
	  if(!found)LV_pair=LV;
	  else if(LV_pair.M()>LV.M())LV_pair=LV;
	  found=true;
	}
      }
      if(found)return LV_pair.M();
    }
  }
  return 0.0;
}

double PFTau3ProngSummary::M_23()const{
  for(unsigned int i=0;i<has3ProngSolution_.size();i++){
    if(has3ProngSolution_.at(i)==true){
      int charge=Tau_Charge();
      TLorentzVector LV_opp;
      for(unsigned int j=0;j<daughter_p4_.at(i).size();j++){
        if(daughter_charge_.at(i).at(j)==-1*charge)LV_opp=daughter_p4_.at(i).at(j);
      }
      TLorentzVector LV_pair;
      bool found(false);
      for(unsigned int j=0;j<daughter_p4_.at(i).size();j++){
        if(daughter_charge_.at(i).at(j)==charge){
          TLorentzVector LV=daughter_p4_.at(i).at(j);
          LV+=LV_opp;
          if(!found)LV_pair=LV;
          else if(LV_pair.M()<LV.M())LV_pair=LV;
          found=true;
        }
      }
      if(found)return LV_pair.M();
    }
  }
  return 0.0;
}

int PFTau3ProngSummary::Tau_Charge()const{
  for(unsigned int i=0;i<has3ProngSolution_.size();i++){
    if(has3ProngSolution_.at(i)==true){
      int charge;
      for(unsigned int j=0;j<daughter_p4_.at(i).size();j++)charge+=daughter_charge_.at(i).at(j);
      return charge;
    }
  }
  return 0;
}

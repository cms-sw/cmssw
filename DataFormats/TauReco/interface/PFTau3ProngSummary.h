#ifndef DataFormats_TauReco_PFTau3ProngSummary_h
#define DataFormats_TauReco_PFTau3ProngSummary_h

/* class PFTau3ProngSummary
 * 
 * Stores information on the 3 prong summary for a fully reconstructed tau lepton
 *
 * author: Ian M. Nugent
 * The idea of the fully reconstructing the tau using a kinematic fit comes from
 * Lars Perchalla and Philip Sauerland Theses under Achim Stahl supervision. This
 * code is a result of the continuation of this work by Ian M. Nugent and Vladimir Cherepanov.
 */
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Error.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameter.h"
#include "DataFormats/TauReco/interface/PFTauTransverseImpactParameterFwd.h"
#include "TVector3.h"
#include "TLorentzVector.h"
#include "TMath.h"

namespace reco {
   class PFTau3ProngSummary {
   public:
     enum {ambiguity,minus,plus,nsolutions};
     enum { dimension = 3 };
     enum { covarianceSize = dimension * ( dimension + 1 ) / 2 };
     typedef math::Error<dimension>::type CovMatrix;
     typedef math::XYZPoint Point;
     typedef math::XYZVector Vector;

     /// constructor from values
     PFTau3ProngSummary();
     PFTau3ProngSummary(reco::PFTauTransverseImpactParameterRef TIP,TLorentzVector a1,double vertex_chi2,double vertex_ndf);
     PFTau3ProngSummary(reco::PFTauTransverseImpactParameterRef TIP,TLorentzVector a1,double vertex_chi2,double vertex_ndf,TVector3 sv,CovMatrix svcov);

     virtual ~PFTau3ProngSummary(){}

     PFTau3ProngSummary* clone() const;

     virtual bool AddSolution(unsigned int solution,TLorentzVector tau, std::vector<TLorentzVector> daughter_p4,
			      std::vector<int> daughter_charge,std::vector<int> daughter_PDGID,
			      bool has3ProngSolution,double solutionChi2,double thetaGJsig);
     
     const reco::PFTauTransverseImpactParameterRef PFTauTIP()const{return TIP_;}
     // interface for relevant TIP functions;
     const VertexRef      primaryVertex()const{return TIP_->primaryVertex();}
     const CovMatrix      primaryVertexCov()const{return TIP_->primaryVertexCov();}
     const bool           hasSecondaryVertex()const{return TIP_->hasSecondaryVertex();}
     const VertexRef      secondaryVertex()const{return TIP_->secondaryVertex();}
     const CovMatrix      secondaryVertexCov()const{return TIP_->secondaryVertexCov();}
     const Vector         flightLength()const{return TIP_->flightLength();}
     const double         flightLengthSig()const{return TIP_->flightLengthSig();}
     const CovMatrix      flightLenghtCov()const{return TIP_->flightLengthCov();}

     // Tau 3 prong functions
     const TLorentzVector A1_LV()const{return a1_;}
     const double         M_A1()const{return a1_.M();}
           double         M_12()const; //pi-pi- 
           double         M_13()const; //pi-pi+ Dalitz masses
           double         M_23()const; //pi-pi+ Dalitz masses
           int            Tau_Charge()const;
     const TVector3       HelixFitSecondaryVertex()const{return sv_;}
     const CovMatrix      HelixFitSecondaryVertexCov()const{return svcov_;}
     const double         Vertex_chi2()const{return vertex_chi2_;}
     const double         Vertex_ndf()const{return vertex_ndf_;}
     const double         Vertex_Prob()const{return TMath::Prob(vertex_chi2_,vertex_ndf_);}
     const bool                        has3ProngSolution(unsigned int i)const{return has3ProngSolution_.at(i);}
     const double                      Solution_Chi2(unsigned int i)const{return solution_Chi2_.at(i);}
     const double                      SignificanceOfThetaGJ(unsigned int i)const{return thetaGJsig_.at(i);} // 0 or less means the theta_GF has 
                                                                                                  // a physical solution 
     const TLorentzVector              Tau(unsigned int i)const{return tau_p4_.at(i);}
     const std::vector<int>            Daughter_PDGID(unsigned int i)const{return daughter_PDGID_.at(i);}
     const std::vector<int>            Daughter_Charge(unsigned int i)const{return daughter_charge_.at(i);}
     const std::vector<TLorentzVector> Daughter_P4(unsigned int i)const{return daughter_p4_.at(i);}
     
   private:
     reco::PFTauTransverseImpactParameterRef TIP_;
     TLorentzVector a1_;
     TVector3       sv_;
     CovMatrix      svcov_;
     double         vertex_chi2_;
     double         vertex_ndf_;
     std::vector<bool>   has3ProngSolution_;
     std::vector<double> solution_Chi2_;
     std::vector<double> thetaGJsig_;
     std::vector<TLorentzVector>               tau_p4_;
     std::vector<std::vector<int> >            daughter_PDGID_;
     std::vector<std::vector<int> >            daughter_charge_;
     std::vector<std::vector<TLorentzVector> > daughter_p4_;
     
   };
}

#endif

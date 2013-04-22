/* L2TauIsolationInfo Class
Holds output of the Tau L2 IsolationProducer
 
Author: Michail Bachtis
University of Wisconsin-Madison
e-mail: bachtis@hep.wisc.edu
*/



#ifndef L2TAUISOLATION_INFO_H
#define L2TAUISOLATION_INFO_H
#include <vector>

namespace reco {

class L2TauIsolationInfo
{
 public:
  L2TauIsolationInfo()
    {
      ecalIsolEt_=0.; 
      seedEcalHitEt_=-1.;
      ecalClusterShape_.push_back(0.);
      ecalClusterShape_.push_back(0.);
      ecalClusterShape_.push_back(0.);
      nEcalHits_=0;

      hcalIsolEt_=0.; 
      seedHcalHitEt_=-1.;
      hcalClusterShape_.push_back(0.);
      hcalClusterShape_.push_back(0.);
      hcalClusterShape_.push_back(0.);
      nHcalHits_=0;
    }


  ~L2TauIsolationInfo()
    {

    }

  //getters
  double ecalIsolEt() const {return ecalIsolEt_;}
  double seedEcalHitEt() const {return seedEcalHitEt_;}
  std::vector<double> ecalClusterShape() const {return ecalClusterShape_;}
  int nEcalHits() const {return nEcalHits_;}

  double hcalIsolEt() const {return hcalIsolEt_;}
  double seedHcalHitEt() const {return seedHcalHitEt_;}
  std::vector<double> hcalClusterShape() const {return hcalClusterShape_;}
  int nHcalHits() const {return nHcalHits_;}

  //setters
  void setEcalIsolEt(double et) { ecalIsolEt_ = et;}
  void setSeedEcalHitEt(double et) { seedEcalHitEt_ = et;}
  void setEcalClusterShape(const std::vector<double>& shape) { ecalClusterShape_ = shape;}
  void setNEcalHits(int hits) { nEcalHits_ = hits;}
  void setHcalIsolEt(double et) { hcalIsolEt_ = et;}
  void setSeedHcalHitEt(double et) { seedHcalHitEt_ = et;}
  void setHcalClusterShape(const std::vector<double>& shape) { hcalClusterShape_ = shape;}
  void setNHcalHits(int hits) { nHcalHits_ = hits;}
   
 private:

  //ECAL Isolation 
  double ecalIsolEt_; 
  double seedEcalHitEt_;
  std::vector<double> ecalClusterShape_;
  int nEcalHits_;


  //HCAL Isolation
  double hcalIsolEt_;
  double seedHcalHitEt_;
  std::vector<double> hcalClusterShape_;
  int nHcalHits_;

};

}
#endif


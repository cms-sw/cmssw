//
// Author:  Steven Lowette
// Created: Wed May  2 16:48:32 PDT 2007
//
// $Id: TopLepton.h,v 1.6 2007/06/23 07:06:04 lowette Exp $
//

#ifndef TopObjects_TopLepton_h
#define TopObjects_TopLepton_h

/**
  \class    TopLepton TopLepton.h "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
  \brief    High-level top lepton container

   TopLepton contains a lepton as a TopObject, and provides the means to
   store and retrieve the high-level likelihood ratio information.

  \author   Steven Lowette
  \version  $Id: TopLepton.h,v 1.6 2007/06/23 07:06:04 lowette Exp $
*/


#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


typedef reco::PixelMatchGsfElectron TopElectronType;
typedef reco::Muon TopMuonType;


template <class LeptonType>
class TopLepton : public TopObject<LeptonType> {

  friend class TopElectronProducer;
  friend class TopMuonProducer;
  friend class TtSemiKinFitterEMom;
  friend class TtSemiKinFitterEtEtaPhi;
  friend class TtSemiKinFitterEtThetaPhi;
  friend class StKinFitterEMom;
  friend class StKinFitterEtEtaPhi;
  friend class StKinFitterEtThetaPhi;
  friend class TopLeptonLRCalc;

  public:

    TopLepton();
    TopLepton(const LeptonType & aLepton);
    virtual ~TopLepton();

    reco::GenParticleCandidate getGenLepton() const;
    TopParticle                getFitLepton() const;
    double                     getLRVar(const unsigned int i) const;
    double                     getLRVal(const unsigned int i) const;
    double                     getLRComb() const;
    double                     getTrackIso() const;
    double                     getCaloIso() const;
    
  protected:

    void setGenLepton(const reco::GenParticleCandidate & gl);
    void setFitLepton(const TopParticle & fl);
    void setLRVarVal(const std::pair<double, double> lrVarVal, const unsigned int i);
    void setLRComb(const double lr);
    void setTrackIso(const double trackIso);
    void setCaloIso(const double caloIso);
    unsigned int getLRSize() const;

  protected:

    std::vector<reco::GenParticleCandidate> genLepton_;
    std::vector<TopParticle>                fitLepton_;
    std::vector<std::pair<double, double> > lrVarVal_;
    double lrComb_;
    double trackIso_;
    double caloIso_;

};


/// default constructor
template <class LeptonType>
TopLepton<LeptonType>::TopLepton() :
    TopObject<LeptonType>(LeptonType()),
    lrComb_(0) {
  // no common constructor, so initialize the candidate manually
  this->setCharge(0);
  this->setP4(reco::Particle::LorentzVector(0, 0, 0, 0));
  this->setVertex(reco::Particle::Point(0, 0, 0));
}


/// constructor from LeptonType
template <class LeptonType>
TopLepton<LeptonType>::TopLepton(const LeptonType & aLepton) :
  TopObject<LeptonType>(aLepton),
  lrComb_(0) {
}


/// destructor
template <class LeptonType>
TopLepton<LeptonType>::~TopLepton() {
}


/// return the match to the generated lepton
template <class LeptonType>
reco::GenParticleCandidate TopLepton<LeptonType>::getGenLepton() const {
  return (genLepton_.size() > 0 ?
    genLepton_.front() :
    reco::GenParticleCandidate(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), 0, 0)
  );
}


/// return the fitted lepton
template <class LeptonType>
TopParticle  TopLepton<LeptonType>::getFitLepton() const {
  return (genLepton_.size() > 0 ?
    fitLepton_.front() :
    TopParticle()
  );
}


/// return the i'th lepton likelihood ratio variable
template <class LeptonType>
double TopLepton<LeptonType>::getLRVar(const unsigned int i) const {
  return (i < lrVarVal_.size() ? lrVarVal_[i].first  : 0);
}


/// return the lepton likelihood value for the i'th variable
template <class LeptonType>
double TopLepton<LeptonType>::getLRVal(const unsigned int i) const {
  return (i < lrVarVal_.size() ? lrVarVal_[i].second : 1);
}


/// return the combined lepton likelihood ratio value
template <class LeptonType>
double TopLepton<LeptonType>::getLRComb() const {
  return lrComb_;
}

template <class LeptonType>
double TopLepton<LeptonType>::getTrackIso() const {
  return trackIso_;
}

template <class LeptonType>
double TopLepton<LeptonType>::getCaloIso() const {
  return caloIso_;
}




/// method to set the generated lepton
template <class LeptonType>
void TopLepton<LeptonType>::setGenLepton(const reco::GenParticleCandidate & gl) {
  genLepton_.clear();
  genLepton_.push_back(gl);
}


/// method to set the fitted lepton
template <class LeptonType>
void TopLepton<LeptonType>::setFitLepton(const TopParticle & fl) {
  fitLepton_.clear();
  fitLepton_.push_back(fl);
}


/// method to set the i'th lepton LR variable and value pair
template <class LeptonType>
void TopLepton<LeptonType>::setLRVarVal(const std::pair<double, double> lrVarVal, const unsigned int i) {
  while (lrVarVal_.size() <= i) lrVarVal_.push_back(std::pair<double, double>(0, 1));
  lrVarVal_[i] = lrVarVal;
}


/// method to set the combined lepton likelihood ratio
template <class LeptonType>
void TopLepton<LeptonType>::setLRComb(const double lr) {
  lrComb_ = lr;
}

template <class LeptonType>
void TopLepton<LeptonType>::setTrackIso(const double trackIso) {
  trackIso_=trackIso;
}

template <class LeptonType>
void TopLepton<LeptonType>::setCaloIso(const double caloIso) {
  caloIso_=caloIso;
}



/// method to give back the size of the LR vector
template <class LeptonType>
unsigned int TopLepton<LeptonType>::getLRSize() const {
  return lrVarVal_.size();
}


/// definition of TopElectron as a TopLepton of TopElectronType
typedef TopLepton<TopElectronType> TopElectron;
/// definition of TopMuon as a TopLepton of TopMuonType
typedef TopLepton<TopMuonType> TopMuon;


#endif

//
// $Id: TopLepton.h,v 1.8 2007/07/05 23:34:19 lowette Exp $
//

#ifndef TopObjects_TopLepton_h
#define TopObjects_TopLepton_h

/**
  \class    TopLepton TopLepton.h "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
  \brief    High-level top lepton container

   TopLepton contains a lepton as a TopObject, and provides the means to
   store and retrieve the high-level likelihood ratio information.

  \author   Steven Lowette
  \version  $Id: TopLepton.h,v 1.8 2007/07/05 23:34:19 lowette Exp $
*/


#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"


typedef reco::PixelMatchGsfElectron TopElectronType;
typedef reco::Muon TopMuonType;
typedef reco::IsolatedTauTagInfo TopTauType;


template <class LeptonType>
class TopLepton : public TopObject<LeptonType> {

  friend class TopElectronProducer;
  friend class TopMuonProducer;
  friend class TopTauProducer;
  friend class TopLeptonLRCalc;

  public:

    TopLepton();
    TopLepton(const LeptonType & aLepton);
    virtual ~TopLepton();

    reco::GenParticleCandidate getGenLepton() const;
    double                     getTrackIso() const;
    double                     getCaloIso() const;
    double                     getLRVar(const unsigned int i) const;
    double                     getLRVal(const unsigned int i) const;
    double                     getLRComb() const;
    
  protected:

    void setGenLepton(const reco::GenParticleCandidate & gl);
    void setTrackIso(const double trackIso);
    void setCaloIso(const double caloIso);
    void setLRVarVal(const std::pair<double, double> lrVarVal, const unsigned int i);
    void setLRComb(const double lr);
    unsigned int getLRSize() const;

  protected:

    std::vector<reco::GenParticleCandidate> genLepton_;
    double trackIso_;
    double caloIso_;
    std::vector<std::pair<double, double> > lrVarVal_;
    double lrComb_;

};


/// default constructor
template <class LeptonType>
TopLepton<LeptonType>::TopLepton() :
    TopObject<LeptonType>(LeptonType()),
    lrComb_(0) {
  /*
  // no common constructor, so initialize the candidate manually
  this->setCharge(0);
  this->setP4(reco::Particle::LorentzVector(0, 0, 0, 0));
  this->setVertex(reco::Particle::Point(0, 0, 0));
  */
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


/// return the tracker isolation variable
template <class LeptonType>
double TopLepton<LeptonType>::getTrackIso() const {
  return trackIso_;
}


/// return the calorimeter isolation variable
template <class LeptonType>
double TopLepton<LeptonType>::getCaloIso() const {
  return caloIso_;
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


/// method to set the generated lepton
template <class LeptonType>
void TopLepton<LeptonType>::setGenLepton(const reco::GenParticleCandidate & gl) {
  genLepton_.clear();
  genLepton_.push_back(gl);
}


/// method to set the tracker isolation variable
template <class LeptonType>
void TopLepton<LeptonType>::setTrackIso(double trackIso) {
  trackIso_ = trackIso;
}


/// method to set the calorimeter isolation variable
template <class LeptonType>
void TopLepton<LeptonType>::setCaloIso(double caloIso) {
  caloIso_ = caloIso;
}


/// method to set the i'th lepton LR variable and value pair
template <class LeptonType>
void TopLepton<LeptonType>::setLRVarVal(const std::pair<double, double> lrVarVal, const unsigned int i) {
  while (lrVarVal_.size() <= i) lrVarVal_.push_back(std::pair<double, double>(0, 1));
  lrVarVal_[i] = lrVarVal;
}


/// method to set the combined lepton likelihood ratio
template <class LeptonType>
void TopLepton<LeptonType>::setLRComb(double lr) {
  lrComb_ = lr;
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
/// definition of TopTau as a TopLepton of TopTauType
typedef TopLepton<TopTauType> TopTau;


#endif

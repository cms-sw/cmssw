//
// $Id: Lepton.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Lepton_h
#define DataFormats_PatCandidates_Lepton_h

/**
  \class    Lepton Lepton.h "DataFormats/PatCandidates/interface/Lepton.h"
  \brief    Analysis-level lepton class

   Lepton implements the analysis-level charged lepton class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Lepton.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  class TauProducer;
  class LeptonLRCalc;


  template <class LeptonType>
  class Lepton : public PATObject<LeptonType> {

    friend class PATTauProducer;
    friend class LeptonLRCalc;

    public:

      Lepton();
      Lepton(const LeptonType & aLepton);
      virtual ~Lepton();

      reco::GenParticleCandidate getGenLepton() const;
      float getLRVar(const unsigned int i) const;
      float getLRVal(const unsigned int i) const;
      float getLRComb() const;

    protected:

      void setGenLepton(const reco::GenParticleCandidate & gl);
      void setLRVarVal(const std::pair<float, float> lrVarVal, const unsigned int i);
      void setLRComb(const float lr);
      unsigned int getLRSize() const;

    protected:

      std::vector<reco::GenParticleCandidate> genLepton_;
      std::vector<std::pair<float, float> > lrVarVal_;
      float lrComb_;

  };


  /// default constructor
  template <class LeptonType>
  Lepton<LeptonType>::Lepton() :
    PATObject<LeptonType>(LeptonType()),
    lrComb_(0) {
    // no common constructor, so initialize the candidate manually
    this->setCharge(0);
    this->setP4(reco::Particle::LorentzVector(0, 0, 0, 0));
    this->setVertex(reco::Particle::Point(0, 0, 0));
  }


  /// constructor from LeptonType
  template <class LeptonType>
  Lepton<LeptonType>::Lepton(const LeptonType & aLepton) :
    PATObject<LeptonType>(aLepton),
    lrComb_(0) {
  }


  /// destructor
  template <class LeptonType>
  Lepton<LeptonType>::~Lepton() {
  }


  /// return the match to the generated lepton
  template <class LeptonType>
  reco::GenParticleCandidate Lepton<LeptonType>::getGenLepton() const {
    return (genLepton_.size() > 0 ?
      genLepton_.front() :
      reco::GenParticleCandidate(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), 0, 0, true)
    );
  }


  /// return the i'th lepton likelihood ratio variable
  template <class LeptonType>
  float Lepton<LeptonType>::getLRVar(const unsigned int i) const {
    return (i < lrVarVal_.size() ? lrVarVal_[i].first  : 0);
  }


  /// return the lepton likelihood value for the i'th variable
  template <class LeptonType>
  float Lepton<LeptonType>::getLRVal(const unsigned int i) const {
    return (i < lrVarVal_.size() ? lrVarVal_[i].second : 1);
  }


  /// return the combined lepton likelihood ratio value
  template <class LeptonType>
  float Lepton<LeptonType>::getLRComb() const {
    return lrComb_;
  }


  /// method to give back the size of the LR vector
  template <class LeptonType>
  unsigned int Lepton<LeptonType>::getLRSize() const {
    return lrVarVal_.size();
  }


  /// method to set the generated lepton
  template <class LeptonType>
  void Lepton<LeptonType>::setGenLepton(const reco::GenParticleCandidate & gl) {
    genLepton_.clear();
    genLepton_.push_back(gl);
  }


  /// method to set the i'th lepton LR variable and value pair
  template <class LeptonType>
  void Lepton<LeptonType>::setLRVarVal(const std::pair<float, float> lrVarVal, const unsigned int i) {
    while (lrVarVal_.size() <= i) lrVarVal_.push_back(std::pair<float, float>(0, 1));
    lrVarVal_[i] = lrVarVal;
  }


  /// method to set the combined lepton likelihood ratio
  template <class LeptonType>
  void Lepton<LeptonType>::setLRComb(float lr) {
    lrComb_ = lr;
  }


}

#endif

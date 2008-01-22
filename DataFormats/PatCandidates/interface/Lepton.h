//
// $Id: Lepton.h,v 1.3 2008/01/21 22:01:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Lepton_h
#define DataFormats_PatCandidates_Lepton_h

/**
  \class    Lepton Lepton.h "DataFormats/PatCandidates/interface/Lepton.h"
  \brief    Analysis-level lepton class

   Lepton implements the analysis-level charged lepton class within the 'pat'
   namespace.

  \author   Steven Lowette
  \version  $Id: Lepton.h,v 1.3 2008/01/21 22:01:25 lowette Exp $
*/

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/PatCandidates/interface/PATObject.h"


namespace pat {


  template <class LeptonType>
  class Lepton : public PATObject<LeptonType> {

    public:

      Lepton();
      Lepton(const LeptonType & aLepton);
      Lepton(const edm::Ref<std::vector<LeptonType> > & aLeptonRef);
      virtual ~Lepton();

      const reco::Particle * genLepton() const;
      float lrVar(const unsigned int i) const;
      float lrVal(const unsigned int i) const;
      float lrComb() const;
      unsigned int lrSize() const;

      void setGenLepton(const reco::Particle & gl);
      void setLRVarVal(const std::pair<float, float> lrVarVal, const unsigned int i);
      void setLRComb(const float lr);

    protected:

      std::vector<reco::Particle> genLepton_;
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


  /// constructor from ref to LeptonType
  template <class LeptonType>
  Lepton<LeptonType>::Lepton(const edm::Ref<std::vector<LeptonType> > & aLeptonRef) :
    PATObject<LeptonType>(aLeptonRef),
    lrComb_(0) {
  }


  /// destructor
  template <class LeptonType>
  Lepton<LeptonType>::~Lepton() {
  }


  /// return the match to the generated lepton
  template <class LeptonType>
  const reco::Particle * Lepton<LeptonType>::genLepton() const {
    return (genLepton_.size() > 0 ? &genLepton_.front() : 0);
  }


  /// return the i'th lepton likelihood ratio variable
  template <class LeptonType>
  float Lepton<LeptonType>::lrVar(const unsigned int i) const {
    return (i < lrVarVal_.size() ? lrVarVal_[i].first  : 0);
  }


  /// return the lepton likelihood value for the i'th variable
  template <class LeptonType>
  float Lepton<LeptonType>::lrVal(const unsigned int i) const {
    return (i < lrVarVal_.size() ? lrVarVal_[i].second : 1);
  }


  /// return the combined lepton likelihood ratio value
  template <class LeptonType>
  float Lepton<LeptonType>::lrComb() const {
    return lrComb_;
  }


  /// method to give back the size of the LR vector
  template <class LeptonType>
  unsigned int Lepton<LeptonType>::lrSize() const {
    return lrVarVal_.size();
  }


  /// method to set the generated lepton
  template <class LeptonType>
  void Lepton<LeptonType>::setGenLepton(const reco::Particle & gl) {
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

//
// Author:  Steven Lowette
// Created: Wed May  2 16:48:32 PDT 2007
//
// $Id: TopLepton.h,v 1.4 2007/05/30 19:13:18 lowette Exp $
//

#ifndef TopLepton_h
#define TopLepton_h

/**
  \class    TopLepton TopLepton.h "AnalysisDataFormats/TopObjects/interface/TopLepton.h"
  \brief    High-level top lepton container

   TopLepton contains a lepton as a TopObject, and provides the means to
   store and retrieve the high-level likelihood ratio information.

  \author   Steven Lowette
  \version  $Id: TopLepton.h,v 1.4 2007/05/30 19:13:18 lowette Exp $
*/


#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"

#include "AnalysisDataFormats/TopObjects/interface/TopObject.h"
#include "AnalysisDataFormats/TopObjects/interface/TopParticle.h"


typedef reco::PixelMatchGsfElectron ElectronType;
typedef reco::Muon MuonType;
typedef reco::GenParticleCandidate GenPartType;


template <class LeptonType>
class TopLepton : public TopObject<LeptonType> {

  friend class TopLeptonLRCalc;

  public:

    TopLepton() {}
    TopLepton(LeptonType aLep): TopObject<LeptonType>(aLep) {}
    virtual ~TopLepton() {}

    GenPartType getGenLepton() const;
    TopParticle getFitLepton() const;
    double      getLRVar(const unsigned int i) const;
    double      getLRVal(const unsigned int i) const;
    double      getLRComb() const;
    void        setGenLepton(const GenPartType & gl);
    void        setFitLepton(const TopParticle & fl);

  protected:

    unsigned int getLRSize() const;
    void setLRVarVal(const std::pair<double, double> lrVarVal, const unsigned int i);
    void setLRComb(const double lr);

  protected:

    GenPartType genLepton;
    TopParticle fitLepton;
    double lrComb_;
    std::vector<std::pair<double, double> > lrVarVal_;

};


template <class LeptonType> 	GenPartType  TopLepton<LeptonType>::getGenLepton() const                 { return genLepton; }  
template <class LeptonType> 	TopParticle  TopLepton<LeptonType>::getFitLepton() const                 { return fitLepton; }
template <class LeptonType> 	double       TopLepton<LeptonType>::getLRVar(const unsigned int i) const { return (i < lrVarVal_.size() ? lrVarVal_[i].first  : 0); }
template <class LeptonType> 	double       TopLepton<LeptonType>::getLRVal(const unsigned int i) const { return (i < lrVarVal_.size() ? lrVarVal_[i].second : 1); }
template <class LeptonType> 	double       TopLepton<LeptonType>::getLRComb() const                    { return lrComb_; }
template <class LeptonType> 	unsigned int TopLepton<LeptonType>::getLRSize() const                    { return lrVarVal_.size(); }

template <class LeptonType> 	void         TopLepton<LeptonType>::setGenLepton(const GenPartType & gl) { genLepton = gl; }
template <class LeptonType> 	void         TopLepton<LeptonType>::setFitLepton(const TopParticle & fl) { fitLepton = fl; }
template <class LeptonType> 	void         TopLepton<LeptonType>::setLRVarVal(const std::pair<double, double> lrVarVal, const unsigned int i) {
  while (lrVarVal_.size() <= i) lrVarVal_.push_back(std::pair<double, double>(0, 1));
  lrVarVal_[i] = lrVarVal;
}
template <class LeptonType> 	void         TopLepton<LeptonType>::setLRComb(const double lr)           { lrComb_ = lr; }


typedef TopLepton<ElectronType> TopElectron;
typedef TopLepton<MuonType> TopMuon;


#endif

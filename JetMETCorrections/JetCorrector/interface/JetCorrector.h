#ifndef JetMETCorrections_JetCorrector_JetCorrector_h
#define JetMETCorrections_JetCorrector_JetCorrector_h
// -*- C++ -*-
//
// Package:     JetMETCorrections/JetCorrector
// Class  :     reco::JetCorrector
//
/**\class reco::JetCorrector JetCorrector.h "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 29 Aug 2014 15:42:37 GMT
//

// system include files
#include <memory>

// user include files
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"

// forward declarations

namespace reco {

  class JetCorrector {
  public:
    JetCorrector();
    JetCorrector(std::unique_ptr<JetCorrectorImpl const> fImpl) : impl_(std::move(fImpl)) {}
    JetCorrector(JetCorrector&&) = default;
    JetCorrector& operator=(JetCorrector&&) = default;
    JetCorrector(const JetCorrector&) = delete;
    JetCorrector& operator=(const JetCorrector&) = delete;

    typedef reco::Particle::LorentzVector LorentzVector;

    // ---------- const member functions ---------------------
    /// get correction using Jet information only
    double correction(const LorentzVector& fJet) const { return impl_->correction(fJet); }

    /// apply correction using Jet information only
    double correction(const reco::Jet& fJet) const { return impl_->correction(fJet); }

    /// apply correction using Ref
    double correction(const reco::Jet& fJet, const edm::RefToBase<reco::Jet>& fJetRef) const {
      return impl_->correction(fJet, fJetRef);
    }

    /// Apply vectorial correction
    double correction(const reco::Jet& fJet, const edm::RefToBase<reco::Jet>& fJetRef, LorentzVector& corrected) const {
      return impl_->correction(fJet, fJetRef, corrected);
    }

    /// if correction needs the jet reference
    bool refRequired() const { return impl_->refRequired(); }

    /// if vectorial correction is provided
    bool vectorialCorrection() const { return impl_->vectorialCorrection(); }

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------

  private:
    // ---------- member data --------------------------------
    std::unique_ptr<JetCorrectorImpl const> impl_;
  };
}  // namespace reco

#endif

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
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#endif
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"

// forward declarations

namespace reco {

  class JetCorrector
  {
    
  public:
    JetCorrector();
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
    JetCorrector(std::unique_ptr<JetCorrectorImpl const> fImpl):impl_(std::move(fImpl)) {}
    
    typedef reco::Particle::LorentzVector LorentzVector;

    // ---------- const member functions ---------------------
    /// get correction using Jet information only
    double correction (const LorentzVector& fJet) const {
      return impl_->correction(fJet);
    }

    /// apply correction using Jet information only
    double correction (const reco::Jet& fJet) const {
      return impl_->correction(fJet);
    }

    /// apply correction using Ref
    double correction (const reco::Jet& fJet,
		       const edm::RefToBase<reco::Jet>& fJetRef) const {
      return impl_->correction(fJet,fJetRef);
    }

    /// Apply vectorial correction
    double correction ( const reco::Jet& fJet, 
			const edm::RefToBase<reco::Jet>& fJetRef,
			LorentzVector& corrected ) const {
      return impl_->correction(fJet,fJetRef,corrected);
    }
    
    /// if correction needs the jet reference
    bool refRequired () const {
      return impl_->refRequired();
    }
  
    /// if vectorial correction is provided
    bool vectorialCorrection () const {
      return impl_->vectorialCorrection();
    }

    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    void swap(JetCorrector& iOther) {
      std::swap(impl_,iOther.impl_);
    }

  private:
    JetCorrector(const JetCorrector&) = delete; 
    
    JetCorrector& operator=(const JetCorrector&) = delete; 
    JetCorrector& operator=(JetCorrector&&) = default;
    
    // ---------- member data --------------------------------
    std::unique_ptr<JetCorrectorImpl const> impl_;
#else
  private:
    JetCorrector(const JetCorrector&);
    const JetCorrector& operator=(const JetCorrector&);
#endif
  };
}

#endif

#ifndef JetMETCorrections_JetCorrector_JetCorrectorImpl_h
#define JetMETCorrections_JetCorrector_JetCorrectorImpl_h
// -*- C++ -*-
//
// Package:     JetMETCorrections/JetCorrector
// Class  :     JetCorrectorImpl
// 
/**\class reco::JetCorrectorImpl JetCorrectorImpl.h "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Fri, 29 Aug 2014 15:42:44 GMT
//

// system include files

// user include files
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/JetReco/interface/Jet.h"

// forward declarations

namespace reco {

  class JetCorrectorImpl
  {
    
  public:
    JetCorrectorImpl();
    virtual ~JetCorrectorImpl();
    
    typedef reco::Particle::LorentzVector LorentzVector;

    // ---------- const member functions ---------------------
    /// get correction using Jet information only
    virtual double correction (const LorentzVector& fJet) const = 0;

    /// apply correction using Jet information only
    virtual double correction (const reco::Jet& fJet) const = 0;

    /// apply correction using Ref
    virtual double correction (const reco::Jet& fJet,
			       const edm::RefToBase<reco::Jet>& fJetRef) const ;

    /// Apply vectorial correction
    virtual double correction ( const reco::Jet& fJet, 
				const edm::RefToBase<reco::Jet>& fJetRef,
				LorentzVector& corrected ) const ;
    
    /// if correction needs the jet reference
    virtual bool refRequired () const = 0;
  
    /// if vectorial correction is provided
    virtual bool vectorialCorrection () const ;
    
    // ---------- static member functions --------------------
    
    // ---------- member functions ---------------------------
    
  private:
    JetCorrectorImpl(const JetCorrectorImpl&) = delete;
    
    const JetCorrectorImpl& operator=(const JetCorrectorImpl&) = delete;
    
    // ---------- member data --------------------------------
    
  };
}

#endif

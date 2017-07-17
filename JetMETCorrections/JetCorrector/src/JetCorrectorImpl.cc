// -*- C++ -*-
//
// Package:     JetMETCorrections/JetCorrector
// Class  :     JetCorrectorImpl
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 29 Aug 2014 17:58:54 GMT
//

// system include files

// user include files
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
reco::JetCorrectorImpl::JetCorrectorImpl()
{
}

// JetCorrectorImpl::JetCorrectorImpl(const JetCorrectorImpl& rhs)
// {
//    // do actual copying here;
// }

reco::JetCorrectorImpl::~JetCorrectorImpl()
{
}

//
// assignment operators
//
// const JetCorrectorImpl& JetCorrectorImpl::operator=(const JetCorrectorImpl& rhs)
// {
//   //An exception safe implementation is
//   JetCorrectorImpl temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

//
// const member functions
//

double 
reco::JetCorrectorImpl::correction (const reco::Jet& fJet,
				    const edm::RefToBase<reco::Jet>& fJetRef) const  {
  return correction(fJet);
}

    /// Apply vectorial correction
double 
reco::JetCorrectorImpl::correction ( const reco::Jet& fJet, 
				     const edm::RefToBase<reco::Jet>& fJetRef,
				     LorentzVector& corrected ) const {
  return correction(fJet);
}

bool
reco::JetCorrectorImpl::vectorialCorrection() const {
  return false;
}

//
// static member functions
//

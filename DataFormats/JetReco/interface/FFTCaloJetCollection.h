// $Id: FFTCaloJetCollection.h,v 1.1 2010/11/22 23:27:55 igv Exp $

#ifndef DataFormats_JetReco_FFTCaloJetCollection_h
#define DataFormats_JetReco_FFTCaloJetCollection_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

namespace reco {
    typedef FFTAnyJet<CaloJet> FFTCaloJet;
    /// collection of FFTCaloJet objects 
    typedef std::vector<FFTCaloJet> FFTCaloJetCollection;
    /// edm references
    typedef edm::Ref<FFTCaloJetCollection> FFTCaloJetRef;
    typedef edm::FwdRef<FFTCaloJetCollection> FFTCaloJetFwdRef;
    typedef edm::FwdPtr<FFTCaloJet> FFTCaloJetFwdPtr;
    typedef edm::RefVector<FFTCaloJetCollection> FFTCaloJetRefVector;
    typedef edm::RefProd<FFTCaloJetCollection> FFTCaloJetRefProd;
    typedef std::vector<edm::FwdRef<FFTCaloJetCollection> > FFTCaloJetFwdRefVector;
    typedef std::vector<edm::FwdPtr<FFTCaloJet> > FFTCaloJetFwdPtrVector;
}

#endif

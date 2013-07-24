// $Id: FFTGenJetCollection.h,v 1.1 2010/11/22 23:27:55 igv Exp $

#ifndef DataFormats_JetReco_FFTGenJetCollection_h
#define DataFormats_JetReco_FFTGenJetCollection_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/GenJet.h"//INCLUDECHECKER:SKIP
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

namespace reco {
    typedef FFTAnyJet<GenJet> FFTGenJet;
    /// collection of FFTGenJet objects 
    typedef std::vector<FFTGenJet> FFTGenJetCollection;
    /// edm references
    typedef edm::Ref<FFTGenJetCollection> FFTGenJetRef;
    typedef edm::FwdRef<FFTGenJetCollection> FFTGenJetFwdRef;
    typedef edm::FwdPtr<FFTGenJet> FFTGenJetFwdPtr;
    typedef edm::RefVector<FFTGenJetCollection> FFTGenJetRefVector;
    typedef edm::RefProd<FFTGenJetCollection> FFTGenJetRefProd;
    typedef std::vector<edm::FwdRef<FFTGenJetCollection> > FFTGenJetFwdRefVector;
    typedef std::vector<edm::FwdPtr<FFTGenJet> > FFTGenJetFwdPtrVector;
}

#endif

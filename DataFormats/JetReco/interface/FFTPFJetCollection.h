// $Id: FFTPFJetCollection.h,v 1.1 2010/11/22 23:27:56 igv Exp $

#ifndef DataFormats_JetReco_FFTPFJetCollection_h
#define DataFormats_JetReco_FFTPFJetCollection_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/PFJet.h" //INCLUDECHECKER:SKIP
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

namespace reco {
    typedef FFTAnyJet<PFJet> FFTPFJet;
    /// collection of FFTPFJet objects 
    typedef std::vector<FFTPFJet> FFTPFJetCollection;
    /// edm references
    typedef edm::Ref<FFTPFJetCollection> FFTPFJetRef;
    typedef edm::FwdRef<FFTPFJetCollection> FFTPFJetFwdRef;
    typedef edm::FwdPtr<FFTPFJet> FFTPFJetFwdPtr;
    typedef edm::RefVector<FFTPFJetCollection> FFTPFJetRefVector;
    typedef edm::RefProd<FFTPFJetCollection> FFTPFJetRefProd;
    typedef std::vector<edm::FwdRef<FFTPFJetCollection> > FFTPFJetFwdRefVector;
    typedef std::vector<edm::FwdPtr<FFTPFJet> > FFTPFJetFwdPtrVector;
}

#endif

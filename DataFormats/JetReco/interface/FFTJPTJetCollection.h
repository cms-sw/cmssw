// $Id: FFTJPTJetCollection.h,v 1.1 2010/11/22 23:27:55 igv Exp $

#ifndef DataFormats_JetReco_FFTJPTJetCollection_h
#define DataFormats_JetReco_FFTJPTJetCollection_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

namespace reco {
    typedef FFTAnyJet<JPTJet> FFTJPTJet;
    /// collection of FFTJPTJet objects 
    typedef std::vector<FFTJPTJet> FFTJPTJetCollection;
    /// edm references
    typedef edm::Ref<FFTJPTJetCollection> FFTJPTJetRef;
    typedef edm::FwdRef<FFTJPTJetCollection> FFTJPTJetFwdRef;
    typedef edm::FwdPtr<FFTJPTJet> FFTJPTJetFwdPtr;
    typedef edm::RefVector<FFTJPTJetCollection> FFTJPTJetRefVector;
    typedef edm::RefProd<FFTJPTJetCollection> FFTJPTJetRefProd;
    typedef std::vector<edm::FwdRef<FFTJPTJetCollection> > FFTJPTJetFwdRefVector;
    typedef std::vector<edm::FwdPtr<FFTJPTJet> > FFTJPTJetFwdPtrVector;
}

#endif

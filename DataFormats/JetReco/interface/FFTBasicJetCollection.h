// $Id: FFTBasicJetCollection.h,v 1.1 2010/11/22 23:27:55 igv Exp $

#ifndef DataFormats_JetReco_FFTBasicJetCollection_h
#define DataFormats_JetReco_FFTBasicJetCollection_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

namespace reco {
    typedef FFTAnyJet<BasicJet> FFTBasicJet;
    /// collection of FFTBasicJet objects 
    typedef std::vector<FFTBasicJet> FFTBasicJetCollection;
    /// edm references
    typedef edm::Ref<FFTBasicJetCollection> FFTBasicJetRef;
    typedef edm::FwdRef<FFTBasicJetCollection> FFTBasicJetFwdRef;
    typedef edm::FwdPtr<FFTBasicJet> FFTBasicJetFwdPtr;
    typedef edm::RefVector<FFTBasicJetCollection> FFTBasicJetRefVector;
    typedef std::vector<edm::FwdRef<FFTBasicJetCollection> > FFTBasicJetFwdRefVector;
    typedef std::vector<edm::FwdPtr<FFTBasicJet> > FFTBasicJetFwdPtrVector;
    typedef edm::RefProd<FFTBasicJetCollection> FFTBasicJetRefProd;
}

#endif

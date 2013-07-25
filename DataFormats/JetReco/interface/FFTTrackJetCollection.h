// $Id: FFTTrackJetCollection.h,v 1.1 2010/11/22 23:27:56 igv Exp $

#ifndef DataFormats_JetReco_FFTTrackJetCollection_h
#define DataFormats_JetReco_FFTTrackJetCollection_h

#include <vector>

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/FwdPtr.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/JetReco/interface/TrackJet.h"
#include "DataFormats/JetReco/interface/FFTAnyJet.h"

namespace reco {
    typedef FFTAnyJet<TrackJet> FFTTrackJet;
    /// collection of FFTTrackJet objects 
    typedef std::vector<FFTTrackJet> FFTTrackJetCollection;
    /// edm references
    typedef edm::Ref<FFTTrackJetCollection> FFTTrackJetRef;
    typedef edm::FwdRef<FFTTrackJetCollection> FFTTrackJetFwdRef;
    typedef edm::FwdPtr<FFTTrackJet> FFTTrackJetFwdPtr;
    typedef edm::RefVector<FFTTrackJetCollection> FFTTrackJetRefVector;
    typedef std::vector<edm::FwdRef<FFTTrackJetCollection> > FFTTrackJetFwdRefVector;
    typedef std::vector<edm::FwdPtr<FFTTrackJet> > FFTTrackJetFwdPtrVector;
    typedef edm::RefProd<FFTTrackJetCollection> FFTTrackJetRefProd;
}

#endif

#ifndef JetObjects3_classes_h
#define JetObjects3_classes_h

#include "Rtypes.h"

#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#if defined __has_feature
#if __has_feature(modules)
// Workaround the missing CLHEP.modulemap: CLHEP/Vector/ThreeVector.h:41:7: error: redefinition of 'Hep3Vector'
#include "DataFormats/JetReco/interface/PFClusterJetCollection.h"
#endif
#endif
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/TrackJetCollection.h"
#include "DataFormats/JetReco/interface/PFClusterJetCollection.h"
#include "DataFormats/JetReco/interface/GenericJetCollection.h"
#include "DataFormats/JetReco/interface/JetTrackMatch.h"
#include "DataFormats/JetReco/interface/JetFloatAssociation.h"
#include "DataFormats/JetReco/interface/JetTracksAssociation.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

#include "DataFormats/JetReco/interface/FFTJetProducerSummary.h"
#include "DataFormats/JetReco/interface/FFTCaloJetCollection.h"
#include "DataFormats/JetReco/interface/FFTJPTJetCollection.h"
#include "DataFormats/JetReco/interface/FFTGenJetCollection.h"
#include "DataFormats/JetReco/interface/FFTPFJetCollection.h"
#include "DataFormats/JetReco/interface/FFTTrackJetCollection.h"
#include "DataFormats/JetReco/interface/FFTBasicJetCollection.h"
#include "DataFormats/JetReco/interface/FFTJetPileupSummary.h"
#include "DataFormats/JetReco/interface/DiscretizedEnergyFlow.h"
#include "DataFormats/JetReco/interface/PileupJetIdentifier.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/RefHolder.h"
#include "DataFormats/Common/interface/Holder.h"
#include "DataFormats/Common/interface/Association.h"

#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/Ptr.h"

#endif

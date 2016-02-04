#ifndef HLTElectronPFMTFilter_h
#define HLTElectronPFMTFilter_h

/** \class HLTElectronPFMTFilter
 *
 *  \author Jeffrey Berryhill
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"



#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <vector>
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TLorentzVector.h"
#include "TVector3.h"

namespace edm {
   class ConfigurationDescriptions;
}


//
// class declaration
//

class HLTElectronPFMTFilter : public HLTFilter {

   public:
      explicit HLTElectronPFMTFilter(const edm::ParameterSet&);
      ~HLTElectronPFMTFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputMetTag_; // input tag identifying jets
      bool saveTags_;              // whether to save this tag
      double minMht_;

      edm::InputTag inputEleTag_;     // input tag identifying egammas
      double lowerMTCut_;
      double upperMTCut_;
      bool   relaxed_;
      int    minN_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTElectronPFMTFilter_h

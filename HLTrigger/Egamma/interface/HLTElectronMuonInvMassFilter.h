#ifndef HLTElectronMuonInvMassFilter_h
#define HLTElectronMuonInvMassFilter_h

/** \class HLTElectronMuonInvMassFilter
 *
 *  Original Author: Massimiliano Chiorboli
 *  Institution: INFN, Italy
 *  Contact: Massimiliano.Chiorboli@cern.ch
 *  Date: July 6, 2011
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "TLorentzVector.h"
#include "TVector3.h"

namespace edm {
  class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTElectronMuonInvMassFilter : public HLTFilter {

   public:
      explicit HLTElectronMuonInvMassFilter(const edm::ParameterSet&);
      ~HLTElectronMuonInvMassFilter();
      virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct) const override;
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

   private:
      edm::InputTag eleCandTag_;
      edm::InputTag muonCandTag_;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> eleCandToken_;
      edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> muonCandToken_;

      double lowerMassCut_;
      double upperMassCut_;
      int ncandcut_;
      bool relaxed_;
      edm::InputTag L1IsoCollTag_;
      edm::InputTag L1NonIsoCollTag_;
      edm::InputTag MuonCollTag_;
};

#endif //HLTElectronMuonInvMassFilter_h

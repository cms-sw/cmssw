#ifndef HLTRFilter_h
#define HLTRFilter_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLTRFilter : public edm::EDFilter {

   public:

      explicit HLTRFilter(const edm::ParameterSet&);
      ~HLTRFilter();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

      static double CalcMR(TLorentzVector ja,TLorentzVector jb);
      static double CalcR(double MR, TLorentzVector ja,TLorentzVector jb, edm::Handle<reco::CaloMETCollection> met, const std::vector<math::XYZTLorentzVector>& muons);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      edm::InputTag inputMetTag_; // input tag identifying MET product
      bool doMuonCorrection_;  // do the muon corrections
      double min_R_;           // minimum R vaule
      double min_MR_;          // minimum MR vaule
      bool DoRPrime_;          // Do the R' instead of R
      bool accept_NJ_;         // accept or reject events with high NJ
      double R_offset_;        // R offset for parameterized cut
      double MR_offset_;       // MR offset for parameterized cut
      double R_MR_cut_;        // Cut value for parameterized cut

};

#endif //HLTRFilter_h

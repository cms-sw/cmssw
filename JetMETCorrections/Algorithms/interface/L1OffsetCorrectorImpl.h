// L1Offset jet corrector class. Inherits from JetCorrector.h
#ifndef L1OffsetCorrectorImpl_h
#define L1OffsetCorrectorImpl_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrectorImpl.h"
#include "JetMETCorrections/Algorithms/interface/JetCorrectorImplMakerBase.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//----- classes declaration -----------------------------------
namespace edm 
{
  class ParameterSet;
  class Event;
  class EventSetup;
  class ConsumesCollector;
  class ConfigurationDescriptions;
}

class L1OffsetCorrectorImplMaker : public JetCorrectorImplMakerBase {
 public:
  L1OffsetCorrectorImplMaker(edm::ParameterSet const&, edm::ConsumesCollector);
  std::unique_ptr<reco::JetCorrectorImpl> make(edm::Event const&, edm::EventSetup const&);

  static void fillDescriptions(edm::ConfigurationDescriptions& iDescriptions);
 private:
  edm::EDGetTokenT<reco::VertexCollection> verticesToken_;
  int minVtxNdof_;
};

//----- LXXXCorrector interface -------------------------------
class L1OffsetCorrectorImpl : public reco::JetCorrectorImpl 
{
  public:
  typedef L1OffsetCorrectorImplMaker Maker;

    //----- constructors---------------------------------------
  L1OffsetCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> calculator,
			int npv);

    //----- destructor ----------------------------------------

    //----- apply correction using Jet information only -------
    virtual double correction(const LorentzVector& fJet) const override;

    //----- apply correction using Jet information only -------
    virtual double correction(const reco::Jet& fJet) const override;

    //----- if correction needs a jet reference -------------
    virtual bool refRequired() const override { return false; } 

  private:
    //----- member data ---------------------------------------
    std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector_;
    int npv_;

};

#endif

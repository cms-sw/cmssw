// Implementation of class L1OffsetCorrectorImpl.
// L1Offset jet corrector class.

#include "JetMETCorrections/Algorithms/interface/L1OffsetCorrectorImpl.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

using namespace std;

L1OffsetCorrectorImplMaker::L1OffsetCorrectorImplMaker(edm::ParameterSet const& fConfig, 
						       edm::ConsumesCollector fCollector) :
  JetCorrectorImplMakerBase(fConfig),
  verticesToken_(fCollector.consumes<reco::VertexCollection>(fConfig.getParameter<edm::InputTag>("vertexCollection"))),
  minVtxNdof_(fConfig.getParameter<int>("minVtxNdof"))
{
}
std::unique_ptr<reco::JetCorrectorImpl> 
L1OffsetCorrectorImplMaker::make(edm::Event const& fEvent, edm::EventSetup const& fSetup) 
{
  edm::Handle<reco::VertexCollection> recVtxs;
  fEvent.getByToken(verticesToken_,recVtxs);
  int NPV(0);
  for(auto const& vertex : *recVtxs) {
    if ((not vertex.isFake()) and (vertex.ndof() > minVtxNdof_)) {
      NPV++;
    }
  }
  
  auto calculator = getCalculator(fSetup,
				  [](std::string const& level) 
    {
      if ( level != "L1Offset") {
	throw cms::Exception("L1OffsetCorrectorImpl")<<" correction level: "<<level<<" is not L1Offset";
      }
    });
  return std::unique_ptr<reco::JetCorrectorImpl>(new L1OffsetCorrectorImpl(calculator,NPV));
}

void 
L1OffsetCorrectorImplMaker::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions)
{
  edm::ParameterSetDescription desc;
  addToDescription(desc);
  desc.add<edm::InputTag>("vertexCollection");
  desc.add<int>("minVtxNdof");
  iDescriptions.addDefault(desc);
}

//------------------------------------------------------------------------ 
//--- L1OffsetCorrectorImpl constructor ------------------------------------------
//------------------------------------------------------------------------
L1OffsetCorrectorImpl::L1OffsetCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> calculator,
					     int npv):
  corrector_(calculator),
  npv_(npv)
{
}

//------------------------------------------------------------------------ 
//--- Returns correction for a given 4-vector ----------------------------
//------------------------------------------------------------------------
double L1OffsetCorrectorImpl::correction(const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet ---------------------------------
//------------------------------------------------------------------------
double L1OffsetCorrectorImpl::correction(const reco::Jet& fJet) const
{
  double result = 1.;
  if (npv_ > 0) {
    FactorizedJetCorrectorCalculator::VariableValues values;
    values.setJetEta(fJet.eta());
    values.setJetPt(fJet.pt());
    values.setJetE(fJet.energy());
    values.setNPV(npv_);
    result = corrector_->getCorrection(values);
  }
  return result;
}


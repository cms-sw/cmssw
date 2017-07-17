// Implementation of class L1JPTOffsetCorrector.
// L1JPTOffset jet corrector class.

#include "JetMETCorrections/Algorithms/interface/L1JPTOffsetCorrectorImpl.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

using namespace std;

L1JPTOffsetCorrectorImplMaker::L1JPTOffsetCorrectorImplMaker(edm::ParameterSet const& fConfig, edm::ConsumesCollector fCollector):
  JetCorrectorImplMakerBase(fConfig),
  useOffset_(false)
{
  auto const& offsetService = fConfig.getParameter<edm::InputTag>("offsetService");
  if(not offsetService.label().empty()) {
    useOffset_ =true;
    offsetCorrectorToken_ = fCollector.consumes<reco::JetCorrector>(offsetService);
  }
}

std::unique_ptr<reco::JetCorrectorImpl> 
L1JPTOffsetCorrectorImplMaker::make(edm::Event const& fEvent, edm::EventSetup const& fSetup) {
  reco::JetCorrector const* offset = nullptr;
  if(useOffset_) {
    edm::Handle<reco::JetCorrector> hOffset;
    fEvent.getByToken(offsetCorrectorToken_,hOffset);
    offset = &(*hOffset);
  }
  auto calculator = getCalculator(fSetup,
				  [](std::string const& level) 
    {
      if ( level != "L1JPTOffset") {
	throw cms::Exception("L1OffsetCorrector")<<" correction level: "<<level<<" is not L1JPTOffset";
      }
    });
  return std::unique_ptr<reco::JetCorrectorImpl>( new L1JPTOffsetCorrectorImpl(calculator,offset) );
}

void 
L1JPTOffsetCorrectorImplMaker::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions)
{
  edm::ParameterSetDescription desc;
  addToDescription(desc);
  desc.add<edm::InputTag>("offsetService");
  iDescriptions.addDefault(desc);
}


//------------------------------------------------------------------------ 
//--- L1OffsetCorrectorImpl constructor ------------------------------------------
//------------------------------------------------------------------------
L1JPTOffsetCorrectorImpl::L1JPTOffsetCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> corrector,
					   const reco::JetCorrector* offsetService):
  offsetService_(offsetService),
  corrector_(corrector)
{
}
//------------------------------------------------------------------------ 
//--- L1OffsetCorrectorImpl destructor -------------------------------------------
//------------------------------------------------------------------------

//------------------------------------------------------------------------ 
//--- Returns correction for a given 4-vector ----------------------------
//------------------------------------------------------------------------
double L1JPTOffsetCorrectorImpl::correction(const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet ---------------------------------
//------------------------------------------------------------------------
double L1JPTOffsetCorrectorImpl::correction(const reco::Jet& fJet) const
{
  double result = 1.;
  const reco::JPTJet& jptjet = dynamic_cast <const reco::JPTJet&> (fJet);
  edm::RefToBase<reco::Jet> jptjetRef = jptjet.getCaloJetRef();
  reco::CaloJet const * rawcalojet = dynamic_cast<reco::CaloJet const *>( &* jptjetRef);   
  //------ access the offset correction service ----------------
  double offset = 1.0;
  if (offsetService_) {
    offset = offsetService_->correction(*rawcalojet); 
  }
  //------ calculate the correction for the JPT jet ------------
  TLorentzVector JPTrawP4(rawcalojet->px(),rawcalojet->py(),rawcalojet->pz(),rawcalojet->energy());
  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJPTrawP4(JPTrawP4);
  values.setJPTrawOff(offset);
  values.setJetE(fJet.energy());
  result = corrector_->getCorrection(values);
  return result;
}


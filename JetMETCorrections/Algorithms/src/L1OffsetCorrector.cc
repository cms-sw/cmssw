// Implementation of class L1OffsetCorrector.
// L1Offset jet corrector class.

#include "JetMETCorrections/Algorithms/interface/L1OffsetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

using namespace std;


//------------------------------------------------------------------------ 
//--- L1OffsetCorrector constructor ------------------------------------------
//------------------------------------------------------------------------
L1OffsetCorrector::L1OffsetCorrector(const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig) 
{
  mVertexCollName = fConfig.getParameter<std::string>("vertexCollection");
  mMinVtxNdof     = fConfig.getParameter<int>("minVtxNdof");
  if (fParam.definitions().level() != "L1Offset")
    throw cms::Exception("L1OffsetCorrector")<<" correction level: "<<fParam.definitions().level()<<" is not L1Offset"; 
  vector<JetCorrectorParameters> vParam;
  vParam.push_back(fParam);
  mCorrector = new FactorizedJetCorrector(vParam);
}
//------------------------------------------------------------------------ 
//--- L1OffsetCorrector destructor -------------------------------------------
//------------------------------------------------------------------------
L1OffsetCorrector::~L1OffsetCorrector() 
{
  delete mCorrector;
} 
//------------------------------------------------------------------------ 
//--- Returns correction for a given 4-vector ----------------------------
//------------------------------------------------------------------------
double L1OffsetCorrector::correction(const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet ---------------------------------
//------------------------------------------------------------------------
double L1OffsetCorrector::correction(const reco::Jet& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(reco::Jet), event required!";
  return 1.0;
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet using event indormation ---------
//------------------------------------------------------------------------
double L1OffsetCorrector::correction(const reco::Jet& fJet, 
                                     const edm::Event& fEvent, 
                                     const edm::EventSetup& fSetup) const 
{
  double result = 1.;
  edm::Handle<reco::VertexCollection> recVtxs;
  fEvent.getByLabel(mVertexCollName,recVtxs);
  int NPV(0);
  for(unsigned int ind=0;ind<recVtxs->size();ind++) {
    if (!((*recVtxs)[ind].isFake()) && (*recVtxs)[ind].ndof() > mMinVtxNdof) {
      NPV++;
    }
  } 
  if (NPV > 0) {
    mCorrector->setJetEta(fJet.eta());
    mCorrector->setJetPt(fJet.pt());
    mCorrector->setNPV(NPV);
    result = mCorrector->getCorrection();
  }
  return result;
}


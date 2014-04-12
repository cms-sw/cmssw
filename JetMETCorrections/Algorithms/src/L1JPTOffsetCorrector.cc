// Implementation of class L1JPTOffsetCorrector.
// L1JPTOffset jet corrector class.

#include "JetMETCorrections/Algorithms/interface/L1JPTOffsetCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

using namespace std;


//------------------------------------------------------------------------ 
//--- L1OffsetCorrector constructor ------------------------------------------
//------------------------------------------------------------------------
L1JPTOffsetCorrector::L1JPTOffsetCorrector(const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig) 
{
  mOffsetService = fConfig.getParameter<std::string>("offsetService");
  mIsOffsetSet = false;
  if (mOffsetService != "")
    mIsOffsetSet = true;
  if (fParam.definitions().level() != "L1JPTOffset")
    throw cms::Exception("L1OffsetCorrector")<<" correction level: "<<fParam.definitions().level()<<" is not L1JPTOffset"; 
  vector<JetCorrectorParameters> vParam;
  vParam.push_back(fParam);
  mCorrector = new FactorizedJetCorrectorCalculator(vParam);
}
//------------------------------------------------------------------------ 
//--- L1OffsetCorrector destructor -------------------------------------------
//------------------------------------------------------------------------
L1JPTOffsetCorrector::~L1JPTOffsetCorrector() 
{
  delete mCorrector;
} 
//------------------------------------------------------------------------ 
//--- Returns correction for a given 4-vector ----------------------------
//------------------------------------------------------------------------
double L1JPTOffsetCorrector::correction(const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet ---------------------------------
//------------------------------------------------------------------------
double L1JPTOffsetCorrector::correction(const reco::Jet& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(reco::Jet), event required!";
  return 1.0;
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet using event indormation ---------
//------------------------------------------------------------------------
double L1JPTOffsetCorrector::correction(const reco::Jet& fJet, 
                                     const edm::Event& fEvent, 
                                     const edm::EventSetup& fSetup) const 
{
  double result = 1.;
  const reco::JPTJet& jptjet = dynamic_cast <const reco::JPTJet&> (fJet);
  edm::RefToBase<reco::Jet> jptjetRef = jptjet.getCaloJetRef();
  reco::CaloJet const * rawcalojet = dynamic_cast<reco::CaloJet const *>( &* jptjetRef);   
  //------ access the offset correction service ----------------
  double offset = 1.0;
  if (mIsOffsetSet) {
    const JetCorrector* OffsetCorrector = JetCorrector::getJetCorrector(mOffsetService,fSetup); 
    offset = OffsetCorrector->correction(*rawcalojet,fEvent,fSetup); 
  }
  //------ calculate the correction for the JPT jet ------------
  TLorentzVector JPTrawP4(rawcalojet->px(),rawcalojet->py(),rawcalojet->pz(),rawcalojet->energy());
  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJPTrawP4(JPTrawP4);
  values.setJPTrawOff(offset);
  values.setJetE(fJet.energy());
  result = mCorrector->getCorrection(values);
  return result;
}


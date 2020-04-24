////////////////////////////////////////////////////////////////////////////////
//
// L1FastjetCorrector
// ------------------
//
//            08/09/2009 Philipp Schieferdecker <philipp.schieferdecker@cern.ch>
////////////////////////////////////////////////////////////////////////////////

#include "JetMETCorrections/Algorithms/interface/L1FastjetCorrector.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace std;


////////////////////////////////////////////////////////////////////////////////
// construction / destruction
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
L1FastjetCorrector::L1FastjetCorrector (const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig)
  : srcRho_(fConfig.getParameter<edm::InputTag>("srcRho"))
{
  if (fParam.definitions().level() != "L1FastJet")
    throw cms::Exception("L1FastjetCorrector")<<" correction level: "<<fParam.definitions().level()<<" is not L1FastJet"; 
  vector<JetCorrectorParameters> vParam;
  vParam.push_back(fParam);
  mCorrector = new FactorizedJetCorrectorCalculator(vParam);
}

//______________________________________________________________________________
L1FastjetCorrector::~L1FastjetCorrector ()
{
  delete mCorrector;
} 


////////////////////////////////////////////////////////////////////////////////
// implementation of member functions
////////////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
double L1FastjetCorrector::correction (const LorentzVector& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(LorentzVector), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L1FastjetCorrector::correction (const reco::Jet& fJet) const
{
  throw cms::Exception("EventRequired")
    <<"Wrong interface correction(reco::Jet), event required!";
  return 1.0;
}


//______________________________________________________________________________
double L1FastjetCorrector::correction(const reco::Jet& fJet,
				      const edm::Event& fEvent,
				      const edm::EventSetup& fSetup) const
{
  edm::Handle<double> rho;
  fEvent.getByLabel(srcRho_,rho);
  double result(1.0);
  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJetEta(fJet.eta());
  values.setJetPt(fJet.pt());
  values.setJetE(fJet.energy());
  values.setJetA(fJet.jetArea());
  values.setRho(*rho);
  result = mCorrector->getCorrection(values);  
  return result;
}





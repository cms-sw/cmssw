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
  : srcMedianPt_(fConfig.getParameter<edm::InputTag>("srcMedianPt"))
{
  
}

//______________________________________________________________________________
L1FastjetCorrector::~L1FastjetCorrector ()
{
  
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
				      const edm::RefToBase<reco::Jet>& fJetRef,
				      const edm::Event& fEvent,
				      const edm::EventSetup& fSetup) const
{
  edm::Handle<double> medianPt;
  fEvent.getByLabel(srcMedianPt_,medianPt);
  double result = (fJet.pt()-(*medianPt)*fJet.jetArea())/fJet.pt();
  return (result>0) ? result : 0.0;
}

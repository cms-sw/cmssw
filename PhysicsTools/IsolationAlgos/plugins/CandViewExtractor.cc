#include "CandViewExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositDirection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace reco;
using namespace muonisolation;

#include "CandViewExtractor.icc"

CandViewExtractor::CandViewExtractor( const ParameterSet& par ) :
  theCandViewTag(par.getParameter<edm::InputTag>("inputCandView")),
  theDepositLabel(par.getUntrackedParameter<std::string>("DepositLabel")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto"))
{
}
/*
reco::IsoDeposit::Vetos CandViewExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const
{
  reco::isodeposit::Direction dir(cand.eta(),cand.phi());
  return reco::IsoDeposit::Vetos(1,veto(dir));
}
*/

reco::IsoDeposit::Veto CandViewExtractor::veto(const reco::IsoDeposit::Direction & dir) const
{
  reco::IsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}
 
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "CandViewExtractor.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::CandViewExtractor, "CandViewExtractor");

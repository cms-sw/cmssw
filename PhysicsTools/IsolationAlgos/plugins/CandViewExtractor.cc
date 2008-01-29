#include "CandViewExtractor.h"

#include "RecoMuon/MuonIsolation/interface/Range.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace muonisolation;

#include "CandViewExtractor.icc"

CandViewExtractor::CandViewExtractor( const ParameterSet& par ) :
  theCandViewTag(par.getParameter<edm::InputTag>("inputCandView")),
  theDepositLabel(par.getUntrackedParameter<string>("DepositLabel")),
  theDiff_r(par.getParameter<double>("Diff_r")),
  theDiff_z(par.getParameter<double>("Diff_z")),
  theDR_Max(par.getParameter<double>("DR_Max")),
  theDR_Veto(par.getParameter<double>("DR_Veto"))
{
}
/*
reco::MuIsoDeposit::Vetos CandViewExtractor::vetos(const edm::Event & ev,
      const edm::EventSetup & evSetup, const reco::Candidate & cand) const
{
  Direction dir(cand.eta(),cand.phi());
  return reco::MuIsoDeposit::Vetos(1,veto(dir));
}
*/

reco::MuIsoDeposit::Veto CandViewExtractor::veto(const reco::MuIsoDeposit::Direction & dir) const
{
  reco::MuIsoDeposit::Veto result;
  result.vetoDir = dir;
  result.dR = theDR_Veto;
  return result;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"
#include "CandViewExtractor.h"
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::CandViewExtractor, "CandViewExtractor");

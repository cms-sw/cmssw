#include "PhysicsTools/SelectorUtils/interface/CutApplicatorBase.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include <boost/algorithm/string.hpp>

class MuonTypeByOrCut : public CutApplicatorBase
{
public:
  MuonTypeByOrCut(const edm::ParameterSet& c);

  result_type operator()(const reco::MuonPtr&) const override final;
  CandidateType candidateType() const override final { return MUON; }
  double value(const reco::CandidatePtr&) const override final;

private:
  unsigned int type_;
};
DEFINE_EDM_PLUGIN(CutApplicatorFactory, MuonTypeByOrCut, "MuonTypeByOrCut");

// Define constructors and initialization routines
MuonTypeByOrCut::MuonTypeByOrCut(const edm::ParameterSet& c):
  CutApplicatorBase(c),
  type_(0)
{
  const auto muonTypes = c.getParameter<std::vector<std::string> >("types");
  for ( const auto& x : muonTypes )
  {
    if      ( boost::iequals(x, "GlobalMuon")     ) type_ |= reco::Muon::GlobalMuon;
    else if ( boost::iequals(x, "TrackerMuon")    ) type_ |= reco::Muon::TrackerMuon;
    else if ( boost::iequals(x, "StandAloneMuon") ) type_ |= reco::Muon::StandAloneMuon;
    else if ( boost::iequals(x, "CaloMmuon")      ) type_ |= reco::Muon::CaloMuon;
    else if ( boost::iequals(x, "PFMuon")         ) type_ |= reco::Muon::PFMuon;
    else if ( boost::iequals(x, "RPCMuon")        ) type_ |= reco::Muon::RPCMuon;
  }
}

// Functors for evaluation
CutApplicatorBase::result_type MuonTypeByOrCut::operator()(const reco::MuonPtr& muon) const
{
  return (muon->type() & type_) != 0;
}

double MuonTypeByOrCut::value(const reco::CandidatePtr& cand) const
{
  const reco::MuonPtr muon(cand);
  return muon->type() & type_;
}

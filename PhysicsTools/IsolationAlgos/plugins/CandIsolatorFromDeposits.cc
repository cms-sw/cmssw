#include "PhysicsTools/IsolationAlgos/plugins/CandIsolatorFromDeposits.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/Direction.h"
#include "DataFormats/MuonReco/interface/MuIsoDeposit.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositFwd.h"
#include "DataFormats/MuonReco/interface/MuIsoDepositVetos.h"

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>
#include <boost/regex.hpp>

using namespace edm;
using namespace std;
using namespace reco;
using namespace reco::isodeposit;

bool isNumber(const std::string &str) {
   static boost::regex re("^[+-]?(\\d+\\.?|\\d*\\.\\d*)$");
   return regex_match(str.c_str(), re);
}
double toNumber(const std::string &str) {
    return atof(str.c_str());
}

CandIsolatorFromDeposits::SingleDeposit::SingleDeposit(const edm::ParameterSet &iConfig) :
    src_(iConfig.getParameter<edm::InputTag>("src")),
    deltaR_(iConfig.getParameter<double>("deltaR")),
    weightExpr_(iConfig.getParameter<std::string>("weight")),
    skipDefaultVeto_(iConfig.getParameter<bool>("skipDefaultVeto"))
	//,vetos_(new AbsVetos())
{
    std::string mode = iConfig.getParameter<std::string>("mode");
    if (mode == "sum") mode_ = Sum; 
    else if (mode == "sumRelative") mode_ = SumRelative; 
    //else if (mode == "max") mode_ = Max;                  // TODO: on request only
    //else if (mode == "maxRelative") mode_ = MaxRelative;  // TODO: on request only
    else if (mode == "count") mode_ = Count;
    else throw cms::Exception("Not Implemented") << "Mode '" << mode << "' not implemented. " <<
            "Supported modes are 'sum', 'sumRelative', 'count'." << 
            //"Supported modes are 'sum', 'sumRelative', 'max', 'maxRelative', 'count'." << // TODO: on request only
            "New methods can be easily implemented if requested.";
    typedef std::vector<std::string> vstring;
    vstring vetos = iConfig.getParameter< vstring >("vetos");
    for (vstring::const_iterator it = vetos.begin(), ed = vetos.end(); it != ed; ++it) {
        if (!isNumber(*it)) {
			static boost::regex threshold("Threshold\\((\\d+\\.\\d+)\\)"), 
							    cone("ConeVeto\\((\\d+\\.\\d+)\\)"),
							    angleCone("AngleCone\\((\\d+\\.\\d+)\\)"),
							    angleVeto("AngleVeto\\((\\d+\\.\\d+)\\)");
			boost::cmatch match;
			if (regex_match(it->c_str(), match, threshold)) {
				vetos_.push_back(new ThresholdVeto(toNumber(match[1].first)));
			} else if (regex_match(it->c_str(), match, cone)) {
				vetos_.push_back(new ConeVeto(Direction(), toNumber(match[1].first)));
			} else if (regex_match(it->c_str(), match, angleCone)) {
				vetos_.push_back(new AngleCone(Direction(), toNumber(match[1].first)));
			} else if (regex_match(it->c_str(), match, angleVeto)) {
				vetos_.push_back(new AngleConeVeto(Direction(), toNumber(match[1].first)));
			} else {
				throw cms::Exception("Not Implemented") << "Veto " << it->c_str() << " not implemented yet...";
			}
        }  else {
            //std::cout << "Adding veto of radius " << toNumber(*it) << std::endl;
            vetos_.push_back(new ConeVeto(Direction(), toNumber(*it)));
        }
    }
    std::string weight = iConfig.getParameter<std::string>("weight");
    if (isNumber(weight)) {
        //std::cout << "Weight is a simple number, " << toNumber(weight) << std::endl;
        weight_ = toNumber(weight);
        usesFunction_ = false;
    } else {
        usesFunction_ = true;
        //std::cout << "Weight is a function, this might slow you down... " << std::endl;
    }
	std::cout << "Total of " << vetos_.size() << " vetos" << std::endl;
}
void CandIsolatorFromDeposits::SingleDeposit::cleanup() {
    for (AbsVetos::iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        delete *it;
    }
}
void CandIsolatorFromDeposits::SingleDeposit::open(const edm::Event &iEvent) {
    iEvent.getByLabel(src_, hDeps_);
}

double CandIsolatorFromDeposits::SingleDeposit::compute(const reco::CandidateBaseRef &cand) {
    const MuIsoDeposit &dep = (*hDeps_)[cand];
    double eta = dep.eta(), phi = dep.phi(); // better to center on the deposit direction
                                             // that could be, e.g., the impact point at calo
    for (AbsVetos::iterator it = vetos_.begin(), ed = vetos_.end(); it != ed; ++it) {
        (*it)->centerOn(eta, phi);
    }
    double weight = (usesFunction_ ? weightExpr_(*cand) : weight_);
    switch (mode_) {
        case Sum: return weight * dep.depositWithin(deltaR_, vetos_, skipDefaultVeto_);
        case SumRelative: return weight * dep.depositWithin(deltaR_, vetos_, skipDefaultVeto_) / dep.muonEnergy() ;
        case Count: return weight * dep.depositAndCountWithin(deltaR_, vetos_, skipDefaultVeto_).second ;
    }
    throw cms::Exception("Logic error") << "Should not happen at " << __FILE__ << ", line " << __LINE__; // avoid gcc warning
}


/// constructor with config
CandIsolatorFromDeposits::CandIsolatorFromDeposits(const ParameterSet& par) {
  typedef std::vector<edm::ParameterSet> VPSet;
  VPSet depPSets = par.getParameter<VPSet>("deposits");
  for (VPSet::const_iterator it = depPSets.begin(), ed = depPSets.end(); it != ed; ++it) {
    sources_.push_back(SingleDeposit(*it));
  }
  if (sources_.size() == 0) throw cms::Exception("Configuration Error") << "Please specify at least one deposit!";
  produces<CandViewDoubleAssociations>();
}

/// destructor
CandIsolatorFromDeposits::~CandIsolatorFromDeposits() {
  vector<SingleDeposit>::iterator it, begin = sources_.begin(), end = sources_.end();
  for (it = begin; it != end; ++it) it->cleanup();
}

/// build deposits
void CandIsolatorFromDeposits::produce(Event& event, const EventSetup& eventSetup){

  vector<SingleDeposit>::iterator it, begin = sources_.begin(), end = sources_.end();
  for (it = begin; it != end; ++it) it->open(event);

  const CandIsoDepositAssociationVector & vector = begin->vector();

  if (vector.keyProduct().isNull()) { // !!???
        event.put(auto_ptr<CandViewDoubleAssociations>(new CandViewDoubleAssociations()));
        return;
  }
  auto_ptr<CandViewDoubleAssociations> ret(new CandViewDoubleAssociations(vector.keyProduct()));

  for (CandIsoDepositAssociationVector::const_iterator itAV = vector.begin(), edAV = vector.end(); itAV != edAV; ++itAV) {
      double sum = 0;
      const CandidateBaseRef & cand = itAV->first;
      for (it = begin; it != end; ++it) sum += it->compute(cand);
      //ret->operator[](cand) = static_cast<float>(sum);
      ret->operator[](cand) = (sum);
  }

  event.put(ret);
}

DEFINE_FWK_MODULE( CandIsolatorFromDeposits );

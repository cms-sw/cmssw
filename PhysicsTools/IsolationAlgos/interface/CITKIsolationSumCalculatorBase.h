#ifndef IsolationAlgos_CITKIsolationSumCalculatorBase_H
#define IsolationAlgos_CITKIsolationSumCalculatorBase_H

//
//
//


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <unordered_map>

namespace citk {
  class IsolationSumCalculatorBase {
  public:
  IsolationSumCalculatorBase(const edm::ParameterSet& c,
			     edm::ConsumesCollector& cc) :
    _coneSize(c.getParameter<double>("coneSize")),
    _name(c.getParameter<std::string>("isolationAlgo")) {
    }
    
    virtual void getEventSetupInfo(const edm::EventSetup&) {}
    virtual void getEventInfo(const edm::Event&) {}
    virtual void setConsumes(edm::ConsumesCollector&) = 0;

    virtual bool isInIsolationCone(const reco::CandidateBaseRef& physob,
				   const reco::CandidateBaseRef& other) const = 0;

    const std::string& name() const { return name; }

    //! Destructor
    virtual ~IsolationSumCalculatorBase(){};

  protected:
    const double _coneSize;
    
  private:    
    IsolationSumCalculatorBase(const IsolationSumCalculatorBase&) {}
    IsolationSumCalculatorBase& operator=(const IsolationSumCalculatorBase) {}
    const std::string _name;
  };
}// ns citk

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< citk::IsolationSumCalculatorBase* (const edm::ParameterSet&,edm::ConsumesCollector&) > CITKIsolationSumCalculatorFactory;

#endif

// -*- C++ -*-
//
// Package:    LogErrorFilter
// Class:      LogErrorFilter
//
/**\class LogErrorFilter LogErrorFilter.cc FWCore/LogErrorFilter/src/LogErrorFilter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Thu Nov 12 15:59:28 CET 2009
//
//

// user include files
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// system include files
#include <memory>

//
// class declaration
//

class LogErrorFilter : public edm::stream::EDFilter<> {
public:
  explicit LogErrorFilter(edm::ParameterSet const&);
  ~LogErrorFilter() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool filter(edm::Event&, edm::EventSetup const&) override;

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override ;

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<edm::ErrorSummaryEntry>> harvesterToken_;
  bool atLeastOneError_;
  bool atLeastOneWarning_;
  bool atLeastOneEntry_;
  bool useThresholdsPerKind_;

  unsigned int maxErrorKindsPerLumi_;
  unsigned int maxWarningKindsPerLumi_;

  std::vector<std::string> avoidCategories_;

  std::map<std::string, unsigned int> errorCounts_;
  std::map<std::string, unsigned int> warningCounts_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LogErrorFilter::LogErrorFilter(edm::ParameterSet const& iConfig) :
  atLeastOneError_(iConfig.getParameter<bool>("atLeastOneError")),
  atLeastOneWarning_(iConfig.getParameter<bool>("atLeastOneWarning")),
  atLeastOneEntry_(atLeastOneError_ && atLeastOneWarning_),
  useThresholdsPerKind_(iConfig.getParameter<bool>("useThresholdsPerKind")),
  avoidCategories_(iConfig.getParameter<std::vector<std::string> >("avoidCategories")) {
  if(!atLeastOneError_ && !atLeastOneWarning_) {
    throw edm::Exception(edm::errors::Configuration) <<
      "Useless configuration of the error/warning filter. Need to select on an error or a warning or both.\n";
  }
  harvesterToken_ = consumes<std::vector<edm::ErrorSummaryEntry>>(iConfig.getParameter<edm::InputTag>("harvesterTag"));
  maxErrorKindsPerLumi_ = 999999;
  maxWarningKindsPerLumi_ = 999999;
  if (useThresholdsPerKind_){
    maxErrorKindsPerLumi_ = iConfig.getParameter<unsigned int>("maxErrorKindsPerLumi");
    maxWarningKindsPerLumi_ = iConfig.getParameter<unsigned int>("maxWarningKindsPerLumi");
  }
}

LogErrorFilter::~LogErrorFilter() {
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called on each new Event  ------------
bool
LogErrorFilter::filter(edm::Event& iEvent, edm::EventSetup const&) {
  edm::Handle<std::vector<edm::ErrorSummaryEntry> > errorsAndWarnings;
  iEvent.getByToken(harvesterToken_,errorsAndWarnings);

  if(errorsAndWarnings.failedToGet()) {
    return false;
  } else {
    if (useThresholdsPerKind_){
      unsigned int errorsBelowThreshold = 0;
      unsigned int warningsBelowThreshold = 0;
      // update counters here
      for(auto const& summary : *errorsAndWarnings) {
        if (std::find(avoidCategories_.begin(),avoidCategories_.end(), summary.category) != avoidCategories_.end() )
          continue;
        std::string kind= summary.category + ":" + summary.module;
        int iSeverity = summary.severity.getLevel();
        if (iSeverity == edm::ELseverityLevel::ELsev_error){
          unsigned int& iCount = errorCounts_[kind];
          iCount++;
          if (iCount <= maxErrorKindsPerLumi_) errorsBelowThreshold++;
        } else if (iSeverity == edm::ELseverityLevel::ELsev_warning){
          unsigned int& iCount = warningCounts_[kind];
          iCount++;
          if (iCount <= maxWarningKindsPerLumi_) warningsBelowThreshold++;
        }

      }
      return ( (atLeastOneEntry_ && (errorsBelowThreshold > 0 || warningsBelowThreshold > 0))
              || (atLeastOneError_ && errorsBelowThreshold > 0)
              || (atLeastOneWarning_ && warningsBelowThreshold > 0));
    } else {
      //no separation by kind, just count any errors/warnings
      if(atLeastOneEntry_) {
        for(auto const& summary: *errorsAndWarnings) {
          edm::ELseverityLevel const& severity = summary.severity;
          if(severity.getLevel() != edm::ELseverityLevel::ELsev_error and
             severity.getLevel() != edm::ELseverityLevel::ELsev_warning) {
            continue;
          }
          if(!avoidCategories_.empty()) {
            //veto categories from user input.
            if(std::find(avoidCategories_.begin(),avoidCategories_.end(),summary.category) != avoidCategories_.end()) {
              continue;
            }
          }
          return true;
        }
        return false;
      } else {
        if(atLeastOneError_ || atLeastOneWarning_) {
          unsigned int nError = 0;
          unsigned int nWarning = 0;
          for(auto const& summary: *errorsAndWarnings) {
            //veto categories from user input.
            if(!avoidCategories_.empty()) {
              if(std::find(avoidCategories_.begin(),avoidCategories_.end(), summary.category) != avoidCategories_.end()) {
                continue;
              }
            }
            edm::ELseverityLevel const& severity = summary.severity;
            //count errors
            if(severity.getLevel() == edm::ELseverityLevel::ELsev_error) {
              ++nError;
            }
            //count warnings
            if(severity.getLevel() == edm::ELseverityLevel::ELsev_warning) {
              ++nWarning;
            }
          }
          if(atLeastOneError_ && nError != 0) {
            return (true);
          }
          if(atLeastOneWarning_ && nWarning != 0) {
            return (true);
          }
        }
      }
    }
  }
  return (false);
}

void LogErrorFilter::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&){
  if (useThresholdsPerKind_) {
    for(auto& err: errorCounts_) {
      err.second = 0;
    }
    for(auto& err: warningCounts_) {
      err.second = 0;
    }
  }

  return;
}


void
LogErrorFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("harvesterTag");
  desc.add<bool>("atLeastOneError");
  desc.add<bool>("atLeastOneWarning");
  desc.add<bool>("useThresholdsPerKind");
  desc.add<unsigned int>("maxErrorKindsPerLumi", 999999);
  desc.add<unsigned int>("maxWarningKindsPerLumi", 999999);
  desc.add<std::vector<std::string> >("avoidCategories");
  descriptions.add("logErrorFilter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LogErrorFilter);

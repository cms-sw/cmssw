// -*- C++ -*-
//
// Package:    LogMessageMonitor
// Class:      LogMessageMonitor
//
/**\class LogMessageMonitor LogMessageMonitor.cc DQM/LogMonitor/src/LogMessageMonitor.cc

 Description: [one line class summary]
  from https://twiki.cern.ch/twiki/bin/view/CMS/TrackingPOGFilters#Filters
   Events with (partly) aborted track reconstruction 
   The track reconstruction code is protected against events with too large occupancy which can cause an excessive use of CPU time and memory.
   Each iteration of the track reconstruction can be aborted if:
     - too many strip and/or pixel clusters are present as input to the seeding step (*TooManyClusters* error).
       => No track is reconstructed from that iteration
     - too many hit triplets or pairs are produced as input to the seeding step (*TooManyPairs/TooManyTriplets* errors).
       => All the pairs/triplets found are discarded and the iteration continue (to be checked!)
       NB: Despite the thrshold is the same,
           similar iterations may have a different rate of errors depending on the CMSSW release,
	   because the requirement to accept a triplet/pair has been modified (cluster shape filters,...) 
     - too many seeds are produced as input to the track building step (*TooManySeeds*).
	=> No track is reconstructed from that iteration. 

   # ELSeverityLevel
      http://cmssdt.cern.ch/SDT/lxr/source/FWCore/MessageLogger/interface/ELseverityLevel.h?v=CMSSW_5_3_4
108   enum ELsev_  {
109     ELsev_noValueAssigned = 0  // default returned by map when not found
110   , ELsev_zeroSeverity         // threshold use only
111   , ELsev_incidental           // flash this on a screen
112   , ELsev_success              // report reaching a milestone
113   , ELsev_info                 // information
114   , ELsev_warning              // warning
115   , ELsev_warning2             // more serious warning
116   , ELsev_error                // error detected
117   , ELsev_error2               // more serious error
118   , ELsev_next                 // advise to skip to next event
119   , ELsev_unspecified          // severity was not specified
120   , ELsev_severe               // future results are suspect
121   , ELsev_severe2              // more severe
122   , ELsev_abort                // suggest aborting
123   , ELsev_fatal                // strongly suggest aborting!
124   , ELsev_highestSeverity      // threshold use only
125   // -----
126   , nLevels                    // how many levels?
127   };  // ELsev_

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mia Tosi,40 3-B32,+41227671609,
//         Created:  Thu Mar  8 14:34:13 CET 2012
//
//

#include "DQM/TrackingMonitor/interface/LogMessageMonitor.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LogMessageMonitor::LogMessageMonitor(const edm::ParameterSet& iConfig)
    : dqmStore_(edm::Service<DQMStore>().operator->()),
      conf_(iConfig),
      pluginsMonName_(iConfig.getParameter<std::string>("pluginsMonName")),
      modules_vector_(iConfig.getParameter<std::vector<std::string> >("modules")),
      categories_vector_(iConfig.getParameter<std::vector<std::string> >("categories")),
      doWarningsPlots_(iConfig.getParameter<bool>("doWarningsPlots")),
      doPUmonitoring_(iConfig.getParameter<bool>("doPUmonitoring")) {
  errorToken_ = consumes<std::vector<edm::ErrorSummaryEntry> >(edm::InputTag("logErrorHarvester"));

  edm::ConsumesCollector c{consumesCollector()};
  //now do what ever initialization is needed
  lumiDetails_ = new GetLumi(iConfig.getParameter<edm::ParameterSet>("BXlumiSetup"), c);
  genTriggerEventFlag_ = new GenericTriggerEventFlag(
      iConfig.getParameter<edm::ParameterSet>("genericTriggerEventPSet"), consumesCollector(), *this);
}

LogMessageMonitor::~LogMessageMonitor() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //  if ( lumiDetails_         ) delete lumiDetails_;
  if (genTriggerEventFlag_)
    delete genTriggerEventFlag_;
}

//
// member functions
//

// ------------ method called for each event  ------------
void LogMessageMonitor::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Filter out events if Trigger Filtering is requested
  if (genTriggerEventFlag_->on() && !genTriggerEventFlag_->accept(iEvent, iSetup))
    return;

  double BXlumi = -1.;
  if (doPUmonitoring_)
    lumiDetails_->getValue(iEvent);

  // Take the ErrorSummaryEntry container
  edm::Handle<std::vector<edm::ErrorSummaryEntry> > errors;
  iEvent.getByToken(errorToken_, errors);
  // Check that errors is valid
  if (!errors.isValid())
    return;
  // Compare severity level of error with ELseveritylevel instance el : "-e" should be the lowest error
  edm::ELseverityLevel el("-e");

  // Find the total number of errors in iEvent
  if (errors->empty()) {
    if (doPUmonitoring_) {
      for (size_t i = 0; i < modulesMap.size(); i++) {
        ModulesErrorsVsBXlumi[i]->Fill(BXlumi, 0.);
        if (doWarningsPlots_)
          ModulesWarningsVsBXlumi[i]->Fill(BXlumi, 0.);
      }
    }
  } else {
    size_t nCategories = categories_vector_.size();

    for (size_t i = 0, n = errors->size(); i < n; i++) {
      //      std::cout << "LogMessageMonitor::analyze] Severity for error/warning: " << (*errors)[i].severity << " " <<(*errors)[i].module  << std::endl;
      // remove the first part of the module string, what is before ":"
      std::string s = (*errors)[i].module;
      size_t pos = s.find(':');
      std::string s_module = s.substr(pos + 1, s.size());
      std::map<std::string, int>::const_iterator it = modulesMap.find(s_module);
      if (it != modulesMap.end()) {
        //      std::cout << "LogMessageMonitor::analyze] it: " << " --> " << s_module << std::endl;

        // IF THIS IS AN ERROR on the ELseverityLevel SCALE, FILL ERROR HISTS
        if ((*errors)[i].severity.getLevel() >= edm::ELseverityLevel::ELsev_error) {
          if (doPUmonitoring_)
            ModulesErrorsVsBXlumi[it->second]->Fill(BXlumi, (*errors)[i].count);

          // loop over the different categories of errors
          // defined by configuration file
          // if the category is not in the given list
          // it fills the bin "others"
          TString module = it->first;
          TString category = (*errors)[i].category;
          int ibinX = CategoriesVsModules->getTH1()->GetXaxis()->FindBin(module);
          int ibinY = CategoriesVsModules->getTH1()->GetYaxis()->FindBin(category);
          /*
	  std::cout << "LogMessageMonitor::analyze] ibinX: " << ibinX << " it->second: " << it->second << " it->first: " << it->first << std::endl;
	  std::cout << "LogMessageMonitor::analyze] ibinY: " << ibinY << " (*errors)[i].category: " << (*errors)[i].category << std::endl;
	  size_t nbinsX = CategoriesVsModules->getTH1()->GetNbinsX();
	  for (size_t bin = 1; bin<=nbinsX; bin++) {
	    std::cout << "binX" << bin << ": " << CategoriesVsModules->getTH1()->GetXaxis()->GetBinLabel(bin) << std::endl;
	  }
	  size_t nbinsY = CategoriesVsModules->getTH1()->GetNbinsY();
	  for (size_t bin = 1; bin<=nbinsY; bin++) {
	    std::cout << "binY" << bin << ": " << CategoriesVsModules->getTH1()->GetYaxis()->GetBinLabel(bin) << std::endl;
	  }
	  */
          if (ibinY > 0) {
            int ncount = CategoriesVsModules->getTH1()->GetBinContent(ibinX, ibinY) + 1;
            CategoriesVsModules->getTH1()->SetBinContent(ibinX, ibinY, ncount);
          } else {
            int ncount = CategoriesVsModules->getTH1()->GetBinContent(ibinX, nCategories) + 1;
            CategoriesVsModules->getTH1()->SetBinContent(ibinX, nCategories, ncount);
          }
        } else {
          // IF ONLY WARNING, FILL WARNING HISTS
          if (doWarningsPlots_)
            if (doPUmonitoring_)
              ModulesWarningsVsBXlumi[it->second]->Fill(BXlumi, (*errors)[i].count);
        }
      }
    }
  }
}

void LogMessageMonitor::bookHistograms(DQMStore::IBooker& ibooker,
                                       edm::Run const& iRun,
                                       edm::EventSetup const& iSetup) {
  if (genTriggerEventFlag_->on())
    genTriggerEventFlag_->initRun(iRun, iSetup);

  std::string MEFolderName = conf_.getParameter<std::string>("LogFolderName");

  ibooker.setCurrentFolder(MEFolderName);

  categories_vector_.push_back("others");
  size_t nModules = modules_vector_.size();
  size_t nCategories = categories_vector_.size();

  histname = pluginsMonName_ + "ErrorsVsModules";
  CategoriesVsModules =
      ibooker.book2D(histname, histname, nModules, 0., double(nModules), nCategories, 0., double(nCategories));
  CategoriesVsModules->getTH1()->GetXaxis()->SetNoAlphanumeric();
  CategoriesVsModules->getTH1()->GetYaxis()->SetNoAlphanumeric();
  CategoriesVsModules->setAxisTitle("modules", 1);
  for (size_t imodule = 0; imodule < nModules; imodule++)
    CategoriesVsModules->setBinLabel(imodule + 1, modules_vector_[imodule], 1);
  CategoriesVsModules->setAxisTitle("categories", 2);
  //   CategoriesVsModules->getTH1()->GetXaxis()->LabelsOption("v");
  for (size_t icategories = 0; icategories < nCategories; icategories++)
    CategoriesVsModules->setBinLabel(icategories + 1, categories_vector_[icategories], 2);

  // MAKE MODULEMAP USING INPUT FROM CFG FILE
  for (size_t i = 0; i < modules_vector_.size(); i++) {
    modulesMap.insert(std::pair<std::string, int>(modules_vector_[i], i));
  }

  if (doPUmonitoring_) {
    // BOOK THE HISTOGRAMS
    // get binning from the configuration
    edm::ParameterSet BXlumiParameters = conf_.getParameter<edm::ParameterSet>("BXlumiSetup");
    int BXlumiBin = BXlumiParameters.getParameter<int>("BXlumiBin");
    double BXlumiMin = BXlumiParameters.getParameter<double>("BXlumiMin");
    double BXlumiMax = BXlumiParameters.getParameter<double>("BXlumiMax");

    size_t i = 0;
    for (std::map<std::string, int>::const_iterator it = modulesMap.begin(); it != modulesMap.end(); ++it, i++) {
      ibooker.setCurrentFolder(MEFolderName + "/PUmonitoring/Errors");

      histname = "errorsVsBXlumi_" + it->first;
      ModulesErrorsVsBXlumi.push_back(dynamic_cast<MonitorElement*>(
          ibooker.bookProfile(histname, histname, BXlumiBin, BXlumiMin, BXlumiMax, 0., 100, "")));
      ModulesErrorsVsBXlumi[i]->setAxisTitle("BXlumi [10^{30} Hz cm^{-2}]", 1);
      ModulesErrorsVsBXlumi[i]->setAxisTitle("Mean number of errors", 2);

      if (doWarningsPlots_) {
        ibooker.setCurrentFolder(MEFolderName + "/PUmonitoring/Warnings");

        histname = "warningVsBXlumi_" + it->first;
        ModulesWarningsVsBXlumi.push_back(dynamic_cast<MonitorElement*>(
            ibooker.bookProfile(histname, histname, BXlumiBin, BXlumiMin, BXlumiMax, 0., 100, "")));
        ModulesWarningsVsBXlumi[i]->setAxisTitle("BXlumi [10^{30} Hz cm^{-2}]", 1);
        ModulesWarningsVsBXlumi[i]->setAxisTitle("Mean number of warnings", 2);
      }
    }
  }
}

/*
// ------------ method called once each job just before starting event loop  ------------
void 
LogMessageMonitor::beginJob()
{

}
*/
// ------------ method called once each job just after ending the event loop  ------------
void LogMessageMonitor::endJob() {
  bool outputMEsInRootFile = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = conf_.getParameter<std::string>("OutputFileName");
  if (outputMEsInRootFile) {
    dqmStore_->save(outputFileName);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LogMessageMonitor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(LogMessageMonitor);

// -*- C++ -*-
//
// Package:    GeneralHLTOffline
// Class:      GeneralHLTOffline
//
/**\class GeneralHLTOffline

Description: [one line class summary]
Implementation:
[Notes on implementation]
*/
//
// Original Author:  Jason Michael Slaunwhite,512 1-008,`+41227670494,
//         Created:  Fri Aug  5 10:34:47 CEST 2011
//
//

// system include files
#include <memory>

// user include files
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TMath.h"
#include "TStyle.h"

//
// class declaration
//

class GeneralHLTOffline : public DQMEDAnalyzer {
 public:
  explicit GeneralHLTOffline(const edm::ParameterSet&);
  ~GeneralHLTOffline();

 private:
  // virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const & iRun,
			      edm::EventSetup const & iSetup) override;
  virtual void dqmBeginRun(edm::Run const& iRun,edm::EventSetup const& iSetup) override;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
                                    edm::EventSetup const&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&,
                                  edm::EventSetup const&) override;
  virtual void setupHltMatrix(DQMStore::IBooker & iBooker, const std::string &, int);
  virtual void fillHltMatrix(const std::string &,
                             const std::string &,
                             bool,
                             bool,
			     edm::Handle<trigger::TriggerEventWithRefs>,
			     edm::Handle<trigger::TriggerEvent>);

  // ----------member data ---------------------------


  bool debugPrint;
  bool outputPrint;
  bool streamA_found_;
  HLTConfigProvider hlt_config_;


  std::string plotDirectoryName;
  std::string hltTag;
  std::string hlt_menu_;
  std::vector< std::vector<std::string> > PDsVectorPathsVector;
  std::vector<std::string> AddedDatasets;
  std::vector<std::string> DataSetNames;
  std::map< std::string, std::vector<std::string> > PathModules;
  edm::EDGetTokenT <edm::TriggerResults>   triggerResultsToken;
  edm::EDGetTokenT <trigger::TriggerEventWithRefs> triggerSummaryTokenRAW;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryTokenAOD;

  MonitorElement * cppath_;
  std::map<std::string, MonitorElement*> cppath_mini_;
  std::map<std::string, MonitorElement*> cpfilt_mini_;
  std::map<std::string, TH1F*> hist_cpfilt_mini_;

};

//
// constructors and destructor
//
GeneralHLTOffline::GeneralHLTOffline(const edm::ParameterSet& ps):streamA_found_(false),
                                                                  hlt_menu_(""),
                                                                  cppath_(0) {
  debugPrint  = false;
  outputPrint = false;

  plotDirectoryName = ps.getUntrackedParameter<std::string>("dirname",
                                                            "HLT/General");

  hltTag = ps.getParameter<std::string> ("HltProcessName");

  triggerSummaryTokenRAW = consumes <trigger::TriggerEventWithRefs> (edm::InputTag(std::string("hltTriggerSummaryRAW"), std::string(""), hltTag));
  triggerSummaryTokenAOD = consumes <trigger::TriggerEvent> (edm::InputTag(std::string("hltTriggerSummaryAOD"), std::string(""), hltTag));
  triggerResultsToken = consumes <edm::TriggerResults>   (edm::InputTag(std::string("TriggerResults"), std::string(""), hltTag));

  if (debugPrint) {
    std::cout << "Inside Constructor" << std::endl;
    std::cout << "Got plot dirname = " << plotDirectoryName << std::endl;
  }
}


GeneralHLTOffline::~GeneralHLTOffline() {
}

// ------------ method called for each event  ------------
void
GeneralHLTOffline::analyze(const edm::Event& iEvent,
                           const edm::EventSetup& iSetup) {
  if (debugPrint)
    std::cout << "Inside analyze - run, block, event "
              << iEvent.id().run() << " , " << iEvent.id().luminosityBlock()
              << " , " << iEvent.id() << " , " << std::endl;

  // Access Trigger Results
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken, triggerResults);

  if (!triggerResults.isValid()) {
    if (debugPrint)
      std::cout << "Trigger results not valid" << std::endl;
    return;
  }

  if (debugPrint)
    std::cout << "Found triggerResults" << std::endl;

  edm::Handle<trigger::TriggerEventWithRefs> rawTriggerEvent;
  iEvent.getByToken(triggerSummaryTokenRAW, rawTriggerEvent);

  edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
  iEvent.getByToken(triggerSummaryTokenAOD, aodTriggerEvent);

  bool hasRawTriggerEvent = true;
  if( !rawTriggerEvent.isValid() ){
    hasRawTriggerEvent = false;
    if (debugPrint)
      std::cout << "No RAW trigger summary found! Returning...";

    if( !aodTriggerEvent.isValid() ){
      if (debugPrint)
	std::cout << "No AOD trigger summary found! Returning...";
      return;
    }
  }

  if (streamA_found_) {
    const std::vector<std::string> &datasetNames =  DataSetNames;
    // Loop over PDs
    for (unsigned int iPD = 0; iPD < datasetNames.size(); iPD++) {
      // Loop over Paths in each PD
      for (unsigned int iPath = 0;
           iPath < PDsVectorPathsVector[iPD].size(); iPath++) {
        std::string &pathName = PDsVectorPathsVector[iPD][iPath];
        unsigned int index = hlt_config_.triggerIndex(pathName);
	if (debugPrint) {
          std::cout << "Looking at path " << pathName << std::endl;
          std::cout << "Index = " << index
                    << " triggerResults->size() = " << triggerResults->size()
                    << std::endl;
	}

        // fill the histos with empty weights......
        const std::string &label = datasetNames[iPD];
        std::string fullPathToCPP = "HLT/GeneralHLTOffline/"
            + label + "/cppath_" + label + hlt_menu_;
        MonitorElement * ME_mini_cppath = NULL;
        TH1F * hist_mini_cppath = NULL;
        if( cppath_mini_.find(fullPathToCPP)!=cppath_mini_.end() ){
	  ME_mini_cppath = cppath_mini_[fullPathToCPP];
          hist_mini_cppath = ME_mini_cppath->getTH1F();
	}

        if (hist_mini_cppath) {
          TAxis * axis = hist_mini_cppath->GetXaxis();
          if (axis) {
	    std::string pathNameNoVer = hlt_config_.removeVersion(PDsVectorPathsVector[iPD][iPath]);
            int bin_num = axis->FindBin(pathNameNoVer.c_str());
            int bn = bin_num - 1;
            hist_mini_cppath->Fill(bn, 0);
            hist_mini_cppath->SetEntries(hist_mini_cppath->Integral());
          }
        }

        if( index < triggerResults->size() ) {
	  bool accept = triggerResults->accept(index);
	  if( accept ) cppath_->Fill(index, 1);

	  fillHltMatrix(datasetNames[iPD], pathName,
			accept,
			hasRawTriggerEvent,
			rawTriggerEvent,
			aodTriggerEvent);
        }  // end if (index < triggerResults->size())
      }  // end Loop over Paths in each PD
    }  // end Loop over PDs
  }
}



// ------------ method called when starting to processes a run  ------------
void
GeneralHLTOffline::dqmBeginRun(edm::Run const& iRun,
			       edm::EventSetup const& iSetup) {
  if (debugPrint)
    std::cout << "Inside beginRun" << std::endl;

  // Reset "condition" variables that could have memory of previous
  // runs.

  PDsVectorPathsVector.clear();
  AddedDatasets.clear();
  DataSetNames.clear();
  PathModules.clear();


  bool changed = true;
  if (!hlt_config_.init(iRun, iSetup, hltTag, changed)) {
    if (debugPrint) {
      std::cout << "Warning, didn't find process HLT" << std::endl;
      return;
    }
  } else {
    if (debugPrint)
      std::cout << " HLTConfig processName " << hlt_config_.processName()
                << " tableName " << hlt_config_.tableName()
                << " size " << hlt_config_.size() << std::endl;
  }
  hlt_menu_ = hlt_config_.tableName();
  for (unsigned int n = 0, e = hlt_menu_.length(); n != e; ++n)
    if (hlt_menu_[n] == '/' || hlt_menu_[n] == '.')
      hlt_menu_[n] = '_';


  const std::vector<std::string> &nameStreams = hlt_config_.streamNames();
  std::vector<std::string>::const_iterator si = nameStreams.begin();
  std::vector<std::string>::const_iterator se = nameStreams.end();
  std::vector<std::string> datasetNames;

  for ( ; si != se; ++si) {
    if (debugPrint) std::cout << "This is stream " << *si << std::endl;

    if ( ((*si).find("Physics") != std::string::npos) ||((*si).find("Scouting") != std::string::npos) ||((*si).find("Parking") != std::string::npos) || (*si) == "A") {
      streamA_found_ = true;

      std::vector<std::string> datasetperStream = hlt_config_.streamContent(*si);
      
      for (auto const & di: datasetperStream) {
	datasetNames.push_back(di);
      }
    }
  }

  if (debugPrint) std::cout << "Number of total  datasets " << datasetNames.size() << std::endl;


  if (streamA_found_) {
   
    DataSetNames = datasetNames;
    
    if (debugPrint)
      std::cout << "Number of datasets to be monitored "
                << datasetNames.size() << std::endl;
    
    for (unsigned int i = 0; i < datasetNames.size(); i++) {
      const std::vector<std::string> &datasetPaths = hlt_config_.datasetContent(datasetNames[i]);
      if (debugPrint) {
        std::cout << "This is dataset " << datasetNames[i]
                  << "datasetPaths.size() = " << datasetPaths.size() << std::endl;
        for (unsigned int iPath = 0;
             iPath < datasetPaths.size(); iPath++) {
          std::cout << "Paths in begin job "
                    << datasetPaths[iPath] << std::endl;
        }
      }
      
      // Check if dataset has been added - if not add it
      // need to loop through AddedDatasets and compare
      bool foundDataset = false;
      int datasetNum = -1;
      for (unsigned int d = 0; d < AddedDatasets.size(); d++) {
        if (AddedDatasets[d].compare(datasetNames[i]) == 0) {
          foundDataset = true;
          datasetNum = d;
          if (debugPrint)
            std::cout << "Dataset " << datasetNames[i]
                      << " found in AddedDatasets at position " << d << std::endl;
          break;
        }
      }
      
      if (!foundDataset) {
        if (debugPrint)
          std::cout << " Fill trigger paths for dataset "
                    << datasetNames[i] << std::endl;
        PDsVectorPathsVector.push_back(datasetPaths);
        // store dataset pathname
        AddedDatasets.push_back(datasetNames[i]);
      } else {
        // This trigger path has already been added - this implies that
        // this is a new run What we want to do is check if there is a
        // new trigger that was not in the original dataset For a given
        // dataset, loop over the stored list of triggers, and compare
        // to the current list of triggers If any of the triggers are
        // missing, add them to the end of the appropriate dataset
        if (debugPrint)
          std::cout << " Additional runs : Check for additional"
                    << "trigger paths per dataset " << std::endl;
        // Loop over correct path of PDsVectorPathsVector
        bool found = false;
	
        // Loop over triggers in the path
        for (unsigned int iTrig = 0; iTrig < datasetPaths.size(); iTrig++) {
          if (debugPrint)
            std::cout << "Looping over trigger list in dataset "
                      <<  iTrig <<  "  "
                      << datasetPaths[iTrig] << std::endl;
          found = false;
          // Loop over triggers already on the list
          for (unsigned int od = 0; od < PDsVectorPathsVector[datasetNum].size(); od++) {
            if (debugPrint)
              std::cout << "Looping over existing trigger list " << od
                        <<  "  " << PDsVectorPathsVector[datasetNum][od] << std::endl;
            // Compare, see if match is found
            if (hlt_config_.removeVersion(datasetPaths[iTrig]).compare(
								       hlt_config_.removeVersion(PDsVectorPathsVector[datasetNum][od])) == 0) {
              found = true;
              if (debugPrint)
                std::cout << " FOUND " << datasetPaths[iTrig] << std::endl;
              break;
            }
          }
          // If match is not found, add trigger to correct path of PDsVectorPathsVector
          if (!found)
            PDsVectorPathsVector[datasetNum].push_back(datasetPaths[iTrig]);
          if (debugPrint)
            std::cout << datasetPaths[iTrig]
                      << "  NOT FOUND - so we added it to the correct dataset "
                      << datasetNames[i] << std::endl;
        }
      }
      // Let's check this whole big structure
      if (debugPrint) {
        for (unsigned int is = 0; is < PDsVectorPathsVector.size(); is++) {
          std::cout << "   PDsVectorPathsVector[" << is << "] is "
                    << PDsVectorPathsVector[is].size() << std::endl;
          for (unsigned int ip = 0; ip < PDsVectorPathsVector[is].size(); ip++) {
            std::cout << "    trigger " << ip << " path "
                      << PDsVectorPathsVector[is][ip] << std::endl;
          }
        }
      }

      if (debugPrint)
        std::cout <<"Found PD: " << datasetNames[i] << std::endl;
    }  // end of loop over dataset names



    std::vector<std::string> triggerNames = hlt_config_.triggerNames();

    for( unsigned int iPath=0; iPath<triggerNames.size(); iPath++ ){
      std::string pathName = triggerNames[iPath];

      const std::vector<std::string>& moduleLabels = hlt_config_.moduleLabels(pathName);
      int NumModules = int( moduleLabels.size() );

      if( !(pathName.find("HLT_") != std::string::npos) ) continue;
      if( (pathName.find("HLT_Physics")!=std::string::npos) ||
	  (pathName.find("HLT_Random")!=std::string::npos) ) continue;

      std::string prefix("hltPre");

      std::vector<std::string> good_module_names;
      good_module_names.clear();
      for( int iMod=0; iMod<NumModules; iMod++ ){
	std::string moduleType = hlt_config_.moduleType(moduleLabels[iMod]);
	std::string moduleEDMType = hlt_config_.moduleEDMType(moduleLabels[iMod]);
	if( !(moduleEDMType == "EDFilter") ) continue;
	if( moduleType.find("Selector")!= std::string::npos ) continue;
	if( moduleType == "HLTTriggerTypeFilter" || 
	    moduleType == "HLTBool" ||
	    moduleType == "PrimaryVertexObjectFilter" ||
	    moduleType == "JetVertexChecker" ||
	    moduleType == "HLTRHemisphere" ||
	    moduleType == "DetectorStateFilter" ) continue;

	if( moduleLabels[iMod].compare(0, prefix.length(), prefix) == 0 ) continue;
	good_module_names.push_back(moduleLabels[iMod]);
      }
      PathModules[pathName] = good_module_names;
    } // loop over paths

  }  // if stream A or Physics streams found


}


// ------------ method called to book histograms before starting event loop  ------------
void GeneralHLTOffline::bookHistograms(DQMStore::IBooker & iBooker,
				       edm::Run const & iRun,
				       edm::EventSetup const & iSetup)
{
  iBooker.setCurrentFolder(plotDirectoryName) ;

  //////////// Book a simple ME

  iBooker.setCurrentFolder("HLT/GeneralHLTOffline/");
  iBooker.bookString("hltMenuName", hlt_menu_.c_str());
  cppath_ = iBooker.book1D("cppath" + hlt_menu_,
			   "Counts/Path",
			   hlt_config_.size(), 0, hlt_config_.size());

  if (streamA_found_) {

    for (unsigned int iPD = 0; iPD < DataSetNames.size(); iPD++)
      setupHltMatrix(iBooker, DataSetNames[iPD], iPD);

  }  // if stream A or Physics streams are found
}  // end of bookHistograms


void GeneralHLTOffline::setupHltMatrix(DQMStore::IBooker & iBooker, const std::string & label, int iPD) {
  std::string h_name;
  std::string h_title;
  std::string pathName;
  std::string PD_Folder;
  std::string Path_Folder;

  PD_Folder = TString("HLT/GeneralHLTOffline/"+label);

  iBooker.setCurrentFolder(PD_Folder.c_str());

  // make it the top level directory, that is on the same dir level as
  // paths
  std::string folderz;
  folderz = TString("HLT/GeneralHLTOffline/"+label);
  iBooker.setCurrentFolder(folderz.c_str());

  std::string dnamez = "cppath_" + label + "_" + hlt_menu_;
  int sizez = PDsVectorPathsVector[iPD].size();
  TH1F * hist_mini_cppath = NULL;
  cppath_mini_[dnamez] = iBooker.book1D(dnamez.c_str(),
					dnamez.c_str(),
					sizez,
					0,
					sizez);
  if( cppath_mini_[dnamez] )
    hist_mini_cppath = cppath_mini_[dnamez]->getTH1F();

  unsigned int jPath;
  for (unsigned int iPath = 0; iPath < PDsVectorPathsVector[iPD].size(); iPath++) {
    pathName = hlt_config_.removeVersion(PDsVectorPathsVector[iPD][iPath]);
    jPath = iPath + 1;

    if (hist_mini_cppath) {
      TAxis * axis = hist_mini_cppath->GetXaxis();
      if (axis)
        axis->SetBinLabel(jPath, pathName.c_str());
    }

    std::string pathNameVer = PDsVectorPathsVector[iPD][iPath];

    std::vector<std::string> moduleLabels = PathModules[pathNameVer];
    int NumModules = int( moduleLabels.size() );

    if( NumModules==0 ) continue;

    std::string pathName_dataset = "cpfilt_" + label + "_" + pathName;

    cpfilt_mini_[pathName_dataset] = iBooker.book1D(pathName_dataset.c_str(),
							  pathName.c_str(),
							  NumModules,
							  0,
							  NumModules);

    if( cpfilt_mini_[pathName_dataset] )
      hist_cpfilt_mini_[pathName_dataset] = cpfilt_mini_[pathName_dataset]->getTH1F();

    for( int iMod=0; iMod<NumModules; iMod++ ){
      if( cpfilt_mini_[pathName_dataset] && hist_cpfilt_mini_[pathName_dataset] ){
	TAxis * axis = hist_cpfilt_mini_[pathName_dataset]->GetXaxis();
	if (axis)
	  axis->SetBinLabel(iMod+1,moduleLabels[iMod].c_str());
      }
    }

    if (debugPrint)
      std::cout << "book1D for " << pathName << std::endl;
  }

  if (debugPrint)
    std::cout << "Success setupHltMatrix( " << label << " , "
              << iPD << " )" << std::endl;
}  // End setupHltMatrix


void GeneralHLTOffline::fillHltMatrix(const std::string & label,
                                      const std::string & path,
                                      bool accept,
				      bool hasRawTriggerEvent,
				      edm::Handle<trigger::TriggerEventWithRefs> triggerEventRAW,
				      edm::Handle<trigger::TriggerEvent> triggerEventAOD) {
  if (debugPrint)
    std::cout << "Inside fillHltMatrix( " << label << " , "
              << path << " ) " << std::endl;

  std::string fullPathToCPP;

  fullPathToCPP = "HLT/GeneralHLTOffline/" + label + "/cppath_" + label + "_" + hlt_menu_;
  
  std::string dnamez = "cppath_" + label + "_" + hlt_menu_;

  TH1F * hist_mini_cppath = NULL;
  MonitorElement * ME_mini_cppath = NULL;
  if( cppath_mini_.find(dnamez)!=cppath_mini_.end() ){
    ME_mini_cppath = cppath_mini_[dnamez];
    hist_mini_cppath = ME_mini_cppath->getTH1F();
  }

  std::string pathNameNoVer = hlt_config_.removeVersion(path);

  if( (path.find("HLT_") != std::string::npos) && 
      !(path.find("HLT_Physics")!=std::string::npos) && 
      !(path.find("HLT_Random")!=std::string::npos) ){

    unsigned int triggerEventSize = 0;
    if( hasRawTriggerEvent && triggerEventRAW.isValid() ) triggerEventSize = triggerEventRAW->size();
    else if( triggerEventAOD.isValid() ) triggerEventSize = triggerEventAOD->sizeFilters();

    std::string pathName_dataset = "cpfilt_" + label + "_" + pathNameNoVer;

    TH1F * hist_cpfilt_mini = NULL;
    MonitorElement * ME_cpfilt_mini = NULL;
    if( cpfilt_mini_.find(pathName_dataset)!=cpfilt_mini_.end() ){
      ME_cpfilt_mini = cpfilt_mini_[pathName_dataset];
      hist_cpfilt_mini = ME_cpfilt_mini->getTH1F();
    }


    std::vector<std::string> moduleLabels = PathModules[path];
    int NumModules = int( moduleLabels.size() );

    for( int iMod=0; iMod<NumModules; iMod++ ){
      edm::InputTag moduleWhoseResultsWeWant(moduleLabels[iMod],
					     "",
					     hltTag);

      unsigned int idx_module_trg = 0;
      if( hasRawTriggerEvent && triggerEventRAW.isValid() ) idx_module_trg = triggerEventRAW->filterIndex(moduleWhoseResultsWeWant);
      else if( triggerEventAOD.isValid() ) idx_module_trg = triggerEventAOD->filterIndex(moduleWhoseResultsWeWant);
  
      if( !(idx_module_trg < triggerEventSize) ) continue;
      if( hist_cpfilt_mini ){
	TAxis * axis = hist_cpfilt_mini->GetXaxis();
	int bin_num = axis->FindBin(moduleLabels[iMod].c_str());
	int bn = bin_num - 1;

	if( bin_num!=1 && hasRawTriggerEvent ){
	  bool passPreviousFilters = true;
	  for( int ibin = bin_num-1; ibin>0; ibin-- ){ 
	    std::string previousFilter(axis->GetBinLabel(ibin));
	    edm::InputTag previousModuleWhoseResultsWeWant(previousFilter,
							   "",
							   hltTag);
	    unsigned int idx_previous_module_trg = 0;
	    if( hasRawTriggerEvent && triggerEventRAW.isValid() ) idx_previous_module_trg = triggerEventRAW->filterIndex(previousModuleWhoseResultsWeWant);
	    else if( triggerEventAOD.isValid() ) idx_previous_module_trg = triggerEventAOD->filterIndex(previousModuleWhoseResultsWeWant);

	    if( !(idx_previous_module_trg < triggerEventSize) ){
	      passPreviousFilters = false;
	      break;
	    }
	  }
	  // Only fill if previous filters have been passed
	  if( passPreviousFilters ) hist_cpfilt_mini->Fill(bn, 1);
	}
	else hist_cpfilt_mini->Fill(bn, 1);

      }
    }
  }
  else{
    if (debugPrint) std::cout << "No AOD trigger summary found! Returning..." << std::endl;
  }

  if( accept && hist_mini_cppath ){
    TAxis * axis = hist_mini_cppath->GetXaxis();
    int bin_num = axis->FindBin(pathNameNoVer.c_str());
    int bn = bin_num - 1;
    hist_mini_cppath->Fill(bn, 1);
  }

  if (debugPrint)
    std::cout << "hist->Fill" << std::endl;
}  // End fillHltMatrix

void GeneralHLTOffline::beginLuminosityBlock(edm::LuminosityBlock const&,
                                             edm::EventSetup const&) {
}

void
GeneralHLTOffline::endLuminosityBlock(edm::LuminosityBlock const&,
                                      edm::EventSetup const&) {
}

DEFINE_FWK_MODULE(GeneralHLTOffline);

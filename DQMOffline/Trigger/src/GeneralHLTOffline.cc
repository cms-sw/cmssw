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
                             double, double, bool);

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
  edm::EDGetTokenT <edm::TriggerResults>   triggerResultsToken;
  edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;

  MonitorElement * cppath_;
  std::map<std::string, MonitorElement*> cppath_mini_;
  std::map<std::string, MonitorElement*> me_etaphi_;
  std::map<std::string, MonitorElement*> me_eta_;
  std::map<std::string, MonitorElement*> me_phi_;

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

  triggerSummaryToken = consumes <trigger::TriggerEvent> (edm::InputTag(std::string("hltTriggerSummaryAOD"), std::string(""), hltTag));
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

  edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
  iEvent.getByToken(triggerSummaryToken, aodTriggerEvent);

  if (!aodTriggerEvent.isValid()) {
    if (debugPrint)
      std::cout << "No AOD trigger summary found! Returning...";
    return;
  }

  const trigger::TriggerObjectCollection objects = aodTriggerEvent->getObjects();

  if (streamA_found_) {
    const std::vector<std::string> &datasetNames =  hlt_config_.streamContent("A");
    // Loop over PDs
    for (unsigned int iPD = 0; iPD < datasetNames.size(); iPD++) {
      // Loop over Paths in each PD
      bool first_count = true;
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

        if (index < triggerResults->size()) {
          if (triggerResults->accept(index)) {
            cppath_->Fill(index, 1);
            if (debugPrint)
              std::cout << "Check Event " <<  iEvent.id()
                        << " Run " << iEvent.id().run()
                        << " fired path " << pathName << std::endl;

            // look up module labels for this path
            const std::vector<std::string> &modulesThisPath =
                hlt_config_.moduleLabels(pathName);

            if (debugPrint)
              std::cout << "Looping over module labels " << std::endl;

            // Loop backward through module names
            for (int iModule = (modulesThisPath.size() - 1);
                 iModule >= 0; iModule--) {
              if (debugPrint)
                std::cout << "Module name is "
                          << modulesThisPath[iModule] << std::endl;
              // check to see if you have savetags information
              if (hlt_config_.saveTags(modulesThisPath[iModule])) {
                if (debugPrint)
                  std::cout << "For path " << pathName
                            << " this module " << modulesThisPath[iModule]
                            <<" is a saveTags module of type "
                            << hlt_config_.moduleType(modulesThisPath[iModule])
                            << std::endl;
                if (hlt_config_.moduleType(modulesThisPath[iModule])
                    == "HLTLevel1GTSeed")
                  break;
                edm::InputTag moduleWhoseResultsWeWant(modulesThisPath[iModule],
                                                       "",
                                                       hltTag);
                unsigned int idx_module_aod_trg =
                    aodTriggerEvent->filterIndex(moduleWhoseResultsWeWant);
                if (idx_module_aod_trg < aodTriggerEvent->sizeFilters()) {
                  const trigger::Keys &keys =
                      aodTriggerEvent->filterKeys(idx_module_aod_trg);
                  if (debugPrint)
                    std::cout << "Got Keys for index "
                              << idx_module_aod_trg
                              <<", size of keys is " << keys.size()
                              << std::endl;
                  if (keys.size() >= 1000)
                    edm::LogWarning("GeneralHLTOffline")
                        << "WARNING!! size of keys is " << keys.size()
                        << " for path " << pathName << " and module "
                        << modulesThisPath[iModule]<< std::endl;

                  // There can be > 100 keys (3-vectors) for some
                  // modules with no ID filled the first one has the
                  // highest value for single-object triggers for
                  // multi-object triggers, seems reasonable to use
                  // the first one as well So loop here has been
                  // commented out for ( size_t iKey = 0; iKey <
                  // keys.size(); iKey++ ) {

                  if (keys.size() > 0) {
                    trigger::TriggerObject foundObject = objects[keys[0]];
                    if (debugPrint || outputPrint)
                      std::cout << "This object has id (pt, eta, phi) = "
                                << " " << foundObject.id() << " "
                                << std::setw(10) << foundObject.pt()
                                << ", " << std::setw(10) << foundObject.eta()
                                << ", " << std::setw(10) << foundObject.phi()
                                << "   for path = " << std::setw(20) << pathName
                                << " module " << std::setw(40)
                                << modulesThisPath[iModule] << std::endl;
                    if (debugPrint)
                      std::cout << "CHECK RUN " << iEvent.id().run() << " "
                                << iEvent.id() << " " << pathName << " "
                                << modulesThisPath[iModule] << " "
                                << datasetNames[iPD] << " "
                                << hlt_config_.moduleType(modulesThisPath[iModule])
                                << " " << keys.size() << " "
                                << std::setprecision(4) << foundObject.pt() << " "
                                << foundObject.eta() << " "
                                << foundObject.phi() << std::endl;

                    // first_count is to make sure that the top-level
                    // histograms of each dataset don't get filled
                    // more than once
                    fillHltMatrix(datasetNames[iPD], pathName,
                                  foundObject.eta(), foundObject.phi(),
                                  first_count);
                    first_count = false;
                  }  // at least one key
                }  // end if filter in aodTriggerEvent
                // OK, we found the last module. No need to look at
                // the others.  get out of the loop
                break;
              }  // end if saveTags
            }  // end Loop backward through module names
          }  // end if(triggerResults->accept(index))
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
}


// ------------ method called to book histograms before starting event loop  ------------
void GeneralHLTOffline::bookHistograms(DQMStore::IBooker & iBooker,
				       edm::Run const & iRun,
				       edm::EventSetup const & iSetup)
{
  iBooker.setCurrentFolder(plotDirectoryName) ;

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


  //////////// Book a simple ME

  iBooker.setCurrentFolder("HLT/GeneralHLTOffline/");
  iBooker.bookString("hltMenuName", hlt_menu_.c_str());
  cppath_ = iBooker.book1D("cppath" + hlt_menu_,
			   "Counts/Path",
			   hlt_config_.size(), 0, hlt_config_.size());

  const std::vector<std::string> &nameStreams = hlt_config_.streamNames();
  std::vector<std::string>::const_iterator si = nameStreams.begin();
  std::vector<std::string>::const_iterator se = nameStreams.end();
  for ( ; si != se; ++si) {
    if ((*si) == "A") {
      streamA_found_ = true;
      break;
    }
  }

  if (streamA_found_) {
    const std::vector<std::string> &datasetNames =  hlt_config_.streamContent("A");
    if (debugPrint)
      std::cout << "Number of Stream A datasets "
                << datasetNames.size() << std::endl;

    for (unsigned int i = 0; i < datasetNames.size(); i++) {
      const std::vector<std::string> &datasetPaths = hlt_config_.datasetContent(datasetNames[i]);
      if (debugPrint) {
        std::cout << "This is dataset " << datasetNames[i]
                  << "datasetPaths.size() = " << datasetPaths.size() << std::endl;
        for (unsigned int iPath = 0;
             iPath < datasetPaths.size(); iPath++) {
          std::cout << "Before setupHltMatrix -  MET dataset "
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

      setupHltMatrix(iBooker, datasetNames[i], i);
    }  // end of loop over dataset names
  }  // if stream A found
}  // end of beginRun

// // ------------ method called when ending the processing of a run  ------------
// void GeneralHLTOffline::endRun(edm::Run const&, edm::EventSetup const&) {
//   if (debugPrint)
//     std::cout << " endRun called " << std::endl;
// }


void GeneralHLTOffline::setupHltMatrix(DQMStore::IBooker & iBooker, const std::string & label, int iPD) {
  std::string h_name;
  std::string h_title;
  std::string h_name_1dEta;
  std::string h_name_1dPhi;
  std::string h_title_1dEta;
  std::string h_title_1dPhi;
  std::string h_name_1dEtaPath;
  std::string h_name_1dPhiPath;
  std::string h_title_1dEtaPath;
  std::string h_title_1dPhiPath;
  std::string pathName;
  std::string PD_Folder;
  std::string Path_Folder;

  PD_Folder = TString("HLT/GeneralHLTOffline/"+label);

  iBooker.setCurrentFolder(PD_Folder.c_str());
  h_name = "HLT_" +label + "_EtaVsPhi";
  h_title = "HLT_" + label + "_EtaVsPhi";
  h_name_1dEta = "HLT_" + label + "_1dEta";
  h_name_1dPhi = "HLT_" + label + "_1dPhi";
  h_title_1dEta = label + " Occupancy Vs Eta";
  h_title_1dPhi = label + " Occupancy Vs Phi";

  Int_t numBinsEta = 30;
  Int_t numBinsPhi = 34;
  Int_t numBinsEtaFine = 60;
  Int_t numBinsPhiFine = 66;
  Double_t EtaMax = 2.610;
  Double_t PhiMax = 17.0*TMath::Pi()/16.0;
  Double_t PhiMaxFine = 33.0*TMath::Pi()/32.0;

  me_etaphi_[h_name] = iBooker.book2D(h_name.c_str(),
				      h_title.c_str(),
				      numBinsEta, -EtaMax, EtaMax,
				      numBinsPhi, -PhiMax, PhiMax);
  if (TH1 * service_histo = me_etaphi_[h_name]->getTH2F())
    service_histo->SetMinimum(0);

  me_eta_[h_name_1dEta] = iBooker.book1D(h_name_1dEta.c_str(),
					 h_title_1dEta.c_str(),
					 numBinsEtaFine, -EtaMax, EtaMax);
  if (TH1 * service_histo = me_eta_[h_name_1dEta]->getTH1F())
    service_histo->SetMinimum(0);
  
  me_phi_[h_name_1dPhi] = iBooker.book1D(h_name_1dPhi.c_str(),
					 h_title_1dPhi.c_str(),
					 numBinsPhiFine, -PhiMaxFine, PhiMaxFine);
  if (TH1 * service_histo = me_phi_[h_name_1dPhi]->getTH1F())
    service_histo->SetMinimum(0);


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


    if (debugPrint)
      std::cout << "book1D for " << pathName << std::endl;
  }

  if (debugPrint)
    std::cout << "Success setupHltMatrix( " << label << " , "
              << iPD << " )" << std::cout;
}  // End setupHltMatrix


void GeneralHLTOffline::fillHltMatrix(const std::string & label,
                                      const std::string & path,
                                      double Eta,
                                      double Phi,
                                      bool first_count) {
  if (debugPrint)
    std::cout << "Inside fillHltMatrix( " << label << " , "
              << path << " ) " << std::endl;

  std::string fullPathToME;
  std::string fullPathToME1dEta;
  std::string fullPathToME1dPhi;
  std::string fullPathToME1dEtaPath;
  std::string fullPathToME1dPhiPath;
  std::string fullPathToCPP;

  fullPathToME = "HLT/GeneralHLTOffline/HLT_" + label + "_EtaVsPhi";
  fullPathToME1dEta = "HLT/GeneralHLTOffline/HLT_" + label + "_1dEta";
  fullPathToME1dPhi = "HLT/GeneralHLTOffline/HLT_" + label + "_1dPhi";
  fullPathToCPP = "HLT/GeneralHLTOffline/" + label
      + "/cppath_" + label + "_" + hlt_menu_;

  fullPathToME = "HLT/GeneralHLTOffline/"
    + label + "/HLT_" + label + "_EtaVsPhi";
  fullPathToME1dEta = "HLT/GeneralHLTOffline/"
    + label + "/HLT_" + label + "_1dEta";
  fullPathToME1dPhi = "HLT/GeneralHLTOffline/"
    + label + "/HLT_" + label + "_1dPhi";
  


  TH1F * hist_mini_cppath = NULL;
  MonitorElement * ME_mini_cppath = NULL;
  if( cppath_mini_.find(fullPathToCPP)!=cppath_mini_.end() ){
    ME_mini_cppath = cppath_mini_[fullPathToCPP];
    hist_mini_cppath = ME_mini_cppath->getTH1F();
  }

  // fill top-level histograms
  if (first_count) {
    if (debugPrint)
      std::cout << " label " << label << " fullPathToME1dPhi "
                << fullPathToME1dPhi << " path "  << path
                << " Phi " << Phi << " Eta " << Eta << std::endl;
    
    MonitorElement * ME_1dEta = NULL;
    if( me_eta_.find(fullPathToME1dEta)!=me_eta_.end() ){
      ME_1dEta = me_eta_[fullPathToME1dEta];
      TH1F * hist_1dEta = ME_1dEta->getTH1F();
      if (hist_1dEta)
	hist_1dEta->Fill(Eta);
    }
    MonitorElement * ME_1dPhi = NULL;
    if( me_phi_.find(fullPathToME1dPhi)!=me_phi_.end() ){
      ME_1dPhi = me_phi_[fullPathToME1dPhi];
      TH1F * hist_1dPhi = ME_1dPhi->getTH1F();
      if (hist_1dPhi)
	hist_1dPhi->Fill(Phi);
      if (debugPrint)
	std::cout << "  **FILLED** label " << label << " fullPathToME1dPhi "
		  << fullPathToME1dPhi << " path "  << path
		  << " Phi " << Phi << " Eta " << Eta << std::endl;
    }
    MonitorElement * ME_2d = NULL;
    if( me_etaphi_.find(fullPathToME)!=me_etaphi_.end() ){
      ME_2d = me_etaphi_[fullPathToME];
      TH2F * hist_2d = ME_2d->getTH2F();
      if (hist_2d)
	hist_2d->Fill(Eta, Phi);
    }
  }  // end fill top-level histograms


  if (debugPrint)
    if (label == "MET")
      std::cout << " MET Eta is " << Eta << std::endl;

  if (hist_mini_cppath) {
    TAxis * axis = hist_mini_cppath->GetXaxis();
    std::string pathNameNoVer = hlt_config_.removeVersion(path);
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

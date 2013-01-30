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
// $Id: GeneralHLTOffline.cc,v 1.9 2012/10/19 20:02:11 bjk Exp $
//
//

// system include files
#include <memory>

// user include files
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
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

using namespace edm;
using namespace trigger;
using namespace std; 
using std::vector;
using std::string;

class GeneralHLTOffline : public edm::EDAnalyzer {
   public:
      explicit GeneralHLTOffline(const edm::ParameterSet&);
      ~GeneralHLTOffline();

  //static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void setupHltMatrix(std::string, int);
      virtual void fillHltMatrix(std::string, std::string, double, double, bool);
      virtual string removeVersions(std::string);


      // ----------member data ---------------------------


  bool debugPrint;
  bool outputPrint;
  
  std::string plotDirectoryName;
  std::string hltTag;

  DQMStore * dbe;
  
  HLTConfigProvider hltConfig_;
  
  vector< vector<string> > PDsVectorPathsVector;
  vector<string> AddedDatasets;

  MonitorElement * cppath;
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
GeneralHLTOffline::GeneralHLTOffline(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed

  debugPrint = false;
  outputPrint = false;

  if (debugPrint) std::cout << "Inside Constructor" << std::endl;

  plotDirectoryName = iConfig.getUntrackedParameter<std::string>("dirname", "HLT/General");

   if (debugPrint) std::cout << "Got plot dirname = " << plotDirectoryName << std::endl;
  
  hltTag = iConfig.getParameter<std::string> ("HltProcessName");
  

}


GeneralHLTOffline::~GeneralHLTOffline()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
GeneralHLTOffline::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using std::string;


   if (debugPrint) std::cout << "Inside analyze - run, block, event " << iEvent.id().run() << " , " << iEvent.id().luminosityBlock() << " , " << iEvent.id() << " , " << std::endl;

		     //LuminosityBLock() << " , " << event() << std::endl;

    // Access Trigger Results
   edm::Handle<edm::TriggerResults> triggerResults;
   iEvent.getByLabel(InputTag("TriggerResults","", hltTag), triggerResults);
   
   if (!triggerResults.isValid()) {
     if (debugPrint) std::cout << "Trigger results not valid" << std::endl;
     return; }

    if (debugPrint) std::cout << "Found triggerResults" << std::endl;

   edm::Handle<trigger::TriggerEvent>         aodTriggerEvent;   
   iEvent.getByLabel(InputTag("hltTriggerSummaryAOD", "", hltTag), aodTriggerEvent);
   
   if ( !aodTriggerEvent.isValid() ) { 
     if (debugPrint) std::cout << "No AOD trigger summary found! Returning..."; 
     return; 
   }

  std::vector<std::string> nameStreams = hltConfig_.streamNames();

  const TriggerObjectCollection objects = aodTriggerEvent->getObjects();

  bool streamAfound = false;
  int i = 0;
  


  for (vector<string>::iterator streamName = nameStreams.begin(); 
       streamName != nameStreams.end(); ++streamName) {
    
    if  (hltConfig_.streamName(i) == "A") {
      if (debugPrint) std::cout << " Stream A not found " << std::endl;
      streamAfound = true;
    }
    else
      if (debugPrint) std::cout << " Stream A found " << std::endl;
    i++;
  }

   if (streamAfound) {
     vector<string> datasetNames =  hltConfig_.streamContent("A");
     // Loop over PDs
     for (unsigned int iPD = 0; iPD < datasetNames.size(); iPD++) { 
       
       //     if (datasetNames[iPD] != "SingleMu" && datasetNames[iPD] != "SingleElectron" && datasetNames[iPD] != "Jet") continue;  
       
       // unsigned int keyTracker[1000]; // Array to eliminate double counts by tracking what Keys have already been fired
       // for(unsigned int irreproduceableIterator = 0; irreproduceableIterator < 1000; irreproduceableIterator++) {
       // 	 keyTracker[irreproduceableIterator] = 1001;
       // }
       // Loop over Paths in each PD
       bool first_count = true;
       for (unsigned int iPath = 0; iPath < PDsVectorPathsVector[iPD].size(); iPath++) { 
	 
	 std::string pathName = PDsVectorPathsVector[iPD][iPath];
	 
	 if (debugPrint) std::cout << "Looking at path " << pathName << std::endl;
	 
	 unsigned int index = hltConfig_.triggerIndex(pathName);
	 
	 if (debugPrint) std::cout << "Index = " << index << " triggerResults->size() = " << triggerResults->size() << std::endl;
	 

	 //fill the histos with empty weights......
	 std::string label = datasetNames[iPD];
	 std:: string fullPathToCPP = "HLT/GeneralHLTOffline/"+label+"/cppath_"+label;
	 MonitorElement * ME_mini_cppath = dbe->get(fullPathToCPP);/////charlie
	 TH1F * hist_mini_cppath = ME_mini_cppath->getTH1F();//charlie

	 TAxis * axis = hist_mini_cppath->GetXaxis();
	 int bin_num = axis->FindBin(pathName.c_str());
	 int bn = bin_num - 1;
	 hist_mini_cppath->Fill(bn,0);//charlieeeee
	 hist_mini_cppath->SetEntries(hist_mini_cppath->Integral());


	 


	 if (index < triggerResults->size()) {
	   if(triggerResults->accept(index)) {
	     	     
	     cppath->Fill(index,1);
	     	     
	     if (debugPrint) std::cout << "Check Event " <<  iEvent.id() << " Run " << iEvent.id().run() 
				       << " fired path " << pathName << std::endl;
	     
	     // look up module labels for this path
	     
	     vector<std::string> modulesThisPath = hltConfig_.moduleLabels(pathName);
	     
	     if (debugPrint) std::cout << "Looping over module labels " << std::endl;
	     
	     // Loop backward through module names
	     for ( int iModule = (modulesThisPath.size()-1); iModule >= 0; iModule--) {
	       
	       if (debugPrint) std::cout << "Module name is " << modulesThisPath[iModule] << std::endl;
	       
	       // check to see if you have savetags information
	       if (hltConfig_.saveTags(modulesThisPath[iModule])) {
		 
		 if (debugPrint) std::cout << "For path " << pathName << " this module " << modulesThisPath[iModule] <<" is a saveTags module of type " << hltConfig_.moduleType(modulesThisPath[iModule]) << std::endl;
		 
		 if (hltConfig_.moduleType(modulesThisPath[iModule]) == "HLTLevel1GTSeed") break;
		 
		 InputTag moduleWhoseResultsWeWant(modulesThisPath[iModule], "", hltTag);
		 
		 unsigned int indexOfModuleInAodTriggerEvent = aodTriggerEvent->filterIndex(moduleWhoseResultsWeWant);
		 
		 if ( indexOfModuleInAodTriggerEvent < aodTriggerEvent->sizeFilters() ) {
		   const Keys &keys = aodTriggerEvent->filterKeys( indexOfModuleInAodTriggerEvent );
		   if (debugPrint) std::cout << "Got Keys for index " << indexOfModuleInAodTriggerEvent <<", size of keys is " << keys.size() << std::endl;
		   if (keys.size()>=1000) edm::LogWarning("GeneralHLTOffline") << "WARNING!! size of keys is " << keys.size() 
									       << " for path " << pathName << " and module " << modulesThisPath[iModule]<< std::endl;

		   // There can be > 100 keys (3-vectors) for some modules with no ID filled
		   // the first one has the highest value for single-object triggers
		   // for multi-object triggers, seems reasonable to use the first one as well
		   //  So loop here has been commented out
		   //		   for ( size_t iKey = 0; iKey < keys.size(); iKey++ ) {
		   
		   if (keys.size() > 0) {
		     TriggerObject foundObject = objects[keys[0]];

		     
		     //		     if(keyTracker[iKey] != iKey) first_count = true;
		     
		     if (debugPrint || outputPrint) std::cout << "This object has id (pt, eta, phi) = "
							      << " " << foundObject.id() << " " 
							      << std::setw(10) << foundObject.pt()
							      << ", " << std::setw(10) << foundObject.eta() 
							      << ", " << std::setw(10) << foundObject.phi()
							      << "    for path = " << std::setw(20) << pathName
							      << " module " << std::setw(40) << modulesThisPath[iModule]
							      << std::endl;

		     if (debugPrint) std::cout << "CHECK RUN " << iEvent.id().run() << " " << iEvent.id() << " " << pathName << " " 
					       << modulesThisPath[iModule] << " " << datasetNames[iPD] << " "  
					       << hltConfig_.moduleType(modulesThisPath[iModule]) << " " 
					       << keys.size() << " " 
					       << std::setprecision(4) << foundObject.pt() << " " 
					       << foundObject.eta() << " " 
					       << foundObject.phi() << std::endl;

		     // first_count is to make sure that the top-level histograms of each dataset 
		     // don't get filled more than once
		     fillHltMatrix(datasetNames[iPD],pathName,foundObject.eta(),foundObject.phi(),first_count);
		     first_count = false;
		     
		     //		     keyTracker[iKey] = iKey;
		     
		     //   }// end for each key               
		   } // at least one key
		 }// end if filter in aodTriggerEvent
		 
		 // OK, we found the last module. No need to look at the others.
		 // get out of the loop
		 break;
	       }// end if saveTags
	     }//end Loop backward through module names   
	   }// end if(triggerResults->accept(index))

	   // else:
	   
	   
	   

	 }// end if (index < triggerResults->size())
       }// end Loop over Paths in each PD
     }//end Loop over PDs
   }
   
   

}








// ------------ method called once each job just before starting event loop  ------------
void 
GeneralHLTOffline::beginJob()
{using namespace edm;
 using std::string;


  
  if (debugPrint) std::cout << "Inside begin job" << std::endl; 

  dbe = Service<DQMStore>().operator->();
 
 if (dbe) {

    dbe->setCurrentFolder(plotDirectoryName);
    
  }

    
}

// ------------ method called once each job just after ending the event loop  ------------
void 
GeneralHLTOffline::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
void 
GeneralHLTOffline::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{

  if (debugPrint) std::cout << "Inside beginRun" << std::endl;
  
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, hltTag, changed)) {
    if(debugPrint)
      std::cout << "HLT config with process name " 
                << hltTag << " successfully extracted" << std::endl;
  } else {
    if (debugPrint) std::cout << "Warning, didn't find process HLT" << std::endl;
  }

  if (debugPrint) std::cout << " HLTConfig processName " << hltConfig_.processName()
			    << " tableName " << hltConfig_.tableName() 
			    << " size " << hltConfig_.size() << std::endl;

  //////////// Book a simple ME

  dbe->setCurrentFolder("HLT/GeneralHLTOffline/");
  cppath = dbe->book1D("cppath","Counts/Path",hltConfig_.size(),0,hltConfig_.size());  

  std::vector<std::string> nameStreams = hltConfig_.streamNames();

  bool streamAfound = false;
  int i = 0;
  for (vector<string>::iterator streamName = nameStreams.begin(); 
       streamName != nameStreams.end(); ++streamName) {
    
    if  (hltConfig_.streamName(i) == "A") {
      if (debugPrint) std::cout << " Stream A found " << std::endl;
      streamAfound = true;
    }
    else
      if (debugPrint) std::cout << " Stream A not found " << std::endl;
    i++;
  }
 

 
  if (streamAfound) {
    vector<string> datasetNames =  hltConfig_.streamContent("A");
  
    if (debugPrint) std::cout << "Number of Stream A datasets " << datasetNames.size() << std::endl;
    
    for (unsigned int i=0;i<datasetNames.size();i++) {
      
      if (debugPrint) std::cout << "This is dataset " << datasetNames[i] <<std::endl;
      
      vector<string> datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
      
      if (debugPrint) std::cout << "datasetPaths.size() = " << datasetPaths.size() << std::endl;
      
      // Should be fine here 
      if (debugPrint) {
	for (unsigned int iPath = 0;  
	     iPath < datasetPaths.size(); iPath++) {
	  std::cout << "Before setupHltMatrix -  MET dataset " << datasetPaths[iPath] << std::endl;
	}
      }
      
      // Check if dataset has been added - if not add it
      // need to loop through AddedDatasets and compare
      bool foundDataset = false;
      int datasetNum = -1;
      for (unsigned int d = 0; d < AddedDatasets.size(); d++) {
	if (AddedDatasets[d].compare(datasetNames[i]) == 0 ){
	  foundDataset = true;
	  datasetNum = d;
	  if (debugPrint) std::cout << "Dataset " << datasetNames[i] << " found in AddedDatasets at position " << d << std::endl;
	  break;
	}
      }

      if (!foundDataset) {
	if (debugPrint) std::cout << " Fill trigger paths for dataset " << datasetNames[i] << std::endl;
	PDsVectorPathsVector.push_back(datasetPaths);
	// store dataset pathname 
	AddedDatasets.push_back(datasetNames[i]);
      }
      // This trigger path has already been added - 
      //   this implies that this is a new run
      //   What we want to do is check if there is a new trigger that was not in the original dataset
      //    For a given dataset, loop over the stored list of triggers, and compare to the current list of triggers
      //     If any of the triggers are missing, add them to the end of the appropriate dataset
      else  {
	if (debugPrint) std::cout << " Additional runs : Check for additional trigger paths per dataset " << std::endl;
	  // Loop over correct path of PDsVectorPathsVector
	  bool found = false;

	  // Loop over triggers in the path
	  for (unsigned int iTrig = 0; iTrig < datasetPaths.size(); iTrig++) {
	    if (debugPrint) std::cout << "Looping over trigger list in dataset " <<  iTrig <<  "  " << datasetPaths[iTrig] << endl; 
	      found = false;
	    // Loop over triggers already on the list
	    for (unsigned int od = 0; od < PDsVectorPathsVector[datasetNum].size(); od++) { 
	      if (debugPrint) std::cout << "Looping over existing trigger list " <<  od <<  "  " << PDsVectorPathsVector[datasetNum][od] << endl; 
	      // Compare, see if match is found
	      if (removeVersions(datasetPaths[iTrig]).compare(removeVersions(PDsVectorPathsVector[datasetNum][od])) == 0) {
		found = true;
		if (debugPrint) std::cout << " FOUND " << datasetPaths[iTrig] << std::endl;
		break;
	      }
	    }
	  // If match is not found, add trigger to correct path of PDsVectorPathsVector
	  if (!found)
	    PDsVectorPathsVector[datasetNum].push_back(datasetPaths[iTrig]);
	  if (debugPrint) std::cout << datasetPaths[iTrig] << "  NOT FOUND - so we added it to the correct dataset " << datasetNames[i] << std::endl;
	  }
      }      
      
      // Let's check this whole big structure
      if (debugPrint) {
	for (unsigned int is = 0; is < PDsVectorPathsVector.size(); is++) { 
	  std::cout << "   PDsVectorPathsVector[" << is << "] is " << PDsVectorPathsVector[is].size() << std::endl;
	  for (unsigned int ip = 0; ip < PDsVectorPathsVector[is].size(); ip++) { 
	    std::cout << "    trigger " << ip << " path " << PDsVectorPathsVector[is][ip] << std::endl;
	  }
	}
      }

      if (debugPrint) std::cout <<"Found PD: " << datasetNames[i]  << std::endl;     
      setupHltMatrix(datasetNames[i],i);   

    }// end of loop over dataset names
  } // if stream A found

}
// end of beginRun



// ------------ method called when ending the processing of a run  ------------
void GeneralHLTOffline::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (debugPrint) std::cout << " endRun called " << std::endl; 
}


void GeneralHLTOffline::setupHltMatrix(std::string label, int iPD) {

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
std::string HLTMenu;

PD_Folder = TString("HLT/GeneralHLTOffline");
if (label != "SingleMu" && label != "SingleElectron" && label != "Jet")  PD_Folder = TString("HLT/GeneralHLTOffline/"+label); 

dbe->setCurrentFolder(PD_Folder.c_str());
HLTMenu = hltConfig_.tableName();
dbe->bookString("hltMenuName",HLTMenu.c_str());

h_name = "HLT_"+label+"_EtaVsPhi";
h_title = "HLT_"+label+"_EtaVsPhi";
h_name_1dEta = "HLT_"+label+"_1dEta";
h_name_1dPhi = "HLT_"+label+"_1dPhi";
h_title_1dEta = label+" Occupancy Vs Eta";
h_title_1dPhi = label+" Occupancy Vs Phi";
//Int_t numBinsEta = 12;
//Int_t numBinsPhi = 8;
Int_t numBinsEta = 30;
Int_t numBinsPhi = 34;
Int_t numBinsEtaFine = 60;
Int_t numBinsPhiFine = 66;
Double_t EtaMax = 2.610;
Double_t PhiMax = 17.0*TMath::Pi()/16.0;
Double_t PhiMaxFine = 33.0*TMath::Pi()/32.0;

//Double_t eta_bins[] = {-2.610,-2.349,-2.088,-1.827,-1.740,-1.653,-1.566,-1.479,-1.392,-1.305,-1.044,-0.783,-0.522,-0.261,0,0.261,0.522,0.783,1.044,1.305,1.392,1.479,1.566,1.653,1.740,1.827,2.088,2.349,2.610}; //Has narrower bins in Barrel/Endcap border region
//Double_t eta_bins[] = {-2.610,-2.175,-1.740,-1.305,-0.870,-0.435,0,0.435,0.870,1.305,1.740,2.175,2.610};
//Double_t phi_bins[] = {-TMath::Pi(),-3*TMath::Pi()/4,-TMath::Pi()/2,-TMath::Pi()/4,0,TMath::Pi()/4,TMath::Pi()/2,3*TMath::Pi()/4,TMath::Pi()};

 TH2F * hist_EtaVsPhi = new TH2F(h_name.c_str(),h_title.c_str(),numBinsEta,-EtaMax,EtaMax,numBinsPhi,-PhiMax,PhiMax);
 TH1F * hist_1dEta = new TH1F(h_name_1dEta.c_str(),h_title_1dEta.c_str(),numBinsEtaFine,-EtaMax,EtaMax);
 TH1F * hist_1dPhi = new TH1F(h_name_1dPhi.c_str(),h_title_1dPhi.c_str(),numBinsPhiFine,-PhiMaxFine,PhiMaxFine);


 hist_EtaVsPhi->SetMinimum(0);
 hist_1dEta->SetMinimum(0);
 hist_1dPhi->SetMinimum(0);

 // Do not comment out these pointers since it is the booking that is the work here. 
 // MonitorElement * ME_EtaVsPhi = dbe->book2D(h_name.c_str(),hist_EtaVsPhi);
 dbe->book2D(h_name.c_str(),hist_EtaVsPhi);
 // MonitorElement * ME_1dEta = dbe->book1D(h_name_1dEta.c_str(),hist_1dEta);
 if (label != "MET" && label != "HT") 
   dbe->book1D(h_name_1dEta.c_str(),hist_1dEta);
 // MonitorElement * ME_1dPhi = dbe->book1D(h_name_1dPhi.c_str(),hist_1dPhi);
 if (label != "HT") {
   dbe->book1D(h_name_1dPhi.c_str(),hist_1dPhi);
 }

 ////charlieeeeeee
 std::string folderz;
 folderz = TString("HLT/GeneralHLTOffline/"+label);//make it the top level directory, that is on the same dir level as paths
 dbe->setCurrentFolder(folderz.c_str());
 //book charlie's new histogramzzzzzz
 std::string dnamez = "cppath_"+label;
 std::string dtitlez = "cppath_"+label;
 int sizez = PDsVectorPathsVector[iPD].size();
 TH1F * hist_mini_cppath = NULL;
 MonitorElement * hist_mini_cppath_me = dbe->book1D(dnamez.c_str(),
                                                    dtitlez.c_str(),
                                                    sizez,
                                                    0,
                                                    sizez);
 if (hist_mini_cppath_me)
   hist_mini_cppath = hist_mini_cppath_me->getTH1F();
 
 unsigned int jPath;
  for (unsigned int iPath = 0; iPath < PDsVectorPathsVector[iPD].size(); iPath++) { 
    pathName = removeVersions(PDsVectorPathsVector[iPD][iPath]);
    h_name_1dEtaPath = "HLT_"+pathName+"_1dEta";
    h_name_1dPhiPath = "HLT_"+pathName+"_1dPhi";
    h_title_1dEtaPath = pathName+" Occupancy Vs Eta";
    h_title_1dPhiPath = pathName+"Occupancy Vs Phi";
    jPath=iPath+1;

    if (hist_mini_cppath) {
      TAxis * axis = hist_mini_cppath->GetXaxis();
      axis->SetBinLabel(jPath, pathName.c_str());
    }

    Path_Folder = TString("HLT/GeneralHLTOffline/"+label+"/Paths");
    dbe->setCurrentFolder(Path_Folder.c_str());

    // Do not comment out these pointers since it is the booking that is the work here. 
    //     MonitorElement * ME_1dEta = dbe->book1D(h_name_1dEtaPath.c_str(),h_title_1dEtaPath.c_str(),numBinsEtaFine,-EtaMax,EtaMax);
    dbe->book1D(h_name_1dEtaPath.c_str(), h_title_1dEtaPath.c_str(), numBinsEtaFine, -EtaMax, EtaMax);
     //     MonitorElement * ME_1dPhi = dbe->book1D(h_name_1dPhiPath.c_str(),h_title_1dPhiPath.c_str(),numBinsPhiFine,-PhiMaxFine,PhiMaxFine);
    dbe->book1D(h_name_1dPhiPath.c_str(), h_title_1dPhiPath.c_str(), numBinsPhiFine, -PhiMaxFine, PhiMaxFine);

    if (debugPrint) std::cout << "book1D for " << pathName << std::endl;
  }

   
 if (debugPrint) std::cout << "Success setupHltMatrix( " << label << " , " << iPD << " )" << std::cout;
} //End setupHltMatrix

string GeneralHLTOffline::removeVersions(std::string histVersion) {
  for (int ii = 100; ii >= 0; ii--) {
    string ver = "_v";
    string version ="";
    stringstream ss;
    ss << ver << ii;
    ss >> version;
    
    size_t pos = histVersion.find(version);
    if (pos != std::string::npos)
      histVersion.erase(pos,version.size());
  }
  return histVersion;
}


void GeneralHLTOffline::fillHltMatrix(std::string label, std::string path,double Eta, double Phi, bool first_count) {

  if (debugPrint) std::cout << "Inside fillHltMatrix( " << label << " , " << path << " ) " << std::endl;

  std::string fullPathToME;
  std::string fullPathToME1dEta;
  std::string fullPathToME1dPhi;
  std::string fullPathToME1dEtaPath;
  std::string fullPathToME1dPhiPath;
  std::string fullPathToCPP;


 fullPathToME = "HLT/GeneralHLTOffline/HLT_"+label+"_EtaVsPhi"; 
 fullPathToME1dEta = "HLT/GeneralHLTOffline/HLT_"+label+"_1dEta";
 fullPathToME1dPhi = "HLT/GeneralHLTOffline/HLT_"+label+"_1dPhi";
 fullPathToCPP = "HLT/GeneralHLTOffline/"+label+"/cppath_"+label;
 

if (label != "SingleMu" && label != "SingleElectron" && label != "Jet") {
 fullPathToME = "HLT/GeneralHLTOffline/"+label+"/HLT_"+label+"_EtaVsPhi"; 
 fullPathToME1dEta = "HLT/GeneralHLTOffline/"+label+"/HLT_"+label+"_1dEta";
 fullPathToME1dPhi = "HLT/GeneralHLTOffline/"+label+"/HLT_"+label+"_1dPhi";
 }
 
 fullPathToME1dEtaPath = "HLT/GeneralHLTOffline/"+label+"/Paths/HLT_"+removeVersions(path)+"_1dEta";
 fullPathToME1dPhiPath = "HLT/GeneralHLTOffline/"+label+"/Paths/HLT_"+removeVersions(path)+"_1dPhi";

  // MonitorElement * ME_2d = dbe->get(fullPathToME);
  // MonitorElement * ME_1dEta = dbe->get(fullPathToME1dEta);
  // MonitorElement * ME_1dPhi = dbe->get(fullPathToME1dPhi);  
  // MonitorElement * ME_1dEtaPath = dbe->get(fullPathToME1dEtaPath);
  // MonitorElement * ME_1dPhiPath = dbe->get(fullPathToME1dPhiPath);
  MonitorElement * ME_mini_cppath = dbe->get(fullPathToCPP);
  
  // TH2F * hist_2d = ME_2d->getTH2F();
  // TH1F * hist_1dEta = ME_1dEta->getTH1F();
  // TH1F * hist_1dPhi = ME_1dPhi->getTH1F();
  // TH1F * hist_1dEtaPath = ME_1dEtaPath->getTH1F();
  // TH1F * hist_1dPhiPath = ME_1dPhiPath->getTH1F();
  TH1F * hist_mini_cppath = ME_mini_cppath->getTH1F();

  //int i=2;
  //if (Eta>1.305 && Eta<1.872) i=0;
  //if (Eta<-1.305 && Eta>-1.872) i=0;
  //for (int ii=i; ii<3; ++ii) hist_2d->Fill(Eta,Phi); //Scales narrow bins in Barrel/Endcap border region

  // fill top-level histograms
  if(first_count) {
    if (debugPrint)
      std::cout << " label " << label << " fullPathToME1dPhi " << fullPathToME1dPhi << " path "  << path << " Phi " << Phi << " Eta " << Eta << std::endl;

    if (label != "MET" && label != "HT") {
      MonitorElement * ME_1dEta = dbe->get(fullPathToME1dEta);
      TH1F * hist_1dEta = ME_1dEta->getTH1F();
      hist_1dEta->Fill(Eta);
    }
    if (label != "HT") {
      MonitorElement * ME_1dPhi = dbe->get(fullPathToME1dPhi);  
      TH1F * hist_1dPhi = ME_1dPhi->getTH1F();
      hist_1dPhi->Fill(Phi); 
      if (debugPrint) std::cout << "  **FILLED** label " << label << " fullPathToME1dPhi " << fullPathToME1dPhi << " path "  << path << " Phi " << Phi << " Eta " << Eta << std::endl;
    }
    if (label != "MET" && label != "HT") {
      MonitorElement * ME_2d = dbe->get(fullPathToME);
      TH2F * hist_2d = ME_2d->getTH2F();
      hist_2d->Fill(Eta,Phi); 
    }
  }  // end fill top-level histograms


  if (label != "MET" && label != "HT") {
    MonitorElement * ME_1dEtaPath = dbe->get(fullPathToME1dEtaPath);
    TH1F * hist_1dEtaPath = ME_1dEtaPath->getTH1F();
    hist_1dEtaPath->Fill(Eta); 
  }
  if (label != "HT") {
    MonitorElement * ME_1dPhiPath = dbe->get(fullPathToME1dPhiPath);
    TH1F * hist_1dPhiPath = ME_1dPhiPath->getTH1F();
    hist_1dPhiPath->Fill(Phi);
  }
  
  if (debugPrint)
    if (label == "MET") 
      std::cout << " MET Eta is " << Eta << std::endl;
  
  TAxis * axis = hist_mini_cppath->GetXaxis();
  int bin_num = axis->FindBin(path.c_str());
  int bn = bin_num - 1;
  hist_mini_cppath->Fill(bn,1);

    
 if (debugPrint) std::cout << "hist->Fill" << std::endl;

} //End fillHltMatrix

// ------------ method called when starting to processes a luminosity block  ------------
void GeneralHLTOffline::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method called when ending the processing of a luminosity block  ------------
void 
GeneralHLTOffline::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// GeneralHLTOffline::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in for online (not offline ?)
DEFINE_FWK_MODULE(GeneralHLTOffline);

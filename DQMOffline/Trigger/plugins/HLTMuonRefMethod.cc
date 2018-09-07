#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/Framework/interface/Event.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <set>
#include <string>
#include <vector>
#include <TPRegexp.h>
#include <cmath>
#include <climits>
#include <boost/tokenizer.hpp>
#include <boost/regex.hpp>


#include <TH1.h>
#include <TEfficiency.h>




using namespace edm;
using namespace std;



class HLTMuonRefMethod : public DQMEDHarvester {

public:
  explicit HLTMuonRefMethod(const edm::ParameterSet& set);
  ~HLTMuonRefMethod() override = default;;

  void beginJob() override;
  void beginRun(const edm::Run&, const edm::EventSetup&) override ;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override ;
  

private:
  // DQMStore* theDQM;

  std::vector<std::string> subDirs_;
  std::vector<std::string> hltTriggers_;
  std::vector<std::string> efficiency_;
  std::vector<std::string> refEff_;
  std::string              refTriggers_;

  std::string outputFileName_;

  void findAllSubdirectories (DQMStore::IBooker& ibooker,
			      DQMStore::IGetter& igetter,
			      const std::string& dir,
			      std::set<std::string> * myList,
			      const TString& pattern);
  

};


HLTMuonRefMethod::HLTMuonRefMethod(const edm::ParameterSet& pset)
{
  using VPSet = std::vector<edm::ParameterSet>;
  using vstring = std::vector<std::string>;
  using elsc = boost::escaped_list_separator<char>;

  subDirs_     = pset.getUntrackedParameter<vstring>("subDirs");
  hltTriggers_ = pset.getUntrackedParameter<vstring>("hltTriggers");
  refTriggers_ = pset.getUntrackedParameter<string> ("refTriggers");
  efficiency_  = pset.getUntrackedParameter<vstring>("efficiency" );
  refEff_      = pset.getUntrackedParameter<vstring>("refEff");
}

void 
HLTMuonRefMethod::beginJob()
{

}


void HLTMuonRefMethod::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter)
{
  using vstring = std::vector<std::string>;
  boost::regex metacharacters{"[\\^\\$\\.\\*\\+\\?\\|\\(\\)\\{\\}\\[\\]]"};
  boost::smatch what;

  // theDQM = 0;
  // theDQM = Service<DQMStore>().operator->();
  
  ibooker.cd();
  set<string> subDirSet;

  for(auto iSubDir = subDirs_.begin();
      iSubDir != subDirs_.end(); ++iSubDir) {
    string subDir = *iSubDir;

    if ( subDir[subDir.size()-1] == '/' ) subDir.erase(subDir.size()-1);

    if ( boost::regex_search(subDir, what, metacharacters)) {
      const string::size_type shiftPos = subDir.rfind('/');
      const string searchPath = subDir.substr(0, shiftPos);
      const string pattern    = subDir.substr(shiftPos + 1, subDir.length());
      //std::cout << "\n\n\n\nLooking for all subdirs of " << subDir << std::endl;
   
      findAllSubdirectories (ibooker, igetter, searchPath, &subDirSet, pattern);

    }
    else {
      subDirSet.insert(subDir);
    }
  }
  
  for(auto const & subDir : subDirSet) {
    for (unsigned int iEff = 0; iEff != efficiency_.size(); ++iEff){
      string eff = efficiency_[iEff];

      // Getting reference trigger efficiency
      MonitorElement* refEff = igetter.get(subDir + "/" + refTriggers_ + "/" + refEff_[iEff]);

      if (!refEff) continue;

      
      // looping over all reference triggers 
      for (auto iTrigger = hltTriggers_.begin();
    	   iTrigger != hltTriggers_.end(); ++iTrigger){
    	string trig = *iTrigger;
	MonitorElement* trigEff = igetter.get(subDir + "/" + trig + "/" + eff );
	if (!trigEff) continue;
	TH1* hRef  = refEff  -> getTH1();
	TH1* hTrig = trigEff -> getTH1();
	TH1* hEff = (TH1*) hTrig->Clone( ("eff_" + eff + "_ref").c_str() );
	hEff->SetTitle("Efficiency obtained with reference method");
	TClass* myHistClass = hTrig->IsA();
	TString histClassName = myHistClass->GetName();
	
	if (histClassName == "TH1F"){
	  for (int bin = 0; bin < hEff->GetNbinsX(); ++bin){
	    hEff->SetBinContent(bin+1, hEff->GetBinContent(bin+1)*hRef->GetBinContent(bin+1));
	    hEff->SetBinError  (bin+1, hEff->GetBinContent(bin+1)*hRef->GetBinError(bin+1)+hEff->GetBinError(bin+1)*hRef->GetBinContent(bin+1));
	  }
	  ibooker.cd(subDir + "/" + trig);
	  ibooker.book1D(  hEff->GetName(), (TH1F*) hEff);
	
	}
	else if (histClassName == "TH2F"){
	  for (int i = 0; i < hRef->GetXaxis()->GetNbins(); ++i){
	    for (int j = 0; j < hRef->GetYaxis()->GetNbins(); ++j){
	      int bin = hEff->GetBin(i+1,j+1);
	      hEff -> SetBinContent( bin, hRef->GetBinContent(i+1,j+1) * hTrig->GetBinContent(i+1) );
	    }
	  }
	  ibooker.cd(subDir + "/" + trig);
	  ibooker.book2D(  hEff->GetName(), (TH2F*) hEff);
	}
	else{ 
	  LogInfo ("HLTMuonRefMethod") << "Histo class type << " << histClassName << " not implemented";
	}

	delete hEff;
	
      }
    }
  }
}


void 
HLTMuonRefMethod::beginRun(const edm::Run& run, const edm::EventSetup& c)
{


}



void
HLTMuonRefMethod::findAllSubdirectories (DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const std::string& dir, std::set<std::string> * myList,
					 const TString& _pattern = TString("")) {
  TString pattern = _pattern;
  TPRegexp nonPerlWildcard("\\w\\*|^\\*");

  if (!igetter.dirExists(dir)) {
    LogError("DQMGenericClient") << " DQMGenericClient::findAllSubdirectories ==> Missing folder " << dir << " !!!"; 
    return;
  }
  if (pattern != "") {
    if (pattern.Contains(nonPerlWildcard)) pattern.ReplaceAll("*",".*");
    TPRegexp regexp(pattern);
    ibooker.cd(dir);
    vector <string> foundDirs = igetter.getSubdirs();
    for(auto iDir = foundDirs.begin();
	iDir != foundDirs.end(); ++iDir) {
      TString dirName = iDir->substr(iDir->rfind('/') + 1, iDir->length());
      if (dirName.Contains(regexp))
	findAllSubdirectories (ibooker, igetter, *iDir, myList);
    }
  }
  //std::cout << "Looking for directory " << dir ;
  else if (igetter.dirExists(dir)){
    //std::cout << "... it exists! Inserting it into the list ";
    myList->insert(dir);
    //std::cout << "... now list has size " << myList->size() << std::endl;
    ibooker.cd(dir);
    findAllSubdirectories (ibooker, igetter, dir, myList, "*");
  } else {
    //std::cout << "... DOES NOT EXIST!!! Skip bogus dir" << std::endl;
 
    LogInfo ("DQMGenericClient") << "Trying to find sub-directories of " << dir
				 << " failed because " << dir  << " does not exist";
                               
  }
  return;
}


DEFINE_FWK_MODULE(HLTMuonRefMethod);

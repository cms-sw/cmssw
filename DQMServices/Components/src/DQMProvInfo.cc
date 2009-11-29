/*
 * \file DQMEventInfo.cc
 * \author M. Zanetti - CERN PH
 * Last Update:
 * $Date: 2009/09/06 11:10:15 $
 * $Revision: 1.24 $
 * $Author: ameyer $
 *
 */

#include "DQMProvInfo.h"
#include <TSystem.h>

// Framework


#include <stdio.h>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;

DQMProvInfo::DQMProvInfo(const ParameterSet& ps){
  
  parameters_ = ps;

  dbe_ = edm::Service<DQMStore>().operator->();

  string provinfofolder = parameters_.getUntrackedParameter<string>("provInfoFolder", "ProvInfo") ;
  string subsystemname = parameters_.getUntrackedParameter<string>("subSystemFolder", "YourSubsystem") ;
  currentfolder_ = subsystemname + "/" +  provinfofolder ;

}

DQMProvInfo::~DQMProvInfo(){
}

void DQMProvInfo::beginRun(const edm::Run& r, const edm::EventSetup &c ) {

  makeProvInfo();
      
} 

void DQMProvInfo::analyze(const Event& e, const EventSetup& c){
 
//  makeProvInfo();
  return;
}

// run showtag command line
std::string 
DQMProvInfo::getShowTags(void)
{
   TString out;
   FILE *pipe = gSystem->OpenPipe("showtags u -t", "r");

   TString line;
   while (line.Gets(pipe,true)) {
     if (line.Contains("Test Release")) continue;
     if (line.Contains("Base Release")) continue;
     if (line.Contains("Test release")) continue;
     if (line.Contains("--- Tag ---")) continue;
     if (line.Contains(" ")) line.Replace(line.First(" "),1,":");
     line.ReplaceAll(" ","");
     out = out + line + ";";
     if (line.Contains("-------------------")) break;
     if (out.Length()>2000) break;
   }
   out.ReplaceAll("--","");
   out.ReplaceAll(";-",";");
   out.ReplaceAll(";;",";");
   out.ReplaceAll("\n","");

   Int_t r = gSystem->ClosePipe(pipe);
   if (r) {
     gSystem->Error("ShowTags","problem running command showtags -u -t");
   }

   std::string str(out);
   if (str.length()>2000) str.resize(2000);

   std::string safestr =
     "/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-;:";
   size_t found=str.find_first_not_of(safestr);
   if (found!=std::string::npos)
   {
     std::cout << "DQMFileSaver::ShowTags: Illegal character found: " 
               << str[found] 
               << " at position " 
               << int(found) << std::endl;
     return "notags";
   }   
   return str;
}


void 
DQMProvInfo::makeProvInfo()
{
    dbe_->cd() ;
    dbe_->setCurrentFolder(currentfolder_);

    // if (dbe_->get("ProvInfo/CMSSW")) return ;
    
    versCMSSW_     = dbe_->bookString("CMSSW",edm::getReleaseVersion().c_str() );
    hostName_      = dbe_->bookString("hostName",gSystem->HostName());
    workingDir_    = dbe_->bookString("workingDir",gSystem->pwd());
    processId_     = dbe_->bookInt("processID"); processId_->Fill(gSystem->GetPid());

    //versDataset_   = dbe_->bookString("Dataset",workflow_);
    versGlobaltag_ = dbe_->bookString("Globaltag","global tag"); // FIXME
    versTaglist_   = dbe_->bookString("Taglist",getShowTags()); 

    isComplete_ = dbe_->bookInt("runIsComplete"); 
    //isComplete_->Fill((runIsComplete_?1:0));
    fileVersion_ = dbe_->bookInt("fileVersion");
    //fileVersion_->Fill(version_);
    
    return ;
}

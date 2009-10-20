/*
 *  Class:DQMGenericClient 
 *
 *
 *  $Date: 2009/10/06 11:16:53 $
 *  $Revision: 1.12 $
 * 
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "DQMServices/ClientConfig/interface/DQMGenericClient.h"

#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TH1F.h>
#include <cmath>
#include <TClass.h>
#include <TString.h>


using namespace std;
using namespace edm;

typedef MonitorElement ME;
typedef vector<string> vstring;

TPRegexp metacharacters("[\\^\\$\\.\\*\\+\\?\\|\\(\\)\\{\\}\\[\\]]");
TPRegexp nonPerlWildcard("\\w\\*|^\\*");

DQMGenericClient::DQMGenericClient(const ParameterSet& pset)
{

  vstring dummy;

  verbose_ = pset.getUntrackedParameter<unsigned int>("verbose", 0);

  effCmds_ = pset.getParameter<vstring>("efficiency");
  resCmds_ = pset.getParameter<vstring>("resolution");
  normCmds_ = pset.getUntrackedParameter<vstring>("normalization", dummy);
  cdCmds_ = pset.getUntrackedParameter<vstring>("cumulativeDists", dummy);

  resLimitedFit_ = pset.getUntrackedParameter<bool>("resolutionLimitedFit",false);

  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName", "");
  subDirs_ = pset.getUntrackedParameter<vstring>("subDirs");

  isWildcardUsed_ = false;
}

void DQMGenericClient::endRun(const edm::Run& r, const edm::EventSetup& c) {

  // Update 2009-09-23
  // Migrated all code from endJob to this function
  // endJob is not necessarily called in the proper sequence
  // and does not necessarily book histograms produced in
  // that step.
  // It more robust to do the histogram manipulation in
  // this endRun function


  
  theDQM = 0;
  theDQM = Service<DQMStore>().operator->();

  if ( ! theDQM ) {
    LogInfo("DQMGenericClient") << "Cannot create DQMStore instance\n";
    return;
  }

  // Process wildcard in the sub-directory
  set<string> subDirSet;

  for(vstring::const_iterator iSubDir = subDirs_.begin();
      iSubDir != subDirs_.end(); ++iSubDir) {
    string subDir = *iSubDir;

    if ( subDir[subDir.size()-1] == '/' ) subDir.erase(subDir.size()-1);

    if ( TString(subDir).Contains(metacharacters) ) {
      isWildcardUsed_ = true;

      const string::size_type shiftPos = subDir.rfind('/');
      const string searchPath = subDir.substr(0, shiftPos);
      const string pattern    = subDir.substr(shiftPos + 1, subDir.length());
      //std::cout << "\n\n\n\nLooking for all subdirs of " << subDir << std::endl;
      
      findAllSubdirectories (searchPath, &subDirSet, pattern);

    }
    else {
      subDirSet.insert(subDir);
    }
  }

  for(set<string>::const_iterator iSubDir = subDirSet.begin();
      iSubDir != subDirSet.end(); ++iSubDir) {
    typedef boost::escaped_list_separator<char> elsc;

    const string& dirName = *iSubDir;

    for(vstring::const_iterator iCmd = effCmds_.begin();
        iCmd != effCmds_.end(); ++iCmd) {
      if ( iCmd->empty() ) continue;
      boost::tokenizer<elsc> tokens(*iCmd, elsc("\\", " \t", "\'"));

      vector<string> args;
      for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
          iToken != tokens.end(); ++iToken) {
        if ( iToken->empty() ) continue;
        args.push_back(*iToken);
      }

      if ( args.size() < 4 ) {
        LogInfo("DQMGenericClient") << "Wrong input to effCmds\n";
        continue;
      }

      if ( args.size() == 4 ) args.push_back("eff");

      computeEfficiency(dirName, args[0], args[1], args[2], args[3], args[4]);
    }

    for(vstring::const_iterator iCmd = resCmds_.begin();
        iCmd != resCmds_.end(); ++ iCmd) {
      if ( iCmd->empty() ) continue;
      boost::tokenizer<elsc> tokens(*iCmd, elsc("\\", " \t", "\'"));

      vector<string> args;
      for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
          iToken != tokens.end(); ++iToken) {
        if ( iToken->empty() ) continue;
        args.push_back(*iToken);
      }

      if ( args.size() != 3 ) {
        LogInfo("DQMGenericClient") << "Wrong input to resCmds\n";
        continue;
      }

      computeResolution(dirName, args[0], args[1], args[2]);
    }

    for(vstring::const_iterator iCmd = normCmds_.begin();
        iCmd != normCmds_.end(); ++iCmd) {
      if ( iCmd->empty() ) continue;
      boost::tokenizer<elsc> tokens(*iCmd, elsc("\\", "\t", "\'"));

      vector<string> args;
      for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
          iToken != tokens.end(); ++iToken) {
        if ( iToken->empty() ) continue;
        args.push_back(*iToken);
      }

      if ( args.size() != 1 ) {
        LogInfo("DQMGenericClient") << "Wrong input to normCmds\n";
        continue;
      }

      normalizeToEntries(dirName, args[0]);
    }

    for(vstring::const_iterator iCmd = cdCmds_.begin();
        iCmd != cdCmds_.end(); ++ iCmd) {
      if ( iCmd->empty() ) continue;
      boost::tokenizer<elsc> tokens(*iCmd, elsc("\\", " \t", "\'"));

      vector<string> args;
      for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
          iToken != tokens.end(); ++iToken) {
        if ( iToken->empty() ) continue;
        args.push_back(*iToken);
      }

      if ( args.size() != 1 ) {
        LogInfo("DQMGenericClient") << "Wrong input to cdCmds\n";
        continue;
      }

      makeCumulativeDist(dirName, args[0]);
    }
  }

  //if ( verbose_ > 0 ) theDQM->showDirStructure();

  if ( ! outputFileName_.empty() ) theDQM->save(outputFileName_);
  
}

void DQMGenericClient::endJob()
{

  // Update 2009-09-23
  // Migrated all code from here to endRun

  LogTrace ("DQMGenericClient") << "inside of DQMGenericClient::endJob()"
                                << endl;

}

void DQMGenericClient::computeEfficiency(const string& startDir, const string& efficMEName, const string& efficMETitle,
                                         const string& recoMEName, const string& simMEName, const std::string & type)
{
  if ( ! theDQM->dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeEfficiency() : "
                                  << "Cannot find sub-directory " << startDir << endl; 
    }
    return;
  }

  theDQM->cd();

  ME* simME  = theDQM->get(startDir+"/"+simMEName);
  ME* recoME = theDQM->get(startDir+"/"+recoMEName);

  if ( !simME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeEfficiency() : "
                                  << "No sim-ME '" << simMEName << "' found\n";
    }
    return;
  }

  if ( !recoME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeEfficiency() : " 
                                  << "No reco-ME '" << recoMEName << "' found\n";
    }
    return;
  }

  // Treat everything as the base class, TH1
  
  TH1* hSim  = simME ->getTH1();
  TH1* hReco = recoME->getTH1();
  
  if ( !hSim || !hReco ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeEfficiency() : "
                                  << "Cannot create TH1 from ME\n";
    }
    return;
  }

  string efficDir = startDir;
  string newEfficMEName = efficMEName;
  string::size_type shiftPos;
  if ( string::npos != (shiftPos = efficMEName.rfind('/')) ) {
    efficDir += "/"+efficMEName.substr(0, shiftPos);
    newEfficMEName.erase(0, shiftPos+1);
  }
  theDQM->setCurrentFolder(efficDir);

  TH1* efficHist = (TH1*)hSim->Clone(newEfficMEName.c_str());
  efficHist->SetTitle(efficMETitle.c_str());

  // Here is where you have trouble --- you need
  // to understand what type of hist you have.

  ME* efficME = 0;

  // Parse the class name
  // This works, but there might be a better way
  TClass * myHistClass = efficHist->IsA();
  TString histClassName = myHistClass->GetName();
  
  if (histClassName == "TH1F"){
    efficME = theDQM->book1D(newEfficMEName, (TH1F*)efficHist);
  } else if (histClassName == "TH2F"){
    efficME = theDQM->book2D(newEfficMEName, (TH2F*)efficHist);    
  } else if (histClassName == "TH3F"){
    efficME = theDQM->book3D(newEfficMEName, (TH3F*)efficHist);    
  } 
  

  if ( !efficME ) {
    LogInfo("DQMGenericClient") << "computeEfficiency() : "
                                << "Cannot book effic-ME from the DQM\n";
    return;
  }

  // Update: 2009-9-16 slaunwhj
  // call the most generic efficiency function
  // works up to 3-d histograms

  generic_eff (hSim, hReco, efficME, type);
  
  //   const int nBin = efficME->getNbinsX();
  //   for(int bin = 0; bin <= nBin; ++bin) {
  //     const float nSim  = simME ->getBinContent(bin);
  //     const float nReco = recoME->getBinContent(bin);
  //     float eff =0;
  //     if (type=="fake")eff = nSim ? 1-nReco/nSim : 0.;
  //     else eff= nSim ? nReco/nSim : 0.;
  //     const float err = nSim && eff <= 1 ? sqrt(eff*(1-eff)/nSim) : 0.;
  //     efficME->setBinContent(bin, eff);
  //     efficME->setBinError(bin, err);
  //   }
  efficME->setEntries(simME->getEntries());

  // Global efficiency
  ME* globalEfficME = theDQM->get(efficDir+"/globalEfficiencies");
  if ( !globalEfficME ) globalEfficME = theDQM->book1D("globalEfficiencies", "Global efficiencies", 1, 0, 1);
  if ( !globalEfficME ) {
    LogInfo("DQMGenericClient") << "computeEfficiency() : "
                              << "Cannot book globalEffic-ME from the DQM\n";
    return;
  }
  TH1F* hGlobalEffic = globalEfficME->getTH1F();
  if ( !hGlobalEffic ) {
    LogInfo("DQMGenericClient") << "computeEfficiency() : "
                              << "Cannot create TH1F from ME, globalEfficME\n";
    return;
  }

  const float nSimAll = hSim->GetEntries();
  const float nRecoAll = hReco->GetEntries();
  float efficAll=0; 
  if (type=="fake")   efficAll = nSimAll ? 1-nRecoAll/nSimAll : 0;
  else   efficAll = nSimAll ? nRecoAll/nSimAll : 0;
  const float errorAll = nSimAll && efficAll < 1 ? sqrt(efficAll*(1-efficAll)/nSimAll) : 0;

  const int iBin = hGlobalEffic->Fill(newEfficMEName.c_str(), 0);
  hGlobalEffic->SetBinContent(iBin, efficAll);
  hGlobalEffic->SetBinError(iBin, errorAll);
}

void DQMGenericClient::computeResolution(const string& startDir, const string& namePrefix, const string& titlePrefix,
                                         const std::string& srcName)
{
  if ( ! theDQM->dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeResolution() : "
                                  << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  theDQM->cd();

  ME* srcME = theDQM->get(startDir+"/"+srcName);
  if ( !srcME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeResolution() : "
                                  << "No source ME '" << srcName << "' found\n";
    }
    return;
  }

  TH2F* hSrc = srcME->getTH2F();
  if ( !hSrc ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeResolution() : "
                                  << "Cannot create TH2F from source-ME\n";
    }
    return;
  }

  const int nBin = hSrc->GetNbinsX();
  const double xMin = hSrc->GetXaxis()->GetXmin();
  const double xMax = hSrc->GetXaxis()->GetXmax();

  string newDir = startDir;
  string newPrefix = namePrefix;
  string::size_type shiftPos;
  if ( string::npos != (shiftPos = namePrefix.rfind('/')) ) {
    newDir += "/"+namePrefix.substr(0, shiftPos);
    newPrefix.erase(0, shiftPos+1);
  }

  theDQM->setCurrentFolder(newDir);

  ME* meanME = theDQM->book1D(newPrefix+"_Mean", titlePrefix+" Mean", nBin, xMin, xMax);
  ME* sigmaME = theDQM->book1D(newPrefix+"_Sigma", titlePrefix+" Sigma", nBin, xMin, xMax);
//  ME* chi2ME  = theDQM->book1D(namePrefix+"_Chi2" , titlePrefix+" #Chi^{2}", nBin, xMin, xMax); // N/A

  if (! resLimitedFit_ ) {
    FitSlicesYTool fitTool(srcME);
    fitTool.getFittedMeanWithError(meanME);
    fitTool.getFittedSigmaWithError(sigmaME);
    ////  fitTool.getFittedChisqWithError(chi2ME); // N/A
  } else {
    limitedFit(srcME,meanME,sigmaME);
  }
}

void DQMGenericClient::normalizeToEntries(const std::string& startDir, const std::string& histName) 
{
  if ( ! theDQM->dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  theDQM->cd();

  ME* element = theDQM->get(startDir+"/"+histName);

  if ( !element ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "No such element '" << histName << "' found\n";
    }
    return;
  }

  TH1F* hist  = element->getTH1F();
  if ( !hist) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "Cannot create TH1F from ME\n";
    }
    return;
  }

  const double entries = hist->GetEntries();
  if ( entries != 0 ) {
    hist->Scale(1./entries);
  }
  else {
    LogInfo("DQMGenericClient") << "normalizeToEntries() : " 
                                   << "Zero entries in histogram\n";
  }

  return;
}

void DQMGenericClient::makeCumulativeDist(const std::string& startDir, const std::string& cdName) 
{
  if ( ! theDQM->dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "makeCumulativeDist() : "
                                     << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  theDQM->cd();

  ME* element_cd = theDQM->get(startDir+"/"+cdName);

  if ( !element_cd ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "makeCumulativeDist() : "
                                     << "No such element '" << cdName << "' found\n";
    }
    return;
  }

  TH1F* cd  = element_cd->getTH1F();

  if ( !cd ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "makeCumulativeDist() : "
                                     << "Cannot create TH1F from ME\n";
    }
    return;
  }

  int n_bins = cd->GetNbinsX() + 1;

  for (int i = 1; i <= n_bins; i++) {
    cd->SetBinContent(i,cd->GetBinContent(i) + cd->GetBinContent(i-1));
  }

  return;
}

void DQMGenericClient::limitedFit(MonitorElement * srcME, MonitorElement * meanME, MonitorElement * sigmaME)
{
  TH2F * histo = srcME->getTH2F();

  static int i = 0;
  i++;

  // Fit slices projected along Y from bins in X 
  double cont_min = 100;    //Minimum number of entries
  Int_t binx =  histo->GetXaxis()->GetNbins();

  for (int i = 1; i <= binx ; i++) {
    TString iString(i);
    TH1 *histoY =  histo->ProjectionY(" ", i, i);
    double cont = histoY->GetEntries();

    if (cont >= cont_min) {
      float minfit = histoY->GetMean() - histoY->GetRMS();
      float maxfit = histoY->GetMean() + histoY->GetRMS();
      
      TF1 *fitFcn = new TF1(TString("g")+histo->GetName()+iString,"gaus",minfit,maxfit);
      double x1,x2;
      fitFcn->GetRange(x1,x2);

      histoY->Fit(fitFcn,"QR0","",x1,x2);

//      histoY->Fit(fitFcn->GetName(),"RME");
      double *par = fitFcn->GetParameters();
      double *err = fitFcn->GetParErrors();

      meanME->setBinContent(i,par[1]);
      meanME->setBinError(i,err[1]);

      sigmaME->setBinContent(i,par[2]);
      sigmaME->setBinError(i,err[2]);
      if(fitFcn) delete fitFcn;
      if(histoY) delete histoY;
    }
    else {
      if(histoY) delete histoY;
      continue;
    }
  }
}

//=================================

void DQMGenericClient::findAllSubdirectories (std::string dir, std::set<std::string> * myList, TString pattern = "") {

  if (pattern != "") {
    if (pattern.Contains(nonPerlWildcard)) pattern.ReplaceAll("*",".*");
    TPRegexp regexp(pattern);
    theDQM->cd(dir);
    vector <string> foundDirs = theDQM->getSubdirs();
    for(vector<string>::const_iterator iDir = foundDirs.begin();
        iDir != foundDirs.end(); ++iDir) {
      TString dirName = iDir->substr(iDir->rfind('/') + 1, iDir->length());
      if (dirName.Contains(regexp))
        findAllSubdirectories ( *iDir, myList);
    }
  }
  //std::cout << "Looking for directory " << dir ;
  else if (theDQM->dirExists(dir)){
    //std::cout << "... it exists! Inserting it into the list ";
    myList->insert(dir);
    //std::cout << "... now list has size " << myList->size() << std::endl;
    theDQM->cd(dir);
    findAllSubdirectories (dir, myList, "*");
  } else {
    //std::cout << "... DOES NOT EXIST!!! Skip bogus dir" << std::endl;
    
    LogInfo ("DQMGenericClient") << "Trying to find sub-directories of " << dir
                                 << " failed because " << dir  << " does not exist";
                                 
  }
  return;
}


void DQMGenericClient::generic_eff (TH1* denom, TH1* numer, MonitorElement* efficiencyHist, const std::string & type) {
  for (int iBinX = 1; iBinX < denom->GetNbinsX()+1; iBinX++){
    for (int iBinY = 1; iBinY < denom->GetNbinsY()+1; iBinY++){
      for (int iBinZ = 1; iBinZ < denom->GetNbinsZ()+1; iBinZ++){

        int globalBinNum = denom->GetBin(iBinX, iBinY, iBinZ);
        
               
        float numerVal = numer->GetBinContent(globalBinNum);
        float denomVal = denom->GetBinContent(globalBinNum);

        float effVal = 0;

        // fake eff is in use
        if (type == "fake") {          
          effVal = denomVal ? (1 - numerVal / denomVal) : 0;
        } else {
          effVal = denomVal ? numerVal / denomVal : 0;
        }

        float errVal = (denomVal && (effVal <=1)) ? sqrt(effVal*(1-effVal)/denomVal) : 0;

        LogDebug ("DQMGenericClient") << "(iBinX, iBinY, iBinZ)  = "
             << iBinX << ", "
             << iBinY << ", "
             << iBinZ << "), global bin =  "  << globalBinNum
             << "eff = " << numerVal << "  /  " << denomVal
             << " =  " << effVal 
             << " ... setting the error for that bin ... " << endl
             << endl;

        
        efficiencyHist->setBinContent(globalBinNum, effVal);
        efficiencyHist->setBinError(globalBinNum, errVal);
      }
    }
  }

  //efficiencyHist->setMinimum(0.0);
  //efficiencyHist->setMaximum(1.0);
}

/* vim:set ts=2 sts=2 sw=2 expandtab: */

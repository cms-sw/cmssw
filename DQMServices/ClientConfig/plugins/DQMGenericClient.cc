/*
 *  Class:DQMGenericClient 
 *
 *
 *  $Date: 2009/04/07 17:18:36 $
 *  $Revision: 1.3 $
 * 
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "DQMServices/ClientConfig/interface/DQMGenericClient.h"

#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TH1F.h>
#include <cmath>

using namespace std;
using namespace edm;

typedef MonitorElement ME;
typedef vector<string> vstring;

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

void DQMGenericClient::endJob()
{
  theDQM = 0;
  theDQM = Service<DQMStore>().operator->();

  if ( ! theDQM ) {
    LogError("DQMGenericClient") << "Cannot create DQMStore instance\n";
    return;
  }

  // Process wildcard in the sub-directory
  set<string> subDirSet;

  for(vstring::const_iterator iSubDir = subDirs_.begin();
      iSubDir != subDirs_.end(); ++iSubDir) {
    string subDir = *iSubDir;

    if ( subDir[subDir.size()-1] == '/' ) subDir.erase(subDir.size()-1);

    if ( subDir[subDir.size()-1] == '*' ) {
      isWildcardUsed_ = true;
      const string::size_type shiftPos = subDir.rfind('/');

      const string searchPath = subDir.substr(0, shiftPos);
      theDQM->cd(searchPath);

      vector<string> foundDirs = theDQM->getSubdirs();
      const string matchStr = subDir.substr(0, subDir.size()-2);

      for(vector<string>::const_iterator iDir = foundDirs.begin();
          iDir != foundDirs.end(); ++iDir) {
        const string dirPrefix = iDir->substr(0, matchStr.size());

        if ( dirPrefix == matchStr ) {
          subDirSet.insert(*iDir);
        }
      }
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
        LogError("DQMGenericClient") << "Wrong input to effCmds\n";
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
        LogError("DQMGenericClient") << "Wrong input to resCmds\n";
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
        LogError("DQMGenericClient") << "Wrong input to normCmds\n";
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
       LogError("DQMGenericClient") << "Wrong input to cdCmds\n";
       continue;
     }

     makeCumulativeDist(dirName, args[0]);
   }
  }

  if ( verbose_ > 0 ) theDQM->showDirStructure();

  if ( ! outputFileName_.empty() ) theDQM->save(outputFileName_);
}

void DQMGenericClient::computeEfficiency(const string& startDir, const string& efficMEName, const string& efficMETitle,
                                         const string& recoMEName, const string& simMEName, const std::string & type)
{
  if ( ! theDQM->dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "computeEfficiency() : "
                                  << "Cannot find sub-directory " << startDir << endl; 
    }
    return;
  }

  theDQM->cd();

  ME* simME  = theDQM->get(startDir+"/"+simMEName);
  ME* recoME = theDQM->get(startDir+"/"+recoMEName);

  if ( !simME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "computeEfficiency() : "
                                  << "No sim-ME '" << simMEName << "' found\n";
    }
    return;
  }

  if ( !recoME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "computeEfficiency() : " 
                                  << "No reco-ME '" << recoMEName << "' found\n";
    }
    return;
  }

  TH1F* hSim  = simME ->getTH1F();
  TH1F* hReco = recoME->getTH1F();
  if ( !hSim || !hReco ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "computeEfficiency() : "
                                  << "Cannot create TH1F from ME\n";
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
  ME* efficME = theDQM->book1D(newEfficMEName, efficMETitle, hSim->GetNbinsX(), hSim->GetXaxis()->GetXmin(), hSim->GetXaxis()->GetXmax()); 

  if ( !efficME ) {
    LogWarning("DQMGenericClient") << "computeEfficiency() : "
                                << "Cannot book effic-ME from the DQM\n";
    return;
  }

  const int nBin = efficME->getNbinsX();
  for(int bin = 0; bin <= nBin; ++bin) {
    const float nSim  = simME ->getBinContent(bin);
    const float nReco = recoME->getBinContent(bin);
    float eff =0;
    if (type=="fake")eff = nSim ? 1-nReco/nSim : 0.;
    else eff= nSim ? nReco/nSim : 0.;
    const float err = nSim && eff <= 1 ? sqrt(eff*(1-eff)/nSim) : 0.;
    efficME->setBinContent(bin, eff);
    efficME->setBinError(bin, err);
  }
  efficME->setEntries(simME->getEntries());

  // Global efficiency
  ME* globalEfficME = theDQM->get(efficDir+"/globalEfficiencies");
  if ( !globalEfficME ) globalEfficME = theDQM->book1D("globalEfficiencies", "Global efficiencies", 1, 0, 1);
  if ( !globalEfficME ) {
    LogError("DQMGenericClient") << "computeEfficiency() : "
                              << "Cannot book globalEffic-ME from the DQM\n";
    return;
  }
  TH1F* hGlobalEffic = globalEfficME->getTH1F();
  if ( !hGlobalEffic ) {
    LogError("DQMGenericClient") << "computeEfficiency() : "
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
      LogWarning("DQMGenericClient") << "computeResolution() : "
                                  << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  theDQM->cd();

  ME* srcME = theDQM->get(startDir+"/"+srcName);
  if ( !srcME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "computeResolution() : "
                                  << "No source ME '" << srcName << "' found\n";
    }
    return;
  }

  TH2F* hSrc = srcME->getTH2F();
  if ( !hSrc ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "computeResolution() : "
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
      LogWarning("DQMGenericClient") << "normalizeToEntries() : "
                                     << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  theDQM->cd();

  ME* element = theDQM->get(startDir+"/"+histName);

  if ( !element ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "normalizeToEntries() : "
                                     << "No such element '" << histName << "' found\n";
    }
    return;
  }

  TH1F* hist  = element->getTH1F();
  if ( !hist) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "normalizeToEntries() : "
                                     << "Cannot create TH1F from ME\n";
    }
    return;
  }

  const double entries = hist->GetEntries();
  if ( entries != 0 ) {
    hist->Scale(1./entries);
  }
  else {
    LogWarning("DQMGenericClient") << "normalizeToEntries() : " 
                                   << "Zero entries in histogram\n";
  }

  return;
}

void DQMGenericClient::makeCumulativeDist(const std::string& startDir, const std::string& cdName) 
{
  if ( ! theDQM->dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "makeCumulativeDist() : "
                                     << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  theDQM->cd();

  ME* element_cd = theDQM->get(startDir+"/"+cdName);

  if ( !element_cd ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "makeCumulativeDist() : "
                                     << "No such element '" << cdName << "' found\n";
    }
    return;
  }

  TH1F* cd  = element_cd->getTH1F();

  if ( !cd ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogWarning("DQMGenericClient") << "makeCumulativeDist() : "
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

/* vim:set ts=2 sts=2 sw=2 expandtab: */

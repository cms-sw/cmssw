/*
 *  Class:DQMRivetClient 
 *
 *
 * 
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "GeneratorInterface/RivetInterface/interface/DQMRivetClient.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TH1F.h>
#include <TClass.h>
#include <TString.h>
#include <TPRegexp.h>

#include <cmath>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace edm;

typedef DQMRivetClient::MonitorElement ME;

DQMRivetClient::DQMRivetClient(const ParameterSet& pset) {
  typedef std::vector<edm::ParameterSet> VPSet;
  typedef std::vector<std::string> vstring;
  typedef boost::escaped_list_separator<char> elsc;

  elsc commonEscapes("\\", " \t", "\'");

  // Parse Normalization commands
  vstring normCmds = pset.getUntrackedParameter<vstring>("normalizationToIntegral", vstring());
  for (vstring::const_iterator normCmd = normCmds.begin(); normCmd != normCmds.end(); ++normCmd) {
    if (normCmd->empty())
      continue;
    boost::tokenizer<elsc> tokens(*normCmd, commonEscapes);

    vector<string> args;
    for (boost::tokenizer<elsc>::const_iterator iToken = tokens.begin(); iToken != tokens.end(); ++iToken) {
      if (iToken->empty())
        continue;
      args.push_back(*iToken);
    }

    if (args.empty() or args.size() > 2) {
      LogInfo("DQMRivetClient") << "Wrong input to normCmds\n";
      continue;
    }

    NormOption opt;
    opt.name = args[0];
    opt.normHistName = args.size() == 2 ? args[1] : args[0];

    normOptions_.push_back(opt);
  }

  VPSet normSets = pset.getUntrackedParameter<VPSet>("normalizationToIntegralSets", VPSet());
  for (VPSet::const_iterator normSet = normSets.begin(); normSet != normSets.end(); ++normSet) {
    NormOption opt;
    opt.name = normSet->getUntrackedParameter<string>("name");
    opt.normHistName = normSet->getUntrackedParameter<string>("normalizedTo", opt.name);

    normOptions_.push_back(opt);
  }

  //normalize to lumi
  vstring lumiCmds = pset.getUntrackedParameter<vstring>("normalizationToLumi", vstring());
  for (vstring::const_iterator lumiCmd = lumiCmds.begin(); lumiCmd != lumiCmds.end(); ++lumiCmd) {
    if (lumiCmd->empty())
      continue;
    boost::tokenizer<elsc> tokens(*lumiCmd, commonEscapes);

    vector<string> args;
    for (boost::tokenizer<elsc>::const_iterator iToken = tokens.begin(); iToken != tokens.end(); ++iToken) {
      if (iToken->empty())
        continue;
      args.push_back(*iToken);
    }

    if (args.size() != 2) {
      LogInfo("DQMRivetClient") << "Wrong input to lumiCmds\n";
      continue;
    }

    DQMRivetClient::LumiOption opt;
    opt.name = args[0];
    opt.normHistName = args[1];
    opt.xsection = pset.getUntrackedParameter<double>("xsection", -1.);
    //opt.xsection = atof(args[2].c_str());

    //std::cout << opt.name << " " << opt.normHistName << " " << opt.xsection << std::endl;
    lumiOptions_.push_back(opt);
  }

  //multiply by a number
  vstring scaleCmds = pset.getUntrackedParameter<vstring>("scaleBy", vstring());
  for (vstring::const_iterator scaleCmd = scaleCmds.begin(); scaleCmd != scaleCmds.end(); ++scaleCmd) {
    if (scaleCmd->empty())
      continue;
    boost::tokenizer<elsc> tokens(*scaleCmd, commonEscapes);

    vector<string> args;
    for (boost::tokenizer<elsc>::const_iterator iToken = tokens.begin(); iToken != tokens.end(); ++iToken) {
      if (iToken->empty())
        continue;
      args.push_back(*iToken);
    }

    if (args.empty() or args.size() > 2) {
      LogInfo("DQMRivetClient") << "Wrong input to normCmds\n";
      continue;
    }

    ScaleFactorOption opt;
    opt.name = args[0];
    opt.scale = atof(args[1].c_str());
    scaleOptions_.push_back(opt);
  }

  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName", "");
  subDirs_ = pset.getUntrackedParameter<vstring>("subDirs");
}

void DQMRivetClient::endRun(const edm::Run& r, const edm::EventSetup& c) {
  typedef vector<string> vstring;

  // Update 2009-09-23
  // Migrated all code from endJob to this function
  // endJob is not necessarily called in the proper sequence
  // and does not necessarily book histograms produced in
  // that step.
  // It more robust to do the histogram manipulation in
  // this endRun function

  theDQM = nullptr;
  theDQM = Service<DQMStore>().operator->();

  if (!theDQM) {
    LogInfo("DQMRivetClient") << "Cannot create DQMStore instance\n";
    return;
  }

  // Process wildcard in the sub-directory
  set<string> subDirSet;

  for (vstring::const_iterator iSubDir = subDirs_.begin(); iSubDir != subDirs_.end(); ++iSubDir) {
    string subDir = *iSubDir;

    if (subDir[subDir.size() - 1] == '/')
      subDir.erase(subDir.size() - 1);

    subDirSet.insert(subDir);
  }

  for (set<string>::const_iterator iSubDir = subDirSet.begin(); iSubDir != subDirSet.end(); ++iSubDir) {
    const string& dirName = *iSubDir;
    for (vector<NormOption>::const_iterator normOption = normOptions_.begin(); normOption != normOptions_.end();
         ++normOption) {
      normalizeToIntegral(dirName, normOption->name, normOption->normHistName);
    }
  }

  for (set<string>::const_iterator iSubDir = subDirSet.begin(); iSubDir != subDirSet.end(); ++iSubDir) {
    const string& dirName = *iSubDir;
    for (vector<LumiOption>::const_iterator lumiOption = lumiOptions_.begin(); lumiOption != lumiOptions_.end();
         ++lumiOption) {
      normalizeToLumi(dirName, lumiOption->name, lumiOption->normHistName, lumiOption->xsection);
    }
  }

  for (set<string>::const_iterator iSubDir = subDirSet.begin(); iSubDir != subDirSet.end(); ++iSubDir) {
    const string& dirName = *iSubDir;
    for (vector<ScaleFactorOption>::const_iterator scaleOption = scaleOptions_.begin();
         scaleOption != scaleOptions_.end();
         ++scaleOption) {
      scaleByFactor(dirName, scaleOption->name, scaleOption->scale);
    }
  }

  if (!outputFileName_.empty())
    theDQM->save(outputFileName_);
}

void DQMRivetClient::endJob() {
  // Update 2009-09-23
  // Migrated all code from here to endRun

  LogTrace("DQMRivetClient") << "inside of ::endJob()" << endl;
}

void DQMRivetClient::normalizeToIntegral(const std::string& startDir,
                                         const std::string& histName,
                                         const std::string& normHistName) {
  if (!theDQM->dirExists(startDir)) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Cannot find sub-directory " << startDir << endl;
    return;
  }

  theDQM->cd();

  ME* element = theDQM->get(startDir + "/" + histName);
  ME* normME = theDQM->get(startDir + "/" + normHistName);

  if (!element) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "No such element '" << histName << "' found\n";
    return;
  }

  if (!normME) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "No such element '" << normHistName << "' found\n";
    return;
  }

  TH1F* hist = element->getTH1F();
  if (!hist) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Cannot create TH1F from ME\n";
    return;
  }

  TH1F* normHist = normME->getTH1F();
  if (!normHist) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Cannot create TH1F from ME\n";
    return;
  }

  const double entries = normHist->Integral();
  if (entries != 0) {
    hist->Scale(1. / entries, "width");
  } else {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Zero entries in histogram\n";
  }

  return;
}

void DQMRivetClient::normalizeToLumi(const std::string& startDir,
                                     const std::string& histName,
                                     const std::string& normHistName,
                                     double xsection) {
  normalizeToIntegral(startDir, histName, normHistName);
  theDQM->cd();
  ME* element = theDQM->get(startDir + "/" + histName);
  TH1F* hist = element->getTH1F();
  if (!hist) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Cannot create TH1F from ME\n";
    return;
  }
  hist->Scale(xsection);
  return;
}

void DQMRivetClient::scaleByFactor(const std::string& startDir, const std::string& histName, double factor) {
  if (!theDQM->dirExists(startDir)) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Cannot find sub-directory " << startDir << endl;
    return;
  }

  theDQM->cd();

  ME* element = theDQM->get(startDir + "/" + histName);

  if (!element) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "No such element '" << histName << "' found\n";
    return;
  }

  TH1F* hist = element->getTH1F();
  if (!hist) {
    LogInfo("DQMRivetClient") << "normalizeToEntries() : "
                              << "Cannot create TH1F from ME\n";
    return;
  }
  hist->Scale(factor);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(DQMRivetClient);

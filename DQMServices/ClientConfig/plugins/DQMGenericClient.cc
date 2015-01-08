/*
 *  Class:DQMGenericClient 
 *
 *
 * 
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "DQMServices/ClientConfig/interface/DQMGenericClient.h"

#include "DQMServices/ClientConfig/interface/FitSlicesYTool.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TH1F.h>
#include <TClass.h>
#include <TString.h>
#include <TPRegexp.h>

#include <cmath>
#include <climits>
#include <boost/tokenizer.hpp>

using namespace std;
using namespace edm;

typedef MonitorElement ME;

TPRegexp metacharacters("[\\^\\$\\.\\*\\+\\?\\|\\(\\)\\{\\}\\[\\]]");
TPRegexp nonPerlWildcard("\\w\\*|^\\*");

DQMGenericClient::DQMGenericClient(const ParameterSet& pset)
{
  typedef std::vector<edm::ParameterSet> VPSet;
  typedef std::vector<std::string> vstring;
  typedef boost::escaped_list_separator<char> elsc;

  elsc commonEscapes("\\", " \t", "\'");

  verbose_ = pset.getUntrackedParameter<unsigned int>("verbose", 0);

  // Parse efficiency commands
  vstring effCmds = pset.getParameter<vstring>("efficiency");
  for ( vstring::const_iterator effCmd = effCmds.begin();
        effCmd != effCmds.end(); ++effCmd )
  {
    if ( effCmd->empty() ) continue;

    boost::tokenizer<elsc> tokens(*effCmd, commonEscapes);

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

    EfficOption opt;
    opt.name = args[0];
    opt.title = args[1];
    opt.numerator = args[2];
    opt.denominator = args[3];
    opt.isProfile = false;

    const string typeName = args.size() == 4 ? "eff" : args[4];
    if ( typeName == "eff" ) opt.type = 1;
    else if ( typeName == "fake" ) opt.type = 2;
    else opt.type = 0;
 
    efficOptions_.push_back(opt);
  }
  
  VPSet efficSets = pset.getUntrackedParameter<VPSet>("efficiencySets", VPSet());
  for ( VPSet::const_iterator efficSet = efficSets.begin();
        efficSet != efficSets.end(); ++efficSet )
  {
    EfficOption opt;
    opt.name = efficSet->getUntrackedParameter<string>("name");
    opt.title = efficSet->getUntrackedParameter<string>("title");
    opt.numerator = efficSet->getUntrackedParameter<string>("numerator");
    opt.denominator = efficSet->getUntrackedParameter<string>("denominator");
    opt.isProfile = false;

    const string typeName = efficSet->getUntrackedParameter<string>("typeName", "eff");
    if ( typeName == "eff" ) opt.type = 1;
    else if ( typeName == "fake" ) opt.type = 2;
    else opt.type = 0;

    efficOptions_.push_back(opt);
  }

  // Parse efficiency profiles
  vstring effProfileCmds = pset.getUntrackedParameter<vstring>("efficiencyProfile", vstring());
  for ( vstring::const_iterator effProfileCmd = effProfileCmds.begin();
        effProfileCmd != effProfileCmds.end(); ++effProfileCmd )
  {
    if ( effProfileCmd->empty() ) continue;

    boost::tokenizer<elsc> tokens(*effProfileCmd, commonEscapes);

    vector<string> args;
    for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
        iToken != tokens.end(); ++iToken) {
      if ( iToken->empty() ) continue;
      args.push_back(*iToken);
    }

    if ( args.size() < 4 ) {
      LogInfo("DQMGenericClient") << "Wrong input to effProfileCmds\n";
      continue;
    }

    EfficOption opt;
    opt.name = args[0];
    opt.title = args[1];
    opt.numerator = args[2];
    opt.denominator = args[3];
    opt.isProfile = true;

    const string typeName = args.size() == 4 ? "eff" : args[4];
    if ( typeName == "eff" ) opt.type = 1;
    else if ( typeName == "fake" ) opt.type = 2;
    else opt.type = 0;
 
    efficOptions_.push_back(opt);
  }

  VPSet effProfileSets = pset.getUntrackedParameter<VPSet>("efficiencyProfileSets", VPSet());
  for ( VPSet::const_iterator effProfileSet = effProfileSets.begin();
        effProfileSet != effProfileSets.end(); ++effProfileSet )
  {
    EfficOption opt;
    opt.name = effProfileSet->getUntrackedParameter<string>("name");
    opt.title = effProfileSet->getUntrackedParameter<string>("title");
    opt.numerator = effProfileSet->getUntrackedParameter<string>("numerator");
    opt.denominator = effProfileSet->getUntrackedParameter<string>("denominator");
    opt.isProfile = true;

    const string typeName = effProfileSet->getUntrackedParameter<string>("typeName", "eff");
    if ( typeName == "eff" ) opt.type = 1;
    else if ( typeName == "fake" ) opt.type = 2;
    else opt.type = 0;

    efficOptions_.push_back(opt);
  }

  // Parse resolution commands
  vstring resCmds = pset.getParameter<vstring>("resolution");
  for ( vstring::const_iterator resCmd = resCmds.begin();
        resCmd != resCmds.end(); ++resCmd )
  {
    if ( resCmd->empty() ) continue;
    boost::tokenizer<elsc> tokens(*resCmd, commonEscapes);

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

    ResolOption opt;
    opt.namePrefix = args[0];
    opt.titlePrefix = args[1];
    opt.srcName = args[2];

    resolOptions_.push_back(opt);
  }

  VPSet resolSets = pset.getUntrackedParameter<VPSet>("resolutionSets", VPSet());
  for ( VPSet::const_iterator resolSet = resolSets.begin();
        resolSet != resolSets.end(); ++resolSet )
  {
    ResolOption opt;
    opt.namePrefix = resolSet->getUntrackedParameter<string>("namePrefix");
    opt.titlePrefix = resolSet->getUntrackedParameter<string>("titlePrefix");
    opt.srcName = resolSet->getUntrackedParameter<string>("srcName");

    resolOptions_.push_back(opt);
  }

  // Parse profiles
  vstring profileCmds = pset.getUntrackedParameter<vstring>("profile", vstring());
  for(const auto& profileCmd: profileCmds) {
    boost::tokenizer<elsc> tokens(profileCmd, commonEscapes);

    vector<string> args;
    for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
        iToken != tokens.end(); ++iToken) {
      if ( iToken->empty() ) continue;
      args.push_back(*iToken);
    }

    if ( args.size() != 3 ) {
      LogInfo("DQMGenericClient") << "Wrong input to profileCmds\n";
      continue;
    }

    ProfileOption opt;
    opt.name = args[0];
    opt.title = args[1];
    opt.srcName = args[2];

    profileOptions_.push_back(opt);
  }

  VPSet profileSets = pset.getUntrackedParameter<VPSet>("profileSets", VPSet());
  for(const auto& profileSet: profileSets) {
    ProfileOption opt;
    opt.name = profileSet.getUntrackedParameter<string>("name");
    opt.title = profileSet.getUntrackedParameter<string>("title");
    opt.srcName = profileSet.getUntrackedParameter<string>("srcName");

    profileOptions_.push_back(opt);
  }

  // Parse Normalization commands
  vstring normCmds = pset.getUntrackedParameter<vstring>("normalization", vstring());
  for ( vstring::const_iterator normCmd = normCmds.begin();
        normCmd != normCmds.end(); ++normCmd )
  {
    if ( normCmd->empty() ) continue;
    boost::tokenizer<elsc> tokens(*normCmd, commonEscapes);

    vector<string> args;
    for(boost::tokenizer<elsc>::const_iterator iToken = tokens.begin();
        iToken != tokens.end(); ++iToken) {
      if ( iToken->empty() ) continue;
      args.push_back(*iToken);
    }

    if ( args.empty() or args.size() > 2 ) {
      LogInfo("DQMGenericClient") << "Wrong input to normCmds\n";
      continue;
    }

    NormOption opt;
    opt.name = args[0];
    opt.normHistName = args.size() == 2 ? args[1] : args[0];

    normOptions_.push_back(opt);
  }

  VPSet normSets = pset.getUntrackedParameter<VPSet>("normalizationSets", VPSet());
  for ( VPSet::const_iterator normSet = normSets.begin();
        normSet != normSets.end(); ++normSet )
  {
    NormOption opt;
    opt.name = normSet->getUntrackedParameter<string>("name");
    opt.normHistName = normSet->getUntrackedParameter<string>("normalizedTo", opt.name);

    normOptions_.push_back(opt);
  }

  // Cumulative distributions
  vstring cdCmds = pset.getUntrackedParameter<vstring>("cumulativeDists", vstring());
  for ( vstring::const_iterator cdCmd = cdCmds.begin();
        cdCmd != cdCmds.end(); ++cdCmd )
  {
    if ( cdCmd->empty() ) continue;
    boost::tokenizer<elsc> tokens(*cdCmd, commonEscapes);

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

    CDOption opt;
    opt.name = args[0];

    cdOptions_.push_back(opt);
  }

  VPSet cdSets = pset.getUntrackedParameter<VPSet>("cumulativeDistSets", VPSet());
  for ( VPSet::const_iterator cdSet = cdSets.begin();
        cdSet != cdSets.end(); ++cdSet )
  {
    CDOption opt;
    opt.name = cdSet->getUntrackedParameter<string>("name");

    cdOptions_.push_back(opt);
  }

  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName", "");
  subDirs_ = pset.getUntrackedParameter<vstring>("subDirs");

  resLimitedFit_ = pset.getUntrackedParameter<bool>("resolutionLimitedFit",false);
  isWildcardUsed_ = false;
}

void DQMGenericClient::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter) {

  typedef vector<string> vstring;

  // Update 2014-04-02
  // Migrated back to the endJob. the DQMFileSaver logic has
  // to be reviewed to guarantee that the endJob is properly
  // considered. The splitting per run is done centrally when
  // running the harvesting in production

  // Update 2009-09-23
  // Migrated all code from endJob to this function
  // endJob is not necessarily called in the proper sequence
  // and does not necessarily book histograms produced in
  // that step.
  // It more robust to do the histogram manipulation in
  // this endRun function

  // needed to access the DQMStore::save method
  theDQM = 0;
  theDQM = Service<DQMStore>().operator->();

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
      
      findAllSubdirectories (ibooker, igetter, searchPath, &subDirSet, pattern);

    }
    else {
      subDirSet.insert(subDir);
    }
  }

  for(set<string>::const_iterator iSubDir = subDirSet.begin();
      iSubDir != subDirSet.end(); ++iSubDir) {
    const string& dirName = *iSubDir;

    for ( vector<EfficOption>::const_iterator efficOption = efficOptions_.begin();
          efficOption != efficOptions_.end(); ++efficOption )
    {
      computeEfficiency(ibooker, igetter, dirName, efficOption->name, efficOption->title, 
                        efficOption->numerator, efficOption->denominator,
                        efficOption->type, efficOption->isProfile);
    }

    for ( vector<ResolOption>::const_iterator resolOption = resolOptions_.begin();
          resolOption != resolOptions_.end(); ++resolOption )
    {
      computeResolution(ibooker, igetter, dirName, resolOption->namePrefix, resolOption->titlePrefix, resolOption->srcName);
    }

    for(const auto& profileOption: profileOptions_) {
      computeProfile(ibooker, igetter, dirName, profileOption.name, profileOption.title, profileOption.srcName);
    }

    for ( vector<NormOption>::const_iterator normOption = normOptions_.begin();
          normOption != normOptions_.end(); ++normOption )
    {
      normalizeToEntries(ibooker, igetter, dirName, normOption->name, normOption->normHistName);
    }

    for ( vector<CDOption>::const_iterator cdOption = cdOptions_.begin();
          cdOption != cdOptions_.end(); ++cdOption )
    {
      makeCumulativeDist(ibooker, igetter, dirName, cdOption->name);
    }
  }

  if ( ! outputFileName_.empty() ) theDQM->save(outputFileName_);
  
}

void DQMGenericClient::computeEfficiency (DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const string& startDir, const string& efficMEName,
					 const string& efficMETitle, const string& recoMEName, const string& simMEName, const int type, const bool makeProfile)
{
  if ( ! igetter.dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeEfficiency() : "
                                  << "Cannot find sub-directory " << startDir << endl; 
    }
    return;
  }

  ibooker.cd();

  ME* simME  = igetter.get(startDir+"/"+simMEName);
  ME* recoME = igetter.get(startDir+"/"+recoMEName);

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
  ibooker.setCurrentFolder(efficDir);

  if (makeProfile) {
    TProfile * efficHist = (hReco->GetXaxis()->GetXbins()->GetSize()==0) ?
      new TProfile(newEfficMEName.c_str(), efficMETitle.c_str(),
                   hReco->GetXaxis()->GetNbins(),
                   hReco->GetXaxis()->GetXmin(),
                   hReco->GetXaxis()->GetXmax()) :
      new TProfile(newEfficMEName.c_str(), efficMETitle.c_str(),
                   hReco->GetXaxis()->GetNbins(),
                   hReco->GetXaxis()->GetXbins()->GetArray());

    efficHist->GetXaxis()->SetTitle(hSim->GetXaxis()->GetTitle());
    efficHist->GetYaxis()->SetTitle(hSim->GetYaxis()->GetTitle());

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
    for (int i=1; i <= hReco->GetNbinsX(); i++) {

      const double nReco = hReco->GetBinContent(i);
      const double nSim = hSim->GetBinContent(i);
      if(nSim > INT_MAX || nSim < INT_MIN || nReco > INT_MAX || nReco < INT_MIN)
 	{
 	  LogError("DQMGenericClient")  << "computeEfficiency() : "
					<< "Overflow: bin content either too large or too small to be casted to int";
 	  return;
 	}

      if(std::string(hSim->GetXaxis()->GetBinLabel(i)) != "")
	efficHist->GetXaxis()->SetBinLabel(i, hSim->GetXaxis()->GetBinLabel(i));
      
      if ( nSim == 0 || nReco > nSim ) continue;
      const double effVal = nReco/nSim;

      const double errLo = TEfficiency::ClopperPearson((int)nSim, 
						       (int)nReco,
						       0.683,false);
      const double errUp = TEfficiency::ClopperPearson((int)nSim, 
						       (int)nReco,
						       0.683,true);
      const double errVal = (effVal - errLo > errUp - effVal) ? effVal - errLo : errUp - effVal;
      efficHist->SetBinContent(i, effVal);
      efficHist->SetBinEntries(i, 1);
      efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));
    }
#else
    for (int i=1; i <= hReco->GetNbinsX(); i++) {

      const double nReco = hReco->GetBinContent(i);
      const double nSim = hSim->GetBinContent(i);
      if(nSim > INT_MAX || nSim < INT_MIN || nReco > INT_MAX || nReco < INT_MIN)
        {
          LogError("DQMGenericClient")  << "computeEfficiency() : "
                                        << "Overflow: bin content either too large or too small to be casted to int";
          return;
        }

      TGraphAsymmErrorsWrapper asymm;
      std::pair<double, double> efficiencyWithError;
      efficiencyWithError = asymm.efficiency((int)nReco,
                                             (int)nSim);
      double effVal = efficiencyWithError.first;
      double errVal = efficiencyWithError.second;
      if ((int)nSim > 0) {
        efficHist->SetBinContent(i, effVal);
        efficHist->SetBinEntries(i, 1);
        efficHist->SetBinError(i, sqrt(effVal * effVal + errVal * errVal));
      }
      if(std::string(hSim->GetXaxis()->GetBinLabel(i)) != "")
	efficHist->GetXaxis()->SetBinLabel(i, hSim->GetXaxis()->GetBinLabel(i));
    }
#endif
    ibooker.bookProfile(newEfficMEName.c_str(),efficHist);
    delete efficHist;  
  }

  else {

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
      efficME = ibooker.book1D(newEfficMEName, (TH1F*)efficHist);
    } else if (histClassName == "TH2F"){
      efficME = ibooker.book2D(newEfficMEName, (TH2F*)efficHist);    
    } else if (histClassName == "TH3F"){
      efficME = ibooker.book3D(newEfficMEName, (TH3F*)efficHist);    
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

  }

  // Global efficiency
  ME* globalEfficME = igetter.get(efficDir+"/globalEfficiencies");
  if ( !globalEfficME ) globalEfficME = ibooker.book1D("globalEfficiencies", "Global efficiencies", 1, 0, 1);
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
  if ( type == 1 ) efficAll = nSimAll ? nRecoAll/nSimAll : 0;
  else if ( type == 2 ) efficAll = nSimAll ? 1-nRecoAll/nSimAll : 0;
  const float errorAll = nSimAll && efficAll < 1 ? sqrt(efficAll*(1-efficAll)/nSimAll) : 0;

  const int iBin = hGlobalEffic->Fill(newEfficMEName.c_str(), 0);
  hGlobalEffic->SetBinContent(iBin, efficAll);
  hGlobalEffic->SetBinError(iBin, errorAll);
}

void DQMGenericClient::computeResolution(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const string& startDir, const string& namePrefix, const string& titlePrefix,
                                         const std::string& srcName)
{
  if ( ! igetter.dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeResolution() : "
                                  << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  ibooker.cd();

  ME* srcME = igetter.get(startDir+"/"+srcName);
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

  string newDir = startDir;
  string newPrefix = namePrefix;
  string::size_type shiftPos;
  if ( string::npos != (shiftPos = namePrefix.rfind('/')) ) {
    newDir += "/"+namePrefix.substr(0, shiftPos);
    newPrefix.erase(0, shiftPos+1);
  }

  ibooker.setCurrentFolder(newDir);

  float * lowedgesfloats = new float[nBin+1];
  ME* meanME;
  ME* sigmaME;
  if (hSrc->GetXaxis()->GetXbins()->GetSize())
  {
    for (int j=0; j<nBin+1; ++j)
      lowedgesfloats[j] = (float)hSrc->GetXaxis()->GetXbins()->GetAt(j);
    meanME = ibooker.book1D(newPrefix+"_Mean", titlePrefix+" Mean", nBin, lowedgesfloats);
    sigmaME = ibooker.book1D(newPrefix+"_Sigma", titlePrefix+" Sigma", nBin, lowedgesfloats);
  }
  else
  {
    meanME = ibooker.book1D(newPrefix+"_Mean", titlePrefix+" Mean", nBin,
			    hSrc->GetXaxis()->GetXmin(),
			    hSrc->GetXaxis()->GetXmax());
    sigmaME = ibooker.book1D(newPrefix+"_Sigma", titlePrefix+" Sigma", nBin,
			    hSrc->GetXaxis()->GetXmin(),
			    hSrc->GetXaxis()->GetXmax());			     
  }
  
  if (meanME && sigmaME)
  {
    meanME->setEfficiencyFlag();
    sigmaME->setEfficiencyFlag();

    if (! resLimitedFit_ ) {
      FitSlicesYTool fitTool(srcME);
      fitTool.getFittedMeanWithError(meanME);
      fitTool.getFittedSigmaWithError(sigmaME);
      ////  fitTool.getFittedChisqWithError(chi2ME); // N/A
    } else {
      limitedFit(srcME,meanME,sigmaME);
    }
  }
  delete[] lowedgesfloats;
}

void DQMGenericClient::computeProfile(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const std::string& startDir, const std::string& profileMEName, const std::string& profileMETitle, const std::string& srcMEName) {
  if(!igetter.dirExists(startDir)) {
    if(verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeProfile() : "
                                  << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  ibooker.cd();

  ME* srcME = igetter.get(startDir+"/"+srcMEName);
  if ( !srcME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeProfile() : "
                                  << "No source ME '" << srcMEName << "' found\n";
    }
    return;
  }

  TH2F* hSrc = srcME->getTH2F();
  if ( !hSrc ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "computeProfile() : "
                                  << "Cannot create TH2F from source-ME\n";
    }
    return;
  }

  string profileDir = startDir;
  string newProfileMEName = profileMEName;
  string::size_type shiftPos;
  if ( string::npos != (shiftPos = profileMEName.rfind('/')) ) {
    profileDir += "/"+profileMEName.substr(0, shiftPos);
    newProfileMEName.erase(0, shiftPos+1);
  }
  ibooker.setCurrentFolder(profileDir);

  std::unique_ptr<TProfile> profile(hSrc->ProfileX()); // We own the pointer
  profile->SetTitle(profileMETitle.c_str());
  ibooker.bookProfile(profileMEName, profile.get()); // ibooker makes a copy
}


void DQMGenericClient::normalizeToEntries(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const std::string& startDir, const std::string& histName,
					  const std::string& normHistName) 
{
  if ( ! igetter.dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  ibooker.cd();

  ME* element = igetter.get(startDir+"/"+histName);
  ME* normME = igetter.get(startDir+"/"+normHistName);

  if ( !element ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "No such element '" << histName << "' found\n";
    }
    return;
  }

  if ( !normME ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "No such element '" << normHistName << "' found\n";
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

  TH1F* normHist = normME->getTH1F();
  if ( !normHist ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "normalizeToEntries() : "
                                     << "Cannot create TH1F from ME\n";
    }
    return;
  }

  const double entries = normHist->GetEntries();
  if ( entries != 0 ) {
    hist->Scale(1./entries);
  }
  else {
    LogInfo("DQMGenericClient") << "normalizeToEntries() : " 
                                   << "Zero entries in histogram\n";
  }

  return;
}

void DQMGenericClient::makeCumulativeDist(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, const std::string& startDir, const std::string& cdName) 
{
  if ( ! igetter.dirExists(startDir) ) {
    if ( verbose_ >= 2 || (verbose_ == 1 && !isWildcardUsed_) ) {
      LogInfo("DQMGenericClient") << "makeCumulativeDist() : "
                                     << "Cannot find sub-directory " << startDir << endl;
    }
    return;
  }

  ibooker.cd();

  ME* element_cd = igetter.get(startDir+"/"+cdName);

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

      meanME->setBinContent(i, par[1]);
      meanME->setBinError(i, err[1]);
//       meanME->setBinEntries(i, 1.);
//       meanME->setBinError(i,sqrt(err[1]*err[1]+par[1]*par[1]));

      sigmaME->setBinContent(i, par[2]);
      sigmaME->setBinError(i, err[2]);
//       sigmaME->setBinEntries(i, 1.);
//       sigmaME->setBinError(i,sqrt(err[2]*err[2]+par[2]*par[2]));

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

void DQMGenericClient::findAllSubdirectories (DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter, std::string dir, std::set<std::string> * myList,
					      const TString& _pattern = TString("")) {
  TString pattern = _pattern;
  if (!igetter.dirExists(dir)) {
    LogError("DQMGenericClient") << " DQMGenericClient::findAllSubdirectories ==> Missing folder " << dir << " !!!"; 
    return;
  }
  if (pattern != "") {
    if (pattern.Contains(nonPerlWildcard)) pattern.ReplaceAll("*",".*");
    TPRegexp regexp(pattern);
    ibooker.cd(dir);
    vector <string> foundDirs = igetter.getSubdirs();
    for(vector<string>::const_iterator iDir = foundDirs.begin();
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


void DQMGenericClient::generic_eff (TH1* denom, TH1* numer, MonitorElement* efficiencyHist, const int type) {
  for (int iBinX = 1; iBinX < denom->GetNbinsX()+1; iBinX++){
    for (int iBinY = 1; iBinY < denom->GetNbinsY()+1; iBinY++){
      for (int iBinZ = 1; iBinZ < denom->GetNbinsZ()+1; iBinZ++){

        int globalBinNum = denom->GetBin(iBinX, iBinY, iBinZ);
        
        float numerVal = numer->GetBinContent(globalBinNum);
        float denomVal = denom->GetBinContent(globalBinNum);

        float effVal = 0;

        // fake eff is in use
        if (type == 2 ) {          
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
        efficiencyHist->setEfficiencyFlag();
      }
    }
  }

  //efficiencyHist->setMinimum(0.0);
  //efficiencyHist->setMaximum(1.0);
}


/* vim:set ts=2 sts=2 sw=2 expandtab: */

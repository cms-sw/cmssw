#ifndef DQMServices_ClientConfig_DQMGenericClient_H
#define DQMServices_ClientConfig_DQMGenericClient_H

/*
 *  Class:DQMGenericClient 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2013/05/23 16:16:28 $
 *  $Revision: 1.10 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 *  \modified Cesare Calabria - Università & INFN Bari
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <set>
#include <string>
#include <vector>
#include <TH1.h>
#include <RVersion.h>
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)
#include <TEfficiency.h>
#else
#include <TGraphAsymmErrors.h>
#endif

class DQMStore;
class MonitorElement;

class DQMGenericClient : public edm::EDAnalyzer
{
 public:
  DQMGenericClient(const edm::ParameterSet& pset);
  ~DQMGenericClient() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endJob();

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  struct EfficOption
  {
    std::string name, title;
    std::string numerator, denominator;
    int type;
    bool isProfile;
  };

  struct ResolOption
  {
    std::string namePrefix, titlePrefix;
    std::string srcName;
  };

  struct NormOption
  {
    std::string name, normHistName;
  };

  struct CDOption
  {
    std::string name;
  };

  void computeEfficiency(const std::string& startDir, 
                         const std::string& efficMEName, 
                         const std::string& efficMETitle,
                         const std::string& recoMEName, 
                         const std::string& simMEName, 
                         const int type=1,
                         const bool makeProfile = false);
  void computeResolution(const std::string& startDir, 
                         const std::string& fitMEPrefix, const std::string& fitMETitlePrefix, 
                         const std::string& srcMEName);

  void normalizeToEntries(const std::string& startDir, const std::string& histName, const std::string& normHistName);
  void makeCumulativeDist(const std::string& startDir, const std::string& cdName);

  void limitedFit(MonitorElement * srcME, MonitorElement * meanME, MonitorElement * sigmaME, MonitorElement * rmsME);

 private:
  unsigned int verbose_;
  bool isWildcardUsed_;
  bool resLimitedFit_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;

  std::vector<EfficOption> efficOptions_;
  std::vector<ResolOption> resolOptions_;
  std::vector<NormOption> normOptions_;
  std::vector<CDOption> cdOptions_;

  void generic_eff (TH1 * denom, TH1 * numer, MonitorElement * efficiencyHist, const int type=1);

  void findAllSubdirectories (std::string dir, std::set<std::string> * myList, const TString& pattern);

#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,0)

#else
  class TGraphAsymmErrorsWrapper : public TGraphAsymmErrors {
   public:
    std::pair<double, double> efficiency(int numerator, int denominator) {
      double eff, low, high;
      Efficiency(numerator, denominator, 0.683, eff, low, high);
      double error = (eff - low > high - eff) ? eff - low : high - eff;
      return std::pair<double, double>(eff, error);
    }
  };
#endif

};

#endif

/* vim:set ts=2 sts=2 sw=2 expandtab: */

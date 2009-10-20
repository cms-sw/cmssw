#ifndef DQMServices_ClientConfig_DQMGenericClient_H
#define DQMServices_ClientConfig_DQMGenericClient_H

/*
 *  Class:DQMGenericClient 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2009/10/06 11:16:33 $
 *  $Revision: 1.5 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <set>
#include <string>
#include <vector>
#include <boost/tokenizer.hpp>
#include <TH1.h>
#include <TPRegexp.h>

class DQMStore;
class MonitorElement;

typedef boost::escaped_list_separator<char> elsc;

class DQMGenericClient : public edm::EDAnalyzer
{
 public:
  DQMGenericClient(const edm::ParameterSet& pset);
  ~DQMGenericClient() {};

  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) {};
  void endJob();

 /// EndRun
 void endRun(const edm::Run& r, const edm::EventSetup& c);


  void computeEfficiency(const std::string& startDir, 
                         const std::string& efficMEName, const std::string& efficMETitle,
                         const std::string& recoMEName, const std::string& simMEName,const std::string& type="eff");
  void computeResolution(const std::string& startDir, 
                         const std::string& fitMEPrefix, const std::string& fitMETitlePrefix, 
                         const std::string& srcMEName);
  void normalizeToEntries(const std::string& startDir, const std::string& histName);
  void makeCumulativeDist(const std::string& startDir, const std::string& cdName);

  void limitedFit(MonitorElement * srcME, MonitorElement * meanME, MonitorElement * sigmaME);

 private:
  unsigned int verbose_;
  bool isWildcardUsed_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;
  std::vector<std::string> effCmds_, resCmds_, normCmds_, cdCmds_;
  bool resLimitedFit_;

 void generic_eff (TH1 * denom, TH1 * numer, MonitorElement * efficiencyHist, const std::string& type="eff");

 void findAllSubdirectories (std::string dir, std::set<std::string> * myList, TString pattern);

};

#endif

/* vim:set ts=2 sts=2 sw=2 expandtab: */

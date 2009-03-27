#ifndef DQMServices_ClientConfig_DQMGenericClient_H
#define DQMServices_ClientConfig_DQMGenericClient_H

/*
 *  Class:DQMGenericClient 
 *
 *  DQM histogram post processor
 *
 *  $Date: 2008/12/22 08:28:22 $
 *  $Revision: 1.1 $
 *
 *  \author Junghwan Goh - SungKyunKwan University
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <vector>
#include <boost/tokenizer.hpp>

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

  void computeEfficiency(const std::string& startDir, 
                         const std::string& efficMEName, const std::string& efficMETitle,
                         const std::string& recoMEName, const std::string& simMEName,const std::string& type="eff");
  void computeResolution(const std::string &, 
                         const std::string& fitMEPrefix, const std::string& fitMETitlePrefix, 
                         const std::string& srcMEName);

  void limitedFit(MonitorElement * srcME, MonitorElement * meanME, MonitorElement * sigmaME);

 private:
  unsigned int verbose_;
  bool isWildcardUsed_;

  DQMStore* theDQM;
  std::vector<std::string> subDirs_;
  std::string outputFileName_;
  std::vector<std::string> effCmds_, resCmds_;
  bool resLimitedFit_;
};

#endif

/* vim:set ts=2 sts=2 sw=2 expandtab: */

#ifndef DQMSERVICES_COMPONENTS_DQMFILESAVER_H
#define DQMSERVICES_COMPONENTS_DQMFILESAVER_H

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <boost/property_tree/ptree.hpp>
#include <sys/time.h>
#include <string>

namespace evf { class FastMonitoringService; }
namespace saverDetails { struct NoCache {}; }

class DQMStore;
class DQMFileSaver : public edm::global::EDAnalyzer<edm::RunCache<saverDetails::NoCache>, edm::LuminosityBlockCache<saverDetails::NoCache> >
{
public:
  DQMFileSaver(const edm::ParameterSet &ps);


  // used by the JsonWritingTimedPoolOutputModule,
  // fms will be nullptr in such case
  static boost::property_tree::ptree fillJson(
      int run, int lumi, const std::string &dataFilePathName, const std::string transferDestinationStr,
      const std::string mergeTypeStr, evf::FastMonitoringService *fms);

  
protected:
  virtual void beginJob(void);
  virtual std::shared_ptr<saverDetails::NoCache> globalBeginRun(const edm::Run &, const edm::EventSetup &) const;
  virtual std::shared_ptr<saverDetails::NoCache> globalBeginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) const;
  virtual void analyze(edm::StreamID, const edm::Event &e, const edm::EventSetup &) const;
  virtual void globalEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) const;
  virtual void globalEndRun(const edm::Run &, const edm::EventSetup &) const;
  virtual void endJob(void);
  virtual void postForkReacquireResources(unsigned int childIndex, unsigned int numberOfChildren);

public:
  enum Convention
  {
    Online,
    Offline,
    FilterUnit,
  };

  enum FileFormat
  {
    ROOT,
    PB
  };

private:
  void saveForOfflinePB(const std::string &workflow, int run) const;
  void saveForOffline(const std::string &workflow, int run, int lumi) const;

  void saveForOnlinePB(int run, const std::string &suffix) const;
  void saveForOnline(int run, const std::string &suffix, const std::string &rewrite) const;

  void saveForFilterUnit(const std::string& rewrite, int run, int lumi, const FileFormat fileFormat) const;
  void saveJobReport(const std::string &filename) const;

  Convention	convention_;
  FileFormat    fileFormat_;
  std::string	workflow_;
  std::string	producer_;
  std::string   stream_label_;
  std::string	dirName_;
  std::string   child_;
  std::string	filterName_;
  int   version_;
  bool  runIsComplete_;
  bool  enableMultiThread_;
  bool  fakeFilterUnitMode_;

  int		saveByLumiSection_;
  int		saveByRun_;
  bool		saveAtJobEnd_;
  int		saveReference_;
  int		saveReferenceQMin_;
  int		forceRunNumber_;

  std::string	fileBaseName_;
  mutable std::atomic<int> fileUpdate_;

  DQMStore	*dbe_;
  mutable std::atomic<int> nrun_;
  mutable std::atomic<int> nlumi_;

  // needed only for the harvesting step when saving in the endJob
  mutable std::atomic<int> irun_;

  // Services used in DAQ2 (so for FilterUnit case only)
  evf::FastMonitoringService * fms_;

  static const std::string streamPrefix_;
  static const std::string streamSuffix_;
  std::string transferDestination_;
  std::string mergeType_;
};

#endif // DQMSERVICES_COMPONEntS_DQMFILESAVER_H

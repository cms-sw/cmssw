#ifndef DQMSERVICES_COMPONENTS_DQMFILESAVERBASE_H
#define DQMSERVICES_COMPONENTS_DQMFILESAVERBASE_H

#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/GetterOfProducts.h"
#include "FWCore/Framework/interface/ProcessMatch.h"

#include "DataFormats/Histograms/interface/DQMToken.h"

#include <sys/time.h>
#include <string>
#include <mutex>

#include <boost/property_tree/ptree.hpp>

namespace dqm {

  struct NoCache {};

  class DQMFileSaverBase : public edm::global::EDAnalyzer<edm::RunCache<NoCache>, edm::LuminosityBlockCache<NoCache> > {
  public:
    typedef dqm::legacy::MonitorElement MonitorElement;
    typedef dqm::legacy::DQMStore DQMStore;

    DQMFileSaverBase(const edm::ParameterSet &ps);
    ~DQMFileSaverBase() override;

  protected:
    // file name components, in order
    struct FileParameters {
      std::string path_;      //
      std::string producer_;  // DQM or Playback
      int version_;
      std::string tag_;
      long run_;
      long lumi_;
      std::string child_;  // child of a fork
    };

  protected:
    //virtual void beginJob(void) const override final;
    //virtual void endJob(void) const override final;

    std::shared_ptr<NoCache> globalBeginRun(const edm::Run &, const edm::EventSetup &) const final;

    std::shared_ptr<NoCache> globalBeginLuminosityBlock(const edm::LuminosityBlock &,
                                                        const edm::EventSetup &) const final;

    void analyze(edm::StreamID, const edm::Event &e, const edm::EventSetup &) const final;

    void globalEndLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) const final;
    void globalEndRun(const edm::Run &, const edm::EventSetup &) const final;

    // these method (and only these) should be overriden
    // so we need to call all file savers
    virtual void initRun() const {};
    virtual void saveLumi(const FileParameters &fp) const {};
    virtual void saveRun(const FileParameters &fp) const {};

    static const std::string filename(const FileParameters &fp, bool useLumi = false);

    // utilities
    void logFileAction(const std::string &msg, const std::string &fileName) const;
    void saveJobReport(const std::string &filename) const;

    // members
    mutable std::mutex initial_fp_lock_;
    FileParameters initial_fp_;
    int runNumber_;

    edm::GetterOfProducts<DQMToken> lumigetter_;
    edm::GetterOfProducts<DQMToken> rungetter_;

  public:
    static void fillDescription(edm::ParameterSetDescription &d);
  };
}  // namespace dqm

#endif  // DQMSERVICES_COMPONENTS_DQMFILESAVERBASE_H

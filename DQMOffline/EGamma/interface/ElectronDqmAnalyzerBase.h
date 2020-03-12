
#ifndef ElectronDqmAnalyzerBase_h
#define ElectronDqmAnalyzerBase_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <Rtypes.h>
#include <string>
#include <vector>

//DQM
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

namespace edab {
  struct Empty {};
}  // namespace edab
class ElectronDqmAnalyzerBase : public DQMEDAnalyzer {
protected:
  explicit ElectronDqmAnalyzerBase(const edm::ParameterSet &conf);
  ~ElectronDqmAnalyzerBase() override;

  // specific implementation of EDAnalyzer
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // interface to implement in derived classes
  void analyze(const edm::Event &e, const edm::EventSetup &c) override {}

  // utility methods
  int verbosity() { return verbosity_; }

  void setBookPrefix(const std::string &);
  void setBookIndex(short);
  void setBookEfficiencyFlag(const bool &);
  void setBookStatOverflowFlag(const bool &);

  MonitorElement *bookH1(DQMStore::IBooker &,
                         const std::string &name,
                         const std::string &title,
                         int nchX,
                         double lowX,
                         double highX,
                         const std::string &titleX = "",
                         const std::string &titleY = "Events",
                         Option_t *option = "E1 P");

  MonitorElement *bookH1withSumw2(DQMStore::IBooker &,
                                  const std::string &name,
                                  const std::string &title,
                                  int nchX,
                                  double lowX,
                                  double highX,
                                  const std::string &titleX = "",
                                  const std::string &titleY = "Events",
                                  Option_t *option = "E1 P");

  MonitorElement *bookH2(DQMStore::IBooker &,
                         const std::string &name,
                         const std::string &title,
                         int nchX,
                         double lowX,
                         double highX,
                         int nchY,
                         double lowY,
                         double highY,
                         const std::string &titleX = "",
                         const std::string &titleY = "",
                         Option_t *option = "COLZ");

  MonitorElement *bookH2withSumw2(DQMStore::IBooker &,
                                  const std::string &name,
                                  const std::string &title,
                                  int nchX,
                                  double lowX,
                                  double highX,
                                  int nchY,
                                  double lowY,
                                  double highY,
                                  const std::string &titleX = "",
                                  const std::string &titleY = "",
                                  Option_t *option = "COLZ");

  MonitorElement *bookP1(DQMStore::IBooker &,
                         const std::string &name,
                         const std::string &title,
                         int nchX,
                         double lowX,
                         double highX,
                         double lowY,
                         double highY,
                         const std::string &titleX = "",
                         const std::string &titleY = "",
                         Option_t *option = "E1 P");

  MonitorElement *cloneH1(DQMStore::IBooker &iBooker,
                          const std::string &name,
                          MonitorElement *original,
                          const std::string &title = "");

  MonitorElement *cloneH1(DQMStore::IBooker &iBooker,
                          const std::string &name,
                          const std::string &original,
                          const std::string &title = "");

private:
  int verbosity_;
  std::string bookPrefix_;
  short bookIndex_;
  bool bookEfficiencyFlag_ = false;
  bool bookStatOverflowFlag_ = false;
  bool histoNamesReady;
  std::vector<std::string> histoNames_;
  std::string finalStep_;
  std::string inputFile_;
  std::string outputFile_;
  std::string inputInternalPath_;
  std::string outputInternalPath_;

  // utility methods
  std::string newName(const std::string &name);
};

#endif

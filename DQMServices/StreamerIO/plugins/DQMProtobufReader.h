#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"

#include "DQMFileIterator.h"
#include "DQMMonitoringService.h"

namespace dqmservices {

class DQMProtobufReader : public edm::InputSource {
 public:
  explicit DQMProtobufReader(edm::ParameterSet const&,
                             edm::InputSourceDescription const&);
  ~DQMProtobufReader();
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:
  virtual edm::InputSource::ItemType getNextItemType() override;
  virtual std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
  virtual std::shared_ptr<edm::LuminosityBlockAuxiliary>
  readLuminosityBlockAuxiliary_() override;
  virtual void readRun_(edm::RunPrincipal& rpCache) override;
  virtual void readLuminosityBlock_(
      edm::LuminosityBlockPrincipal& lbCache) override;
  virtual void readEvent_(edm::EventPrincipal&) override;

  // actual reading will happen here
  virtual void beginLuminosityBlock(edm::LuminosityBlock& lb) override;

  void logFileAction(char const* msg, char const* fileName) const;
  bool prepareNextFile();

  bool flagSkipFirstLumis_;
  bool flagEndOfRunKills_;
  bool flagDeleteDatFiles_;
  bool flagLoadFiles_;

  std::unique_ptr<double> streamReader_;
  DQMFileIterator fiterator_;
  DQMFileIterator::LumiEntry currentLumi_;

  InputSource::ItemType nextItemType;
};

}  // end of namespace

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSourceMacros.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "FWCore/Sources/interface/PuttableSourceBase.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMFileIterator.h"
#include "DQMMonitoringService.h"

namespace dqmservices {

  class DQMProtobufReader : public edm::PuttableSourceBase {
  public:
    typedef dqm::legacy::MonitorElement MonitorElement;
    typedef dqm::legacy::DQMStore DQMStore;

    explicit DQMProtobufReader(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~DQMProtobufReader() override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void load(DQMStore* store, std::string filename);
    edm::InputSource::ItemType getNextItemType() override;
    std::shared_ptr<edm::RunAuxiliary> readRunAuxiliary_() override;
    std::shared_ptr<edm::LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    void readRun_(edm::RunPrincipal& rpCache) override;
    void readLuminosityBlock_(edm::LuminosityBlockPrincipal& lbCache) override;
    void readEvent_(edm::EventPrincipal&) override;

    // actual reading will happen here
    void beginLuminosityBlock(edm::LuminosityBlock& lb) override;

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

}  // namespace dqmservices

#ifndef DQMServices_StreamerIO_DQMProtobufReader_h
#define DQMServices_StreamerIO_DQMProtobufReader_h

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/PuttableSourceBase.h"

#include "DQMFileIterator.h"

namespace dqmservices {

  class DQMProtobufReader : public edm::PuttableSourceBase {
  public:
    typedef dqm::legacy::MonitorElement MonitorElement;
    typedef dqm::legacy::DQMStore DQMStore;

    explicit DQMProtobufReader(edm::ParameterSet const&, edm::InputSourceDescription const&);
    ~DQMProtobufReader() override = default;

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

    DQMFileIterator fiterator_;
    DQMFileIterator::LumiEntry currentLumi_;

    bool const flagSkipFirstLumis_;
    bool const flagEndOfRunKills_;
    bool const flagDeleteDatFiles_;
    bool const flagLoadFiles_;
  };

}  // namespace dqmservices

#endif  // DQMServices_StreamerIO_DQMProtobufReader_h

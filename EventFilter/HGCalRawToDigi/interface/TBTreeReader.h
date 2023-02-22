#ifndef EventFilter_HGCalRawToDigi_TBTreeReader_h
#define EventFilter_HGCalRawToDigi_TBTreeReader_h

#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulator.h"
#include "TChain.h"

namespace hgcal::econd {
  class TBTreeReader : public Emulator {
  public:
    /// \param[in] tree_name Name of the TB events tree
    /// \param[in] filenames List of filenames to loop on
    /// \param[in] num_channels Channels multiplicity
    explicit TBTreeReader(const EmulatorParameters&,
                          const std::string& tree_name,
                          const std::vector<std::string>& filenames);

    /// Input tree collections
    struct TreeEvent {
      int event, chip, half, channel, adc, adcm, toa, tot, totflag, bxcounter, eventcounter, orbitcounter;
    };
    ECONDEvent next() override;

  private:
    TChain chain_;
    ECONDInputs data_;
    ECONDInputs::const_iterator it_data_;
  };
}  // namespace hgcal::econd

#endif

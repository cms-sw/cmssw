#ifndef EventFilter_HGCalRawToDigi_HGCalModuleTreeReader_h
#define EventFilter_HGCalRawToDigi_HGCalModuleTreeReader_h

#include "EventFilter/HGCalRawToDigi/interface/HGCalECONDEmulator.h"
#include "TChain.h"

namespace hgcal::econd {
  /// Read out a the relevant raw data produced by a module to memory and returns ECON-D frames on request
  /// \note The format is as agreed with system tests convenors so that it can be used in integration/beam tests
  class HGCalModuleTreeReader : public Emulator {
  public:
    /// \param[in] tree_name Name of the TB events tree
    /// \param[in] filenames List of filenames to loop on
    /// \param[in] num_channels Channels multiplicity
    explicit HGCalModuleTreeReader(const EmulatorParameters&,
                                   const std::string& tree_name,
                                   const std::vector<std::string>& filenames);

    /// Input tree collections
    struct HGCModuleTreeEvent {
      UInt_t event, chip;
      Int_t half, bxcounter, eventcounter, orbitcounter, trigtime, trigwidth;
      std::vector<unsigned int>* daqdata{0};
    };
    ECONDInput next() override;

  private:
    TChain chain_;
    ECONDInputColl data_;
    ECONDInputColl::const_iterator it_data_;
  };

}  // namespace hgcal::econd

#endif

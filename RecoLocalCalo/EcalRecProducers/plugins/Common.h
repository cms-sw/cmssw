#ifndef RecoLocalCalo_EcalRecProducers_plugins_Common_h
#define RecoLocalCalo_EcalRecProducers_plugins_Common_h

// a workaround for std::abs not being a constexpr function
namespace ecal {

  // temporary
  namespace mgpa {

    constexpr int adc(uint16_t sample) { return sample & 0xfff; }
    constexpr int gainId(uint16_t sample) { return (sample >> 12) & 0x3; }

  }  // namespace mgpa

}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecProducers_plugins_Common_h

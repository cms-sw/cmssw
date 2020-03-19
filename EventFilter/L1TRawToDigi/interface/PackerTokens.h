#ifndef EventFilter_L1TRawToDigi_PackerTokens_h
#define EventFilter_L1TRawToDigi_PackerTokens_h

namespace edm {
  class ConsumesCollector;
  class ParameterSet;
}  // namespace edm

namespace l1t {
  class PackerTokens {
  public:
    virtual ~PackerTokens() = default;
  };
}  // namespace l1t

#endif

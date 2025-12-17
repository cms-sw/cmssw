#ifndef TrivialSerialisation_Common_interface_SerialiserBase_h
#define TrivialSerialisation_Common_interface_SerialiserBase_h

#include <memory>

#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"

namespace ngt {
  class SerialiserBase {
  public:
    virtual std::unique_ptr<WriterBase> writer() = 0;
    virtual std::unique_ptr<const ReaderBase> reader(const edm::WrapperBase& wrapper) = 0;

    virtual ~SerialiserBase() = default;
  };
}  // namespace ngt

#endif  // TrivialSerialisation_Common_interface_SerialiserBase_h

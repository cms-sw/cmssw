#ifndef TrivialSerialisation_Common_interface_Serialiser_h
#define TrivialSerialisation_Common_interface_Serialiser_h

#include <memory>

#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/SerialiserBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Reader.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/ReaderBase.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/Writer.h"
#include "HeterogeneousCore/TrivialSerialisation/interface/WriterBase.h"

namespace ngt {

  template <typename T>
  class Serialiser : public SerialiserBase {
  public:
    std::unique_ptr<WriterBase> writer() override { return std::make_unique<Writer<T>>(); }

    std::unique_ptr<const ReaderBase> reader(edm::WrapperBase const& wrapper) override {
      edm::Wrapper<T> const& w = dynamic_cast<edm::Wrapper<T> const&>(wrapper);
      return std::make_unique<Reader<T>>(w);
    }
  };

}  // namespace ngt

#endif  // TrivialSerialisation_Common_interface_Serialiser_h

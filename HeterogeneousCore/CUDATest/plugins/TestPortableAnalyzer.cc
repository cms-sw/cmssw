#include <cassert>

#include "CUDADataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace {

  template <typename T>
  class Column {
  public:
    Column(T const* data, size_t size) : data_(data), size_(size) {}

    void print(std::ostream& out) const {
      std::stringstream buffer;
      buffer << "{ ";
      if (size_ > 0) {
        buffer << data_[0];
      }
      if (size_ > 1) {
        buffer << ", " << data_[1];
      }
      if (size_ > 2) {
        buffer << ", " << data_[2];
      }
      if (size_ > 3) {
        buffer << ", ...";
      }
      buffer << '}';
      out << buffer.str();
    }

  private:
    T const* const data_;
    size_t const size_;
  };

  template <typename T>
  std::ostream& operator<<(std::ostream& out, Column<T> const& column) {
    column.print(out);
    return out;
  }
}  // namespace

class TestPortableAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  TestPortableAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")}, token_{consumes(source_)} {}

  void analyze(edm::Event const& event, edm::EventSetup const&) override {
    cudatest::TestHostCollection const& product = event.get(token_);

    auto const& view = product.const_view();
    for (int32_t i = 0; i < view.metadata().size(); ++i) {
      assert(view[i].id() == i);
    }

    edm::LogInfo msg("TestPortableAnalyzer");
    msg << source_.encode() << ".size() = " << view.metadata().size() << '\n';
    msg << "  data @ " << product.buffer().get() << ",\n"
        << "  x    @ " << view.metadata().addressOf_x() << " = " << Column(view.x(), view.metadata().size()) << ",\n"
        << "  y    @ " << view.metadata().addressOf_y() << " = " << Column(view.y(), view.metadata().size()) << ",\n"
        << "  z    @ " << view.metadata().addressOf_z() << " = " << Column(view.z(), view.metadata().size()) << ",\n"
        << "  id   @ " << view.metadata().addressOf_id() << " = " << Column(view.id(), view.metadata().size()) << ",\n"
        << "  r    @ " << view.metadata().addressOf_r() << " = " << view.r() << '\n'
        << "  m    @ " << view.metadata().addressOf_m() << " = { ... {" << view[1].m()(1, Eigen::indexing::all)
        << " } ... } \n";
    msg << std::hex << "  [y - x] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_y()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_x())
        << "  [z - y] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_z()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_y())
        << "  [id - z] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_id()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_z())
        << "  [r - id] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_r()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_id())
        << "  [m - r] = 0x"
        << reinterpret_cast<intptr_t>(view.metadata().addressOf_m()) -
               reinterpret_cast<intptr_t>(view.metadata().addressOf_r());

    const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
    assert(view.r() == 1.);
    for (int32_t i = 0; i < view.metadata().size(); ++i) {
      auto vi = view[i];
      assert(vi.x() == 0.);
      assert(vi.y() == 0.);
      assert(vi.z() == 0.);
      assert(vi.id() == i);
      //assert(vi.m() == matrix * i);
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<cudatest::TestHostCollection> token_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestPortableAnalyzer);

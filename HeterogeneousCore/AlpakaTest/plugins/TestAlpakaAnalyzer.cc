#include <cassert>

#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
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

  template <typename T>
  void checkViewAddresses(T const& view) {
    // columns
    assert(view.metadata().addressOf_x() == view.x());
    assert(view.metadata().addressOf_x() == &view.x(0));
    assert(view.metadata().addressOf_x() == &view[0].x());
    assert(view.metadata().addressOf_y() == view.y());
    assert(view.metadata().addressOf_y() == &view.y(0));
    assert(view.metadata().addressOf_y() == &view[0].y());
    assert(view.metadata().addressOf_z() == view.z());
    assert(view.metadata().addressOf_z() == &view.z(0));
    assert(view.metadata().addressOf_z() == &view[0].z());
    assert(view.metadata().addressOf_id() == view.id());
    assert(view.metadata().addressOf_id() == &view.id(0));
    assert(view.metadata().addressOf_id() == &view[0].id());
    // scalars
    assert(view.metadata().addressOf_r() == &view.r());
    //assert(view.metadata().addressOf_r() == &view.r(0));                  // cannot access a scalar with an index
    //assert(view.metadata().addressOf_r() == &view[0].r());                // cannot access a scalar via a SoA row-like accessor
    // columns of arrays
    assert(view.metadata().addressOf_flags() == view.flags());
    assert(view.metadata().addressOf_flags() == &view.flags(0));
    assert(view.metadata().addressOf_flags() == &view[0].flags());
    // columns of Eigen matrices
    assert(view.metadata().addressOf_m() == view.m());
    assert(view.metadata().addressOf_m() == &view.m(0).coeffRef(0, 0));
    assert(view.metadata().addressOf_m() == &view[0].m().coeffRef(0, 0));
  }

}  // namespace

class TestAlpakaAnalyzer : public edm::global::EDAnalyzer<> {
public:
  TestAlpakaAnalyzer(edm::ParameterSet const& config)
      : source_{config.getParameter<edm::InputTag>("source")},
        token_{consumes(source_)},
        expectSize_{config.getParameter<int>("expectSize")} {
    if (std::string const& eb = config.getParameter<std::string>("expectBackend"); not eb.empty()) {
      expectBackend_ = cms::alpakatools::toBackend(eb);
      backendToken_ = consumes(edm::InputTag(source_.label(), "backend", source_.process()));
    }
  }

  void analyze(edm::StreamID sid, edm::Event const& event, edm::EventSetup const&) const override {
    portabletest::TestHostCollection const& product = event.get(token_);
    auto const& view = product.const_view();
    auto& mview = product.view();
    auto const& cmview = product.view();

    if (expectSize_ >= 0 and expectSize_ != view.metadata().size()) {
      throw cms::Exception("Assert") << "Expected input collection size " << expectSize_ << ", got "
                                     << view.metadata().size();
    }

    {
      edm::LogInfo msg("TestAlpakaAnalyzer");
      msg << source_.encode() << ".size() = " << view.metadata().size() << '\n';
      msg << "  data  @ " << product.buffer().data() << ",\n"
          << "  x     @ " << view.metadata().addressOf_x() << " = " << Column(view.x(), view.metadata().size()) << ",\n"
          << "  y     @ " << view.metadata().addressOf_y() << " = " << Column(view.y(), view.metadata().size()) << ",\n"
          << "  z     @ " << view.metadata().addressOf_z() << " = " << Column(view.z(), view.metadata().size()) << ",\n"
          << "  id    @ " << view.metadata().addressOf_id() << " = " << Column(view.id(), view.metadata().size())
          << ",\n"
          << "  r     @ " << view.metadata().addressOf_r() << " = " << view.r() << '\n'
          << "  flags @ " << view.metadata().addressOf_flags() << " = " << Column(view.flags(), view.metadata().size())
          << ",\n"
          << "  m     @ " << view.metadata().addressOf_m() << " = { ... {" << view[1].m()(1, Eigen::indexing::all)
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
          << "  [flags - r] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_flags()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_r())
          << "  [m - flags] = 0x"
          << reinterpret_cast<intptr_t>(view.metadata().addressOf_m()) -
                 reinterpret_cast<intptr_t>(view.metadata().addressOf_flags());
    }

    checkViewAddresses(view);
    checkViewAddresses(mview);
    checkViewAddresses(cmview);

    const portabletest::Matrix matrix{{1, 2, 3, 4, 5, 6}, {2, 4, 6, 8, 10, 12}, {3, 6, 9, 12, 15, 18}};
    const portabletest::Array flags{{6, 4, 2, 0}};
    assert(view.r() == 1.);
    for (int32_t i = 0; i < view.metadata().size(); ++i) {
      auto vi = view[i];
      assert(vi.x() == 0.);
      assert(vi.y() == 0.);
      assert(vi.z() == 0.);
      assert(vi.id() == i);
      assert(vi.flags() == flags);
      assert(vi.m() == matrix * i);
    }

    if (expectBackend_) {
      auto backend = static_cast<cms::alpakatools::Backend>(event.get(backendToken_));
      if (expectBackend_ != backend) {
        throw cms::Exception("Assert") << "Expected input backend " << cms::alpakatools::toString(*expectBackend_)
                                       << ", got " << cms::alpakatools::toString(backend);
      }
    }
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("source");
    desc.add<int>("expectSize", -1)
        ->setComment("Expected size of the input collection. Values < 0 mean the check is not performed. Default: -1");
    desc.add<std::string>("expectBackend", "")
        ->setComment(
            "Expected backend of the input collection. Empty value means to not perform the check. Default: empty "
            "string");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::InputTag source_;
  const edm::EDGetTokenT<portabletest::TestHostCollection> token_;
  edm::EDGetTokenT<unsigned short> backendToken_;
  std::optional<cms::alpakatools::Backend> expectBackend_;
  const int expectSize_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaAnalyzer);

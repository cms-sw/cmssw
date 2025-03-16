#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/Tutorial/interface/PortableHostTable.h"
#include "HeterogeneousCore/Tutorial/interface/PortableTable.h"
#include "HeterogeneousCore/Tutorial/interface/SoACorrectorRecord.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  using namespace ::tutorial;

  class SoACorrectorESProducer : public ESProducer {
  public:
    SoACorrectorESProducer(edm::ParameterSet const& config) : ESProducer(config) { setWhatProduced(this); }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<PortableHostTable> produce(SoACorrectorRecord const& record) {
      /*
       *        20      25      30      35      40      45      50      55      60      65      70      75      80      85      90      95      100     125     150     175     200     250     300 GeV
       *  0.0   0.6328  0.7606  0.8553  0.9299  0.9676  0.9955  1.0051  1.0049  1.0113  1.0111  1.0177  1.0076  1.0009  0.9943  1.0273  0.9950  1.0168  1.0097  0.9651  1.0432  0.9893  0.9826
       *  1.3   0.6211  0.7394  0.8348  0.8976  0.9382  0.9797  0.9797  0.9954  1.0050  0.9556  1.0050  1.0050  1.0018  0.9766  1.0249  1.0083  0.9922  1.0216  1.0635  0.9585  0.8898  0.9354
       *  2.5   0.9091  0.8621  0.8893  0.8437  0.8047  0.7729  0.8023  0.8167  0.8082  0.8564  0.8206  0.9405  0.8140  0.9129  0.9075  0.8488  0.9258  0.9800  0.9920  1.0000  1.0000  1.0000
       *  3.0
       */

      std::vector<float> pt_axis = {20.f, 25.f, 30.f, 35.f, 40.f,  45.f,  50.f,  55.f,  60.f,  65.f,  70.f, 75.f,
                                    80.f, 85.f, 90.f, 95.f, 100.f, 125.f, 150.f, 175.f, 200.f, 250.f, 300.f};
      std::vector<float> eta_axis = {0.0f, 1.3f, 2.5f, 3.0f};
      std::vector<float> data = {
          // 0.0 ≤ |eta| < 1.3
          0.6328f,  //  20 GeV ≤ pT <  25 GeV
          0.7606f,  //  25 GeV ≤ pT <  30 GeV
          0.8553f,  //  30 GeV ≤ pT <  35 GeV
          0.9299f,  //  35 GeV ≤ pT <  40 GeV
          0.9676f,  //  40 GeV ≤ pT <  45 GeV
          0.9955f,  //  45 GeV ≤ pT <  50 GeV
          1.0051f,  //  50 GeV ≤ pT <  55 GeV
          1.0049f,  //  55 GeV ≤ pT <  60 GeV
          1.0113f,  //  60 GeV ≤ pT <  65 GeV
          1.0111f,  //  65 GeV ≤ pT <  70 GeV
          1.0177f,  //  70 GeV ≤ pT <  75 GeV
          1.0076f,  //  75 GeV ≤ pT <  80 GeV
          1.0009f,  //  80 GeV ≤ pT <  85 GeV
          0.9943f,  //  85 GeV ≤ pT <  90 GeV
          1.0273f,  //  90 GeV ≤ pT <  95 GeV
          0.9950f,  //  95 GeV ≤ pT < 100 GeV
          1.0168f,  // 100 GeV ≤ pT < 125 GeV
          1.0097f,  // 125 GeV ≤ pT < 150 GeV
          0.9651f,  // 150 GeV ≤ pT < 175 GeV
          1.0432f,  // 175 GeV ≤ pT < 200 GeV
          0.9893f,  // 200 GeV ≤ pT < 250 GeV
          0.9826f,  // 250 GeV ≤ pT < 300 GeV
          // 1.3 ≤ |eta| < 2.5
          0.6211f,  //  20 GeV ≤ pT <  25 GeV
          0.7394f,  //  25 GeV ≤ pT <  30 GeV
          0.8348f,  //  30 GeV ≤ pT <  35 GeV
          0.8976f,  //  35 GeV ≤ pT <  40 GeV
          0.9382f,  //  40 GeV ≤ pT <  45 GeV
          0.9797f,  //  45 GeV ≤ pT <  50 GeV
          0.9797f,  //  50 GeV ≤ pT <  55 GeV
          0.9954f,  //  55 GeV ≤ pT <  60 GeV
          1.0050f,  //  60 GeV ≤ pT <  65 GeV
          0.9556f,  //  65 GeV ≤ pT <  70 GeV
          1.0050f,  //  70 GeV ≤ pT <  75 GeV
          1.0050f,  //  75 GeV ≤ pT <  80 GeV
          1.0018f,  //  80 GeV ≤ pT <  85 GeV
          0.9766f,  //  85 GeV ≤ pT <  90 GeV
          1.0249f,  //  90 GeV ≤ pT <  95 GeV
          1.0083f,  //  95 GeV ≤ pT < 100 GeV
          0.9922f,  // 100 GeV ≤ pT < 125 GeV
          1.0216f,  // 125 GeV ≤ pT < 150 GeV
          1.0635f,  // 150 GeV ≤ pT < 175 GeV
          0.9585f,  // 175 GeV ≤ pT < 200 GeV
          0.8898f,  // 200 GeV ≤ pT < 250 GeV
          0.9354f,  // 250 GeV ≤ pT < 300 GeV
          // 2.5 ≤ |eta| < 3.0
          0.9091f,  //  20 GeV ≤ pT <  25 GeV
          0.8621f,  //  25 GeV ≤ pT <  30 GeV
          0.8893f,  //  30 GeV ≤ pT <  35 GeV
          0.8437f,  //  35 GeV ≤ pT <  40 GeV
          0.8047f,  //  40 GeV ≤ pT <  45 GeV
          0.7729f,  //  45 GeV ≤ pT <  50 GeV
          0.8023f,  //  50 GeV ≤ pT <  55 GeV
          0.8167f,  //  55 GeV ≤ pT <  60 GeV
          0.8082f,  //  60 GeV ≤ pT <  65 GeV
          0.8564f,  //  65 GeV ≤ pT <  70 GeV
          0.8206f,  //  70 GeV ≤ pT <  75 GeV
          0.9405f,  //  75 GeV ≤ pT <  80 GeV
          0.8140f,  //  80 GeV ≤ pT <  85 GeV
          0.9129f,  //  85 GeV ≤ pT <  90 GeV
          0.9075f,  //  90 GeV ≤ pT <  95 GeV
          0.8488f,  //  95 GeV ≤ pT < 100 GeV
          0.9258f,  // 100 GeV ≤ pT < 125 GeV
          0.9800f,  // 125 GeV ≤ pT < 150 GeV
          0.9920f,  // 150 GeV ≤ pT < 175 GeV
          1.0000f,  // 175 GeV ≤ pT < 200 GeV
          1.0000f,  // 200 GeV ≤ pT < 250 GeV
          1.0000f,  // 250 GeV ≤ pT < 300 GeV
      };
      auto table =
          std::make_unique<PortableHostTable>(cms::alpakatools::host(), pt_axis.size() - 1, eta_axis.size() - 1);
      table->set_x_axis(std::span<float>(pt_axis.data(), pt_axis.size()));
      table->set_y_axis(std::span<float>(eta_axis.data(), eta_axis.size()));
      table->set_data(std::span<float>(data.data(), data.size()));
      return table;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(tutorial::SoACorrectorESProducer);

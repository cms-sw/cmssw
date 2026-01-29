#ifndef CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHostT_h
#define CondFormats_HcalObjects_interface_HcalRecoParamWithPulseShapeHostT_h

#include "CondFormats/HcalObjects/interface/HcalRecoParamWithPulseShapeSoA.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace hcal {
  template <typename TDev>
  class HcalRecoParamWithPulseShapeT {
  public:
    using RecoParamCollection = PortableCollection<TDev, HcalRecoParamSoA>;
    using PulseShapeCollection = PortableCollection<TDev, HcalPulseShapeSoA>;

    using PulseShapeConstElement = typename PulseShapeCollection::ConstView::const_element;

    class ConstView {
    public:
      using RecoParamConstView = typename RecoParamCollection::ConstView;
      using PulseShapeConstView = typename PulseShapeCollection::ConstView;
      constexpr ConstView(RecoParamConstView recoView, PulseShapeConstView psView)
          : recoParamView_{recoView}, pulseShapeView_{psView} {};

      ALPAKA_FN_ACC PulseShapeConstElement getPulseShape(uint32_t const hashedId) const {
        auto const recoPulseShapeId = recoParamView_[hashedId].ids();
        return pulseShapeView_[recoPulseShapeId];
      }

      constexpr typename RecoParamCollection::ConstView recoParamView() { return recoParamView_; }

    private:
      typename RecoParamCollection::ConstView recoParamView_;
      typename PulseShapeCollection::ConstView pulseShapeView_;
    };

    HcalRecoParamWithPulseShapeT(size_t recoSize, size_t pulseSize, TDev const& dev)
        : recoParam_(dev, recoSize), pulseShape_(dev, pulseSize) {}
    template <typename TQueue, typename = std::enable_if_t<alpaka::isQueue<TQueue>>>
    HcalRecoParamWithPulseShapeT(size_t recoSize, size_t pulseSize, TQueue const& queue)
        : recoParam_(queue, recoSize), pulseShape_(queue, pulseSize) {}
    HcalRecoParamWithPulseShapeT(RecoParamCollection reco, PulseShapeCollection pulse)
        : recoParam_(std::move(reco)), pulseShape_(std::move(pulse)) {}

    const RecoParamCollection& recoParam() const { return recoParam_; }
    const PulseShapeCollection& pulseShape() const { return pulseShape_; }

    typename RecoParamCollection::View recoParamView() { return recoParam_.view(); }
    typename PulseShapeCollection::View pulseShapeView() { return pulseShape_.view(); }

    ConstView const_view() const { return ConstView(recoParam_.view(), pulseShape_.view()); }

  private:
    RecoParamCollection recoParam_;
    PulseShapeCollection pulseShape_;
  };
}  // namespace hcal
#endif

#ifndef RecoPixelVertexing_PixelTrackFitting_plugins_RiemannFitOnGPU_h
#define RecoPixelVertexing_PixelTrackFitting_plugins_RiemannFitOnGPU_h

#include "RecoPixelVertexing/PixelTrackFitting/interface/FitResult.h"
#include "RecoPixelVertexing/PixelTriplets/plugins/pixelTuplesHeterogeneousProduct.h"

namespace siPixelRecHitsHeterogeneousProduct {
   struct HitsOnCPU;
}

namespace Rfit {
  constexpr uint32_t maxNumberOfConcurrentFits() { return 2*1024;}
  constexpr uint32_t stride() { return maxNumberOfConcurrentFits();}
  using Matrix3x4d = Eigen::Matrix<double,3,4>;
  using Map3x4d = Eigen::Map<Matrix3x4d,0,Eigen::Stride<3*stride(),stride()> >;
  using Matrix6x4f = Eigen::Matrix<float,6,4>;
  using Map6x4f = Eigen::Map<Matrix6x4f,0,Eigen::Stride<6*stride(),stride()> >;
  using Map4d = Eigen::Map<Vector4d,0,Eigen::InnerStride<stride()> >;

}


class RiemannFitOnGPU {
public:

   using HitsOnGPU = siPixelRecHitsHeterogeneousProduct::HitsOnGPU;
   using HitsOnCPU = siPixelRecHitsHeterogeneousProduct::HitsOnCPU;

   using TuplesOnGPU = pixelTuplesHeterogeneousProduct::TuplesOnGPU;

   RiemannFitOnGPU() = default;
   ~RiemannFitOnGPU() { deallocateOnGPU();}

   void setBField(double bField) { bField_ = bField;}
   void launchKernels(HitsOnCPU const & hh, uint32_t nhits, uint32_t maxNumberOfTuples, cudaStream_t cudaStream);

   void allocateOnGPU(TuplesOnGPU::Container const * tuples, Rfit::helix_fit * helix_fit_results);
   void deallocateOnGPU();


private:

    static constexpr uint32_t maxNumberOfConcurrentFits_ = Rfit::maxNumberOfConcurrentFits();

    // fowarded
    TuplesOnGPU::Container const * tuples_d = nullptr;
    double bField_;
    Rfit::helix_fit * helix_fit_results_d = nullptr;



   // Riemann Fit internals
   double *hitsGPU_ = nullptr;
   float *hits_geGPU_ = nullptr;
   double *fast_fit_resultsGPU_ = nullptr;
   Rfit::circle_fit *circle_fit_resultsGPU_ = nullptr;

};

#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>
#include "RecoLocalCalo/HGCalRecProducers/plugins/KernelManagerHGCalCellPositions.h"
#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalCellPositionsKernelImpl.cuh"

namespace {  //kernel parameters
  dim3 nb_celpos_;
  constexpr dim3 nt_celpos_(256);
}  // namespace

KernelManagerHGCalCellPositions::KernelManagerHGCalCellPositions(const size_t& nelems) {
  ::nb_celpos_ = (nelems + ::nt_celpos_.x - 1) / ::nt_celpos_.x;
}

void KernelManagerHGCalCellPositions::fill_positions(
    const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* d_conds) {
  fill_positions_from_detids<<<::nb_celpos_, ::nt_celpos_>>>(d_conds);
}

void KernelManagerHGCalCellPositions::test_cell_positions(
    unsigned id, const hgcal_conditions::HeterogeneousHEFCellPositionsConditionsESProduct* d_conds) {
  test<<<::nb_celpos_, ::nt_celpos_>>>(id, d_conds);
}

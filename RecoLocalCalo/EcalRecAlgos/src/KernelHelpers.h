#ifndef RecoLocalCalo_EcalRecAlgos_src_KernelHelpers_h
#define RecoLocalCalo_EcalRecAlgos_src_KernelHelpers_h

namespace ecal {
  namespace multifit {

    __device__ uint32_t hashedIndexEB(uint32_t id);

    __device__ uint32_t hashedIndexEE(uint32_t id);

  }  // namespace multifit
}  // namespace ecal

#endif  // RecoLocalCalo_EcalRecAlgos_src_KernelHelpers_h

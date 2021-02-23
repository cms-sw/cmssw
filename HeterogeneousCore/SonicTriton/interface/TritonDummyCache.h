#ifndef HeterogeneousCore_SonicTriton_TritonDummyCache
#define HeterogeneousCore_SonicTriton_TritonDummyCache

struct TritonDummyCache {};

//Triton modules want to call initializeGlobalCache, but don't want GlobalCache pointer in constructor
//-> override framework function (can't partial specialize function templates)
namespace edm {
  class ParameterSet;
  namespace stream {
    namespace impl {
      template <typename T>
      T* makeStreamModule(edm::ParameterSet const& iPSet, const TritonDummyCache*) {
        return new T(iPSet);
      }
    }  // namespace impl
  }    // namespace stream
}  // namespace edm

#endif

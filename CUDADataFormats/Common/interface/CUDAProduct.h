#ifndef CUDADataFormats_Common_CUDAProduct_h
#define CUDADataFormats_Common_CUDAProduct_h

#include <memory>

#include <cuda/api_wrappers.h>

#include "CUDADataFormats/Common/interface/CUDAProductBase.h"

namespace edm {
  template <typename T> class Wrapper;
}

/**
 * The purpose of this class is to wrap CUDA data to edm::Event in a
 * way which forces correct use of various utilities.
 *
 * The non-default construction has to be done with CUDAScopedContext
 * (in order to properly register the CUDA event).
 *
 * The default constructor is needed only for the ROOT dictionary generation.
 *
 * The CUDA event is in practice needed only for stream-stream
 * synchronization, but someone with long-enough lifetime has to own
 * it. Here is a somewhat natural place. If overhead is too much, we
 * can e.g. make CUDAService own them (creating them on demand) and
 * use them only where synchronization between streams is needed.
 */
template <typename T>
class CUDAProduct: public CUDAProductBase {
public:
  CUDAProduct() = default; // Needed only for ROOT dictionary generation

  CUDAProduct(const CUDAProduct&) = delete;
  CUDAProduct& operator=(const CUDAProduct&) = delete;
  CUDAProduct(CUDAProduct&&) = default;
  CUDAProduct& operator=(CUDAProduct&&) = default;

private:
  friend class CUDAScopedContext;
  friend class edm::Wrapper<CUDAProduct<T>>;

  explicit CUDAProduct(int device, std::shared_ptr<cuda::stream_t<>> stream, T data):
    CUDAProductBase(device, std::move(stream)),
    data_(std::move(data))
  {}

  T data_; //!
};

#endif

// #include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

#include <cstddef>
#include <cstdint>
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCheck.h"
#include "HeterogeneousCore/CUDAUtilities/interface/radixSort.h"
#include <algorithm>

using FLOAT = double;

// A templated unsigned integer type with N bytes
template <int N>
struct uintN;

template <>
struct uintN<8> {
  using type = uint8_t;
};

template <>
struct uintN<16> {
  using type = uint16_t;
};

template <>
struct uintN<32> {
  using type = uint32_t;
};

template <>
struct uintN<64> {
  using type = uint64_t;
};

template <int N>
using uintN_t = typename uintN<N>::type;

// A templated unsigned integer type with the same size as T
template <typename T>
using uintT_t = uintN_t<sizeof(T) * 8>;

// Keep only the `N` most significant bytes of `t`, and set the others to zero
template <int N, typename T, typename SFINAE = std::enable_if_t<N <= sizeof(T)>>
__device__ __host__ T truncate(T const& t) {
  const int shift = 8 * (sizeof(T) - N);
  union {
    T t;
    uintT_t<T> u;
  } c;
  c.t = t;
  c.u = c.u >> shift << shift;
  return c.t;
}

namespace {
  __global__ void testKernel(FLOAT* gpu_input, int* gpu_product, int elements, bool doPrint) {
    size_t firstElement = threadIdx.x + blockIdx.x * blockDim.x;  // This is going to be the track index
    size_t gridSize = blockDim.x * gridDim.x;

    // radix sort works in a single block
    assert(1 == gridDim.x);
    assert(0 == blockIdx.x);
    assert(elements <= 2048);

    __shared__ uint16_t order[2048];
    __shared__ uint16_t sws[2048];
    __shared__ float z[2048];
    __shared__ int iz[2048];
    for (unsigned int itrack = firstElement; itrack < elements; itrack += gridSize) {
      z[itrack] = gpu_input[itrack];
      iz[itrack] = 10000 * gpu_input[itrack];
      // order[itrack] = itrack;
    }
    __syncthreads();
    radixSort<float, 2>(z, order, sws, elements);
    __syncthreads();

    //verify
    for (unsigned int itrack = firstElement; itrack < (elements - 1); itrack += gridSize) {
      auto ntrack = order[itrack];
      auto mtrack = order[itrack + 1];
      assert(truncate<2>(z[ntrack]) <= truncate<2>(z[mtrack]));
    }

    __syncthreads();

    if (doPrint)
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned int itrackO = 0; itrackO < elements; itrackO++) {
          int itrack = order[itrackO];
          printf(
              "Radix sort with %i elements: At position %i, track position at input %i with z at input %f, z fed to "
              "radixSort %f\n",
              elements,
              itrackO,
              itrack,
              gpu_input[itrack],
              z[itrack]);
          gpu_product[itrackO] = itrack;
        }
      }

    __syncthreads();
    radixSort<int, 4>(iz, order, sws, elements);
    __syncthreads();

    for (unsigned int itrack = firstElement; itrack < (elements - 1); itrack += gridSize) {
      auto ntrack = order[itrack];
      auto mtrack = order[itrack + 1];
      assert(iz[ntrack] <= iz[mtrack]);
    }

    if (doPrint)
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (unsigned int itrackO = 0; itrackO < elements; itrackO++) {
          int itrack = order[itrackO];
          printf(
              "Radix sort with %i elements: At position %i, track position at input %i with z at input %f, z fed to "
              "radixSort %d\n",
              elements,
              itrackO,
              itrack,
              gpu_input[itrack],
              iz[itrack]);
          gpu_product[itrackO] = itrack;
        }
      }
  }

  void testWrapper(FLOAT* gpu_input, int* gpu_product, int elements, bool doPrint) {
    auto blockSize = 512;  // somewhat arbitrary
    auto gridSize = 1;     // round up to cover the sample size
    testKernel<<<gridSize, blockSize>>>(gpu_input, gpu_product, elements, doPrint);
    cudaCheck(cudaGetLastError());
  }
}  // namespace

#include "HeterogeneousCore/CUDAUtilities/interface/requireDevices.h"

int main() {
  cms::cudatest::requireDevices();

  FLOAT* gpu_input;
  int* gpu_product;

  int nmax = 4 * 260;
  FLOAT input[nmax] = {
      30.0,         30.0,         -4.4,         -7.1860761642, -6.6870317459, 1.8010582924, 2.2535820007, 2.2666890621,
      2.2677690983, 2.2794606686, 2.2802586555, 2.2821085453,  2.2852313519,  2.2877883911, 2.2946476936, 2.2960267067,
      2.3006286621, 2.3245604038, 2.6755006313, 2.7229132652,  2.783257246,   2.8440306187, 2.9017834663, 2.9252648354,
      2.9254128933, 2.927520752,  2.9422419071, 2.9453969002,  2.9457902908,  2.9465973377, 2.9492356777, 2.9573802948,
      2.9575133324, 2.9575304985, 2.9586606026, 2.9605507851,  2.9622797966,  2.9625515938, 2.9641008377, 2.9646151066,
      2.9676523209, 2.9708273411, 2.974111557,  2.9742531776,  2.9772830009,  2.9877333641, 2.9960610867, 3.013969183,
      3.0187871456, 3.0379793644, 3.0407221317, 3.0415751934,  3.0470511913,  3.0560519695, 3.0592908859, 3.0599737167,
      3.0607066154, 3.0629007816, 3.0632448196, 3.0633215904,  3.0643932819,  3.0645000935, 3.0666446686, 3.068046093,
      3.0697011948, 3.0717656612, 3.0718104839, 3.0718348026,  3.0733406544,  3.0738227367, 3.0738801956, 3.0738828182,
      3.0744686127, 3.0753741264, 3.0758397579, 3.0767207146,  3.0773906708,  3.0778541565, 3.0780284405, 3.0780889988,
      3.0782799721, 3.0789675713, 3.0792205334, 3.0793278217,  3.0795567036,  3.0797944069, 3.0806643963, 3.0809247494,
      3.0815284252, 3.0817306042, 3.0819730759, 3.0820026398,  3.0838682652,  3.084009409,  3.0848178864, 3.0853257179,
      3.0855510235, 3.0856611729, 3.0873703957, 3.0884618759,  3.0891149044,  3.0893011093, 3.0895674229, 3.0901503563,
      3.0903317928, 3.0912668705, 3.0920717716, 3.0954346657,  3.096424818,   3.0995628834, 3.1001036167, 3.1173279285,
      3.1185023785, 3.1195163727, 3.1568386555, 3.1675374508,  3.1676850319,  3.1886672974, 3.3769197464, 3.3821125031,
      3.4780933857, 3.4822063446, 3.4989323616, 3.5076274872,  3.5225863457,  3.5271244049, 3.5298995972, 3.5417425632,
      3.5444457531, 3.5465917587, 3.5473103523, 3.5480232239,  3.5526945591,  3.5531234741, 3.5538012981, 3.5544877052,
      3.5547749996, 3.5549693108, 3.5550665855, 3.5558729172,  3.5560717583,  3.5560848713, 3.5584278107, 3.558681488,
      3.5587313175, 3.5592217445, 3.559384346,  3.5604712963,  3.5634038448,  3.563803196,  3.564593792,  3.5660364628,
      3.5683133602, 3.5696356297, 3.569729805,  3.5740811825,  3.5757565498,  3.5760207176, 3.5760478973, 3.5836098194,
      3.5839796066, 3.5852358341, 3.5901627541, 3.6141786575,  3.6601481438,  3.7187042236, 3.9741659164, 4.4111995697,
      4.5337572098, 4.6292567253, 4.6748633385, 4.6806583405,  4.6868157387,  4.6868577003, 4.6879930496, 4.6888813972,
      4.6910686493, 4.6925001144, 4.6957530975, 4.698094368,   4.6997032166,  4.7017259598, 4.7020640373, 4.7024269104,
      4.7036352158, 4.7038679123, 4.7042069435, 4.7044086456,  4.7044372559,  4.7050771713, 4.7055773735, 4.7060651779,
      4.7062759399, 4.7065420151, 4.70657444,   4.7066287994,  4.7066788673,  4.7067341805, 4.7072944641, 4.7074551582,
      4.7075614929, 4.7075891495, 4.7076044083, 4.7077374458,  4.7080879211,  4.70819664,   4.7086658478, 4.708937645,
      4.7092385292, 4.709479332,  4.7095656395, 4.7100076675,  4.7102108002,  4.7104525566, 4.7105507851, 4.71118927,
      4.7113513947, 4.7115578651, 4.7116270065, 4.7116751671,  4.7117190361,  4.7117333412, 4.7117910385, 4.7119007111,
      4.7120013237, 4.712003231,  4.712044239,  4.7122926712,  4.7135767937,  4.7143669128, 4.7145690918, 4.7148418427,
      4.7149815559, 4.7159647942, 4.7161884308, 4.7177276611,  4.717815876,   4.718059063,  4.7188801765, 4.7190728188,
      4.7199850082, 4.7213058472, 4.7239775658, 4.7243933678,  4.7243990898,  4.7273659706, 4.7294125557, 4.7296204567,
      4.7325615883, 4.7356877327, 4.740146637,  4.742254734,   4.7433848381,  4.7454957962, 4.7462964058, 4.7692604065,
      4.7723139628, 4.774812736,  4.8577151299, 4.890037536};
  for (int i = 0; i < 260; i++) {
    input[i + 260] = -input[i];
    input[i + 2 * 260] = input[i] + 10;
    input[i + 3 * 260] = -input[i] - 10;
  }
  cudaCheck(cudaMalloc(&gpu_input, sizeof(FLOAT) * nmax));
  cudaCheck(cudaMalloc(&gpu_product, sizeof(int) * nmax));
  // copy the input data to the GPU
  cudaCheck(cudaMemcpy(gpu_input, input, sizeof(FLOAT) * nmax, cudaMemcpyHostToDevice));

  for (int k = 2; k <= nmax; k++) {
    std::random_shuffle(input, input + k);
    printf("Test with %d items\n", k);
    // sort  on the GPU
    testWrapper(gpu_input, gpu_product, k, false);
    cudaCheck(cudaDeviceSynchronize());
  }

  // free the GPU memory
  cudaCheck(cudaFree(gpu_input));
  cudaCheck(cudaFree(gpu_product));

  return 0;
}

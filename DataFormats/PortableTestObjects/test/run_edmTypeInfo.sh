#! /bin/bash -e
#
# Run `edmTypeInfo` and compare its output against the expected results.

function can_run() {
  local CMD="$1"

  OUTPUT=$(${CMD})
}

function compare() {
  local CMD="$1"
  local EXPECTED="$2"
  local OUTPUT
  OUTPUT=$(${CMD})

  if [ "${OUTPUT}" != "${EXPECTED}" ]; then
    echo "Error: unexpected output"
    echo "Command:"
    echo "${CMD}"
    echo
    echo "Expected output:"
    echo "${EXPECTED}"
    echo
    echo "Actual output:"
    echo "${OUTPUT}"
    exit 1
  fi

  return 0
}


# Run edmTypeInfo portabletest::TestHostCollection
CMD="edmTypeInfo portabletest::TestHostCollection"
EXPECTED='`portabletest::TestHostCollection` resolves to `PortableHostCollection<portabletest::TestSoALayout<128,false> >`
  with friendly class name `128falseportabletestTestSoALayoutPortableHostCollection`
  with type info `22PortableHostCollectionIN12portabletest13TestSoALayoutILm128ELb0EEEE`'
compare "${CMD}" "${EXPECTED}"

# Run edmTypeInfo ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection
CMD="edmTypeInfo ALPAKA_ACCELERATOR_NAMESPACE::portabletest::TestDeviceCollection"
EXPECTED='`alpaka_serial_sync::portabletest::TestDeviceCollection` does not resolve to any type known to ROOT

`alpaka_cuda_async::portabletest::TestDeviceCollection` resolves to `PortableDeviceCollection<alpaka::DevUniformCudaHipRt<alpaka::ApiCudaRt>,portabletest::TestSoALayout<128,false>,void>`
  with friendly class name `alpakaDevCudaRt128falseportabletestTestSoALayoutvoidPortableDeviceCollection`
  with type info `24PortableDeviceCollectionIN6alpaka19DevUniformCudaHipRtINS0_9ApiCudaRtEEEN12portabletest13TestSoALayoutILm128ELb0EEEvE`

`alpaka_rocm_async::portabletest::TestDeviceCollection` resolves to `PortableDeviceCollection<alpaka::DevUniformCudaHipRt<alpaka::ApiHipRt>,portabletest::TestSoALayout<128,false>,void>`
  with friendly class name `alpakaDevHipRt128falseportabletestTestSoALayoutvoidPortableDeviceCollection`
  with type info `24PortableDeviceCollectionIN6alpaka19DevUniformCudaHipRtINS0_8ApiHipRtEEEN12portabletest13TestSoALayoutILm128ELb0EEEvE`'
compare "${CMD}" "${EXPECTED}"

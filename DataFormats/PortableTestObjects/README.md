## Define the portable SoA-based data formats

Notes:
  - define a full dictionary for `portabletest::TestSoA` and `portabletest::TestHostCollection`
  - do not define a dictionary for `alpaka_serial_sync::portabletest::TestDeviceCollection`,
    because it is the same class as `portabletest::TestHostCollection`;
  - define the dictionary for `alpaka_cuda_async::portabletest::TestDeviceCollection`
    as _transient_ only;
  - the dictionary for `alpaka_cuda_async::portabletest::TestDeviceCollection` should
    be defined in a separate library, to factor out the CUDA dependency.

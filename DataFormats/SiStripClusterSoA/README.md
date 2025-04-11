# SiStripClusterSoA
The `SiStripClustersHost`/`SiStripClustersDevice` is a portable collection storing the SiStrip clusters. This structure is within the `sistrip` namespace to avoid confusion with the corresponding former structure  SiStripCluster CUDA-version (i.e., with same name but in CUDAFormats and stripgpu namespace).

## Run tests
The unit-test consists in populating the structure on host, copying on device and checking back on the host for mismatches. The number of entries is set to the max number of seeds (`kMaxSeedStrips = 200000`), supposedly higher that the number of strips.

To run the tests:
```bash
scram b runtests_SiStripClustersSoA runtests_SiStripClustersSoASerialSync runtests_SiStripClustersSoACudaAsync runtests_SiStripClustersSoAROCmAsync
```

# SONIC TritonClient tests

Test producers `TritonImageProducer` and `TritonGraphProducer` are available.
They generate arbitrary inputs for inference (with ResNet50 or Graph Attention Network, respectively) and print the resulting output.

To run the tests, a local Triton server can be started using Singularity (default, should not require superuser permission)
or Docker (may require superuser permission).
The server can utilize the local CPU (support for AVX instructions required) or a local Nvidia GPU, if one is available.
The default local server address is `0.0.0.0`.

First, the relevant data should be downloaded from Nvidia:
```
./fetch_model.sh
```

Launch a local server (using Singularity with CPU by default):
```
triton -M $CMSSW_BASE/src/HeterogeneousCore/SonicTriton/data/models start
[run test commands]
triton stop
```

## Test commands

Run the image test:
```
cmsRun tritonTest_cfg.py maxEvents=1 producer=TritonImageProducer
```

Run the graph test:
```
cmsRun tritonTest_cfg.py maxEvents=1 producer=TritonGraphProducer
```

## Caveats

* Local CPU server requires support for AVX instructions.
* Multiple users cannot run servers on the same GPU (e.g. on a shared node).

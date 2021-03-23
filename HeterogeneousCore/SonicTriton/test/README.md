# SONIC TritonClient tests

Test modules `TritonImageProducer` and `TritonGraphProducer` (`TritonGraphFilter`, `TritonGraphAnalyzer`) are available.
They generate arbitrary inputs for inference (with ResNet50 or Graph Attention Network, respectively) and print the resulting output.

First, the relevant data for ResNet50 should be downloaded from Nvidia:
```
./fetch_model.sh
```

A local Triton server will be launched automatically when the tests run.
The local server will use Singularity with CPU by default; if a local Nvidia GPU is available, it will be used instead.
(This behavior can also be controlled manually using the "device" argument to [tritonTest_cfg.py](./tritonTest_cfg.py).)

## Test commands

Run the image test:
```
cmsRun tritonTest_cfg.py maxEvents=1 modules=TritonImageProducer
```

Run the graph test:
```
cmsRun tritonTest_cfg.py maxEvents=1 modules=TritonGraphProducer
```

## Caveats

* Local CPU server requires support for AVX instructions.

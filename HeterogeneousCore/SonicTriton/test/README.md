# SONIC TritonClient tests

Test modules `TritonImageProducer`, `TritonIdentityProducer`, and `TritonGraphProducer` (`TritonGraphFilter`, `TritonGraphAnalyzer`) are available.
They generate arbitrary inputs for inference (with Inception/DenseNet, a simple identity model that allows ragged batching, or Graph Attention Network, respectively) and print the resulting output.

First, the relevant data for the image classification networks should be downloaded:
```
./fetch_model.sh
```

A local Triton server will be launched automatically when the tests run.
The local server will use Apptainer with CPU by default; if a local Nvidia GPU is available, it will be used instead.
(This behavior can also be controlled manually using the "device" argument to [tritonTest_cfg.py](./tritonTest_cfg.py).)

## Test commands

Run the image test:
```
cmsRun tritonTest_cfg.py maxEvents=1 modules=TritonImageProducer,TritonImageProducer models=inception_graphdef,densenet_onnx
```

Run the identity test with ragged batching:
```
cmsRun tritonTest_cfg.py maxEvents=1 modules=TritonIdentityProducer models=ragged_io
```

Run the graph test:
```
cmsRun tritonTest_cfg.py maxEvents=1 modules=TritonGraphProducer
```

## Caveats

* Local CPU server requires support for AVX instructions.

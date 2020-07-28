# SONIC TritonClient tests

A test producer `TritonImageProducer` is available.
It generates an arbitrary image for ResNet50 inference and prints the resulting classifications.

To run the tests, a local Triton server can be started using Docker.
(This may require superuser permission.)

First, the relevant data should be downloaded from Nvidia:
```
./fetch_model.sh
```

Execute this Docker command to launch the local server:
```bash
docker run -d --rm --name tritonserver \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v${CMSSW_BASE}/src/HeterogeneousCore/SonicTriton/data/models:/models \
  -v${CMSSW_BASE}/src/HeterogeneousCore/SonicTriton/data/lib:/inputlib \
  -e LD_LIBRARY_PATH="/opt/tritonserver/lib/pytorch:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64" \
  -e LD_PRELOAD="/inputlib/libtorchscatter.so /inputlib/libtorchsparse.so" \
  nvcr.io/nvidia/tritonserver:20.06-v1-py3 tritonserver --model-repository=/models
```

If the machine has Nvidia GPUs, the flag `--gpus all` can be added to the command.
Otherwise, the server will perform inference using the CPU (slower).

To get more debugging information from the server, the flags `--log-verbose=1 --log-error=1 --log-info=1`
can be added to the end of the command.

The default local server address is `0.0.0.0`.

Run the image test:
```
cmsRun tritonTest_cfg.py maxEvents=1 producer=TritonImageProducer
```

Run the graph test:
```
cmsRun tritonTest_cfg.py maxEvents=1 producer=TritonGraphProducer
```

# SONIC TritonClient tests

A test producer `TritonImageProducer` is available.
It generates an arbitrary image for ResNet50 inference and prints the resulting classifications.

To run the tests, a local Triton server can be started using Docker.
(This may require superuser permission.)

First, the relevant data should be downloaded from Nvidia:
```
./fetch_model.sh
```

Execute this Docker command to launch the local server (corresponding to the client version v1.12.0):
```
docker run -d --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v${CMSSW_BASE}/src/HeterogeneousCore/SonicTriton/data/models:/models nvcr.io/nvidia/tritonserver:20.03-py3 trtserver --model-repository=/models
```

If the machine has Nvidia GPUs, the flag `--gpus all` can be added to the command.
Otherwise, the server will perform inference using the CPU (slower).

The default local server address is `0.0.0.0`.

Run the test:
```
cmsRun imageTest.py maxEvents=1
```

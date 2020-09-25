# SONIC TritonClient tests

Test producers `TritonImageProducer` and `TritonGraphProducer` are available.
They generate arbitrary inputs for inference (with ResNet50 or Graph Attention Network, respectively) and print the resulting output.

To run the tests, a local Triton server can be started using Singularity or Docker.
The default local server address is `0.0.0.0`.
(In either case, to get more debugging information from the server, the flags `--log-verbose=1 --log-error=1 --log-info=1`
can be added to the end of the command that includes `tritonserver`.)

First, the relevant data should be downloaded from Nvidia:
```
./fetch_model.sh
```

## Singularity instructions

Using Singularity should not require superuser permissions.

Execute these Singularity commands to start the instance in the background and then activate the server:
```bash
singularity instance start \
  -B /dev/shm:/run/shm -B ${CMSSW_BASE}/src/HeterogeneousCore/SonicTriton/data/models:/models \
  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fastml/triton-torchgeo:20.06-v1-py3-geometric/ triton_server_instance
singularity run instance://triton_server_instance \
  tritonserver --model-repository=/models
```

(Some operating systems may have `/run/shm` rather than `/dev/shm`.)

If the machine has Nvidia GPUs, the flag `--nv` can be added to the command.
Otherwise, the server will perform inference using the CPU (slower).

## Docker instructions

Using Docker may require superuser permissions.

Execute this Docker command to launch the local server:
```bash
docker run -d --name tritonserver \
  --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v${CMSSW_BASE}/src/HeterogeneousCore/SonicTriton/data/models:/models \
  fastml/triton-torchgeo:20.06-v1-py3-geometric tritonserver --model-repository=/models
```

If the machine has Nvidia GPUs, the flag `--gpus all` can be added to the command.
Otherwise, the server will perform inference using the CPU (slower).

## Test commands

Run the image test:
```
cmsRun tritonTest_cfg.py maxEvents=1 producer=TritonImageProducer
```

Run the graph test:
```
cmsRun tritonTest_cfg.py maxEvents=1 producer=TritonGraphProducer
```

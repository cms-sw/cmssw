## PhysicsTools/PyTorchAlpakaTest
Tutorial code and demonstration of a full implementation of a CMSSW pipeline with ML producers running direct inference using the [PyTorchAlpaka](../PyTorchAlpaka) interface. By default pipeline schedules all modules in parallel in 2 threads and 2 streams on 10 events, but can be modified with options provided in [python/options_cff.py](python/options_cff.py)

Example run with default options and `CudaAsync` backend
```sh
cmsRun ${LOCALTOP}/src/PhysicsTools/PyTorchAlpakaTest/test/runPyTorchAlpakaTest.py backend=cuda_async
```

Run with 1000 events, 4 streams, 4 threads, on default backend (`SerialSync`)
```sh
cmsRun ${LOCALTOP}/src/PhysicsTools/PyTorchAlpakaTest/test/runPyTorchAlpakaTest.py numberOfStreams=4 numberOfThreads=4 numberOfEvents=1000
```

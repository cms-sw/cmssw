# MPI Experiment Instructions (CMSSW)

This document explains how to run your MPI-based data exchange experiments using CMSSW and `cmsRun`, including dummy benchmarks, real HLT pipelines, and profiling.

---

## Directory Layout

Make sure you're inside your CMSSW environment:

```bash
cd $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/test_scripts_and_configs/
```

---

## Running Dummy Experiments

### Single Sender & Receiver (no external MPI server)

```bash
cd $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/test_scripts_and_configs/dummy/

mpirun --mca pml ob1 --mca btl vader,tcp,self \
  -np 1 numactl -N 7 env EXPERIMENT_THREADS=8 EXPERIMENT_STREAMS=8 cmsRun dummy_remote_1rec.py : \
  -np 1 numactl -N 6 env EXPERIMENT_THREADS=8 EXPERIMENT_STREAMS=8 cmsRun dummy_local_1send.py
```

---

## Running Experiments via PMIx Server (multi-node or custom networking)

### 1. Start server (in background)

```bash
ompi-server -r server.uri -d
```

### 2. Remote receiver

```bash
mpirun --mca pmix_server_uri file:server.uri -n 1 -bind-to none \
  numactl -N 3 cmsRun hlt_remote.py
```

### 3. Local sender

```bash
mpirun --mca pmix_server_uri file:server.uri -n 1 -bind-to none \
  numactl -N 0 cmsRun hlt_local.py
```

---

## Running Real HLT Pipeline Together

```bash
cd $CMSSW_BASE/src/HeterogeneousCore/MPICore/test/test_scripts_and_configs/real/

mpirun --mca pml ob1 --mca btl vader,tcp,self \
  -np 1 numactl -N 6 env EXPERIMENT_THREADS=4 EXPERIMENT_STREAMS=4 cmsRun hlt_remote.py : \
  -np 1 numactl -N 6 env EXPERIMENT_THREADS=4 EXPERIMENT_STREAMS=4  cmsRun hlt_local.py
```

### Useful Flags

```bash
--mca btl_base_verbose 100 # to have very detailed output about which communication layers MPI chooses
--mca pml ucx # to choose ucx instead of ob1
```

---

## Remote Experiment Over TCP (multi-host)

```bash
cmsenv_mpirun \
  --host gputest-milan-02 -np 1 \
  env OMPI_MCA_btl_tcp_if_include=eno8303 numactl -N 6 cmsRun hlt_remote.py \
  : \
  --host gputest-genoa-02 -np 1 \
  env OMPI_MCA_btl_tcp_if_include=enp34s0f0 numactl -N 6 cmsRun hlt_local.py
```

---

## Remote Over Infiniband (UCX)

```bash
cmsenv_mpirun \
  --host gputest-milan-02 \
  --mca pml ucx \
  --mca ucx_net_devices mlx5_2 \
  -np 1 env UCX_LOG_LEVEL=debug numactl -N 6 cmsRun hlt_remote.py \
  : \
  --host gputest-genoa-02 \
  --mca pml ucx \
  --mca ucx_net_devices mlx5_1 \
  -np 1 numactl -N 6 cmsRun hlt_local.py
```

---

## Profiling Options

Profiler commants are put right before cmsRun command of the pipeline you want to profile

### NVIDIA Nsight Systems (MPI, UCX)

```bash
nsys profile --stats=true --trace=mpi,ucx --output=...
```

### Intel VTune

```bash
vtune -collect hotspots -knob sampling-mode=sw -knob enable-stack-collection=true -r ... -- 
```







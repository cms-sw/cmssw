# MPI tuning notes

MPI related flags or environmental variables that we found useful for
performance tuning, debugging, or to avoid errors or hangs.


## General notes

### Sending GPU buffers to a rank that does not see a GPU
(`CUDA_VISIBLE_DEVICES=`):

When two MPI ranks are on the same machine, and GPUs are made invisible to one
of them (e.g. by setting `CUDA_VISIBLE_DEVICES=`):

- `UCX_TLS=self,rc` on the CPU rank, `self,rc,cuda` on the GPU rank works.
- `UCX_TLS=self,cma,sysv` on the CPU rank, `self,cma,sysv,cuda` on the GPU rank
  does not work [openucx/ucx#9975](https://github.com/openucx/ucx/issues/9975).

When the GPU is made visible to both ranks, but one of them is set via
configuration to use the `cpu` only (`process.options.accelerators = ['cpu']`):

- `UCX_TLS=self,rc,cuda` can be used in both ranks, and the D->H transfers work.

With Open MPI and the `ob1` PML instead of UCX (`--mca pml ob1 --mca btl
self,smcuda`), D->(?)->H transfers work, regardless of the GPU visibility on the
"`H`" side. Throughput looks a few percent lower than with UCX.

## Open MPI

- `--mca pml ucx --mca pml_ucx_tls any --mca pml_ucx_devices any`: use the UCX
  PML and do not hide any TLS or device from it.
- `--mca pml_ucx_progress_iterations 1`: let a blocking call yield on every
  polling iteration instead of every 100th (the default), so that the effect of
  the yield flags below is more noticeable.
- `--mca mpi_yield_when_idle 1`: a blocking call yields the CPU when there is
  nothing to progress, instead of busy-polling ([Open MPI
  docs](https://docs.open-mpi.org/en/v5.0.x/launching-apps/scheduling.html)).
  Can help in CPU-bound scenarios.
- `--mca threads_pthreads_yield_strategy nanosleep`: how a thread yields when
  `mpi_yield_when_idle` is enabled: `sched_yield` (default) or `nanosleep`. Has
  no effect without `mpi_yield_when_idle 1`.
- `--mca threads_pthreads_nanosleep_time 200000`: the duration of the
  `nanosleep` in ns. Longer sleeps can help in very CPU-bound scenarios.
- `--tag-output`: prefix each output line with the rank that printed it.

See `ompi_info --param threads pthreads --level 3` for the two
`threads_pthreads_*` options above.

## MPICH

- `MPIR_CVAR_ENABLE_HEAVY_YIELD=1`: threads waiting in a blocking call yield
  with a `nanosleep` instead of `sched_yield` (the default). The sleep time is
  hardcoded to 1 ns.


## UCX

- `UCX_USE_MT_MUTEX=y`: use a mutex instead of a spinlock for multithreading
  support. Improves the throughput with Open MPI; no effect with MPICH.
- `UCX_RNDV_THRESH=`: `<number>[b|kb|mb|gb]`, `inf`, or `auto`. `inf` to disable
  the rendezvous protocol (forcing host stagigng for every transfer and making
  things slower)
- `UCX_RNDV_SCHEME=`:
  `[auto|get_zcopy|put_zcopy|get_ppln|put_ppln|am|rkey_ptr]`. Can provide small
  performance benefits. Depends a lot on each specific benchmark.
- `UCX_IB_GPU_DIRECT_RDMA=yes`: On paper: use GPUDirect RDMA or fail (do not
  fall back to host staging if GPU RDMA is not available)

See `ucx_info -f` for more information on these options.

## Logging

- `UCX_LOG_LEVEL=info`: print (among other things) the transports that UCX
  actually picks as candidates for each device. `debug` or `trace` (very
  verbose) show what UCX does in real time.
- `UCX_PROTO_INFO=y`: print the protocol selected for each message size and
  transport. Useful to see e.g. why messages above a threshold fail, or for
  which messages the RDMA path gets selected.
- `UCX_LOG_FILE=<file>`: write the UCX logs to a file

## Interaction with the memory allocator

- `MALLOC_CONF=background_thread:true` (jemalloc,
  [TUNING.md](https://github.com/jemalloc/jemalloc/blob/dev/TUNING.md)):
  Delivers a noticeable (O(10%)) throughput improvement, which appears to be
  larger the more CPU-bound the job is.

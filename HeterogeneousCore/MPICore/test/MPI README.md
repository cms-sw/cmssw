# run two jobs on the same machine

#### using OB1 for local communications

```
cmsenv_mpirun \
    -mca orte_base_help_aggregate 0 \
    -mca pml ob1 \
    -mca btl self,vader \
    -H localhost:2 \
    -np 1 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10 \
    : \
    -np 1 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10
```


#### using UCX for local communications:

```
cmsenv_mpirun \
    -mca orte_base_help_aggregate 0 \
    -mca pml ucx \
    -mca pml_ucx_tls any \
    -mca pml_ucx_devices any \
    -mca btl ^openib \
    -H localhost:2 \
    -np 1 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10 \
    : \
    -np 1 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10
```

To select a specific UCX transport, set `UCX_TLS`:
```
    -x UCX_TLS=self,xpmem \
```
```
    -x UCX_TLS=self,sysv \
```
```
    -x UCX_TLS=self,posix \
```
_etc_.

For example
```
cmsenv_mpirun \
    -mca orte_base_help_aggregate 0 \
    -mca pml ucx \
    -mca pml_ucx_tls any \
    -mca pml_ucx_devices any \
    -mca btl ^openib \
    -H localhost:2 \
    -np 1 \
    -x UCX_TLS=self,cma \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10 \
    : \
    -np 1 \
    -x UCX_TLS=self,cma \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10
```

**Note**: the `cma` transport seems to be currently unavailable or brojen.


### using OB1 for network communications
```
cmsenv_mpirun \
    -mca orte_base_help_aggregate 0 \
    -mca pml ob1 \
    -mca btl self,tcp \
    \
    -H gputest-milan-02.cms \
    -np 1 \
    -x OMPI_MCA_oob_tcp_if_include=eno8303 \
    -x OMPI_MCA_btl_tcp_if_include=eno8303 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10 \
    : \
    -H gputest-genoa-02.cms \
    -np 1 \
    -x LD_LIBRARY_PATH \
    -x OMPI_MCA_oob_tcp_if_include=enp34s0f0 \
    -x OMPI_MCA_btl_tcp_if_include=enp34s0f0 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10
```

```
cmsenv_mpirun \
    -mca orte_base_help_aggregate 0 \
    -mca pml ob1 \
    -mca btl self,openib \
    \
    -H gputest-milan-02.cms \
    -np 1 \
    -x OMPI_MCA_oob_tcp_if_include=eno8303 \
    -x OMPI_MCA_btl_openib_if_include=mlx5_3 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10 \
    : \
    -H gputest-genoa-02.cms \
    -np 1 \
    -x LD_LIBRARY_PATH \
    -x OMPI_MCA_oob_tcp_if_include=enp34s0f0 \
    -x OMPI_MCA_btl_openib_if_include=mlx5_4 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10
```
    
### using UCX for network communications
```
cmsenv_mpirun \
    -mca orte_base_help_aggregate 0 \
    -mca pml ucx \
    -mca btl ^openib \
    \
    -H gputest-milan-02.cms \
    -np 1 \
    -x OMPI_MCA_oob_tcp_if_include=eno8303 \
    -x OMPI_MCA_ucx_net_devices=mlx5_3 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10 \
    : \
    -H gputest-genoa-02.cms \
    -np 1 \
    -x OMPI_MCA_oob_tcp_if_include=enp34s0f0 \
    -x OMPI_MCA_ucx_net_devices=mlx5_4 \
    $CMSSW_BASE/test/$SCRAM_ARCH/testMPI -s 1000 -r 12345 -n 10
```


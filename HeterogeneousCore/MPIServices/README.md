This package contains the services related to remote offloading via MPI.

## MPI services

- `MPIService` initializes MPI for a CMSSW job and provides job-wide api for processes
to exchange their names in order to find correspondence between MPI ranks of discovered
processes and their pre-configured names

- `MPIConsistencyChecker` checks
that for all senders/receivers in a configuration there is a correct pair on the other side, 
helping to detect mismatches, that could otherwise cause a distributed job to hang.

### Usage

Both services have to be added in a CMSSW configuration in order to use MPI modules:

```python
process.load("HeterogeneousCore.MPIServices.MPIService_cfi")
process.load("HeterogeneousCore.MPIServices.MPIConsistencyChecker_cfi")
```


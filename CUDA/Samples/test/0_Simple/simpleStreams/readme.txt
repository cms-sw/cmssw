Sample: simpleStreams
Minimum spec: SM 2.0

This sample uses CUDA streams to overlap kernel executions with memory copies between the host and a GPU device.  This sample uses a new CUDA 4.0 feature that supports pinning of generic host memory.  Requires Compute Capability 2.0 or higher.

Key concepts:
Asynchronous Data Transfers
CUDA Streams and Events

Sample: simpleMultiCopy
Minimum spec: SM 2.0

Supported in GPUs with Compute Capability 1.1, overlapping compute with one memcopy is possible from the host system.  For Quadro and Tesla GPUs with Compute Capability 2.0, a second overlapped copy operation in either direction at full speed is possible (PCI-e is symmetric).  This sample illustrates the usage of CUDA streams to achieve overlapping of kernel execution with data copies to and from the device.

Key concepts:
CUDA Streams and Events
Asynchronous Data Transfers
Overlap Compute and Copy
GPU Performance

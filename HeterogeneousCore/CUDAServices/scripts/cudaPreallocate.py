#!/usr/bin/env python

from __future__ import print_function
import re
import sys
import argparse

def main(opts):
    device = []
    host = []

    device_re = re.compile("Device.*allocated new device block.*\((?P<bytes>\d+) bytes")
    host_re = re.compile("Host.*allocated new host block.*\((?P<bytes>\d+) bytes")

    f = open(opts.file)
    for line in f:
        m = device_re.search(line)
        if m:
            device.append(m.group("bytes"))
            continue
        m = host_re.search(line)
        if m:
            host.append(m.group("bytes"))
    f.close()

    print("process.CUDAService.allocator.devicePreallocate = cms.untracked.vuint32(%s)" % ",".join(device))
    print("process.CUDAService.allocator.hostPreallocate = cms.untracked.vuint32(%s)" % ",".join(host))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Extract CUDAService preallocation parameters from a log file.

To use, run the job once with "process.CUDAService.allocator.debug =
True" and direct the output to a file. Then run this script by passing
the file as an argument, and copy the output of this script back to
the configuration file.""")
    parser.add_argument("file", type=str, help="Log file to parse")
    opts = parser.parse_args()
    main(opts)

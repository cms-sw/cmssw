#!/usr/bin/env python3

import argparse
from collections import namedtuple

class Entries:
    common = ["id", "timestamp"]
    Open = namedtuple("Open", common+["filename"])
    common.append("duration")
    Read = namedtuple("Read", common + ["offset", "requested", "actual"])
    Readv = namedtuple("Readv", common + ["requested", "actual", "elements"])
    ReadvElement = namedtuple("ReadvElements", ["index", "offset", "requested"])
    Write = namedtuple("Write", common + ["offset", "requested", "actual"])
    Writev = namedtuple("Writev", common + ["requested", "actual", "elements"])
    WritevElement = namedtuple("WritevElements", ["index", "offset", "requested"])
    Position = namedtuple("Position", common + ["offset", "whence"])
    Prefetch = namedtuple("Prefech", common + ["requested", "elements", "supported"])
    PrefetchElement = namedtuple("PrefetchElements", ["index", "offset", "requested"])
    Resize = namedtuple("Resize", common+["size"])
    Flush = namedtuple("Flush", common)
    Close = namedtuple("Close", common)

    nameToType = dict(
        o = Open,
        r = Read,
        rv = Readv,
        rve = ReadvElement,
        w = Write,
        wv = Writev,
        wve = WritevElement,
        s = Position,
        p = Prefetch,
        pe = PrefetchElement,
        rsz = Resize,
        f = Flush,
        c = Close
    )

    @staticmethod
    def make(name, args):
        def convert(x):
            try:
                return int(x)
            except ValueError:
                if x == "true" or x == "false":
                    return bool(x)
                return x
        return Entries.nameToType.get(name)._make([convert(x) for x in args])

    @staticmethod
    def isVector(obj):
        return isinstance(obj, Entries.Readv) or isinstance(obj, Entries.Writev) or isinstance(obj, Entries.Prefetch)
    @staticmethod
    def isVectorElement(obj):
        return isinstance(obj, Entries.ReadvElement) or isinstance(obj, Entries.WritevElement) or isinstance(obj, Entries.PrefetchElement)

Chunk = namedtuple("Chunk", ("begin", "end"))

def readlog(f):
    ret = []
    vectorOp = None
    vectorOpElements = []
    for line in f:
        line = line.strip().rstrip()
        if len(line) == 0:
            continue
        if line[0] == "#":
            continue
        content = line.split(" ")
        entry = Entries.make(content[0], content[1:])

        # Collect elements of vector operations into the vector operation objects
        if Entries.isVectorElement(entry):
            if vectorOp is None:
                raise Exception("vectorOp should not have been None")
            vectorOpElements.append(entry)
            continue
        if vectorOp is not None:
            if vectorOp.elements != len(vectorOpElements):
                raise Exception(f"Vector operation {vectorOp} should have {vectorOp.elements} elements, but {len(vectorOpElements)} were found from the trace log")
            ret.append(vectorOp._replace(elements = vectorOpElements))
            vectorOp = None
            vectorOpElements = []
        if Entries.isVector(entry):
            if vectorOp is not None:
                raise Exception(f"vectorOp should have been None, was {vectorOp}")
            if len(vectorOpElements) != 0:
                raise Exception("vectorOpElements should have been empty")
            vectorOp = entry
            continue

        # Non-vector entries
        ret.append(entry)
    # last vector op, if there is one
    if vectorOp is not None:
        ret.append(vectorOp._replace(elements = vectorOpElements))

    if len(ret) == 0:
        raise Exception("Trace is empty")
    if not isinstance(ret[-1], Entries.Close):
        raise Exception(f"Last trace entry was {ret[-1].__class__.__name__} instead of Close, the trace is likely incomplete")

    return ret

def format_bytes(num):
    for unit in ["B", "kB", "MB", "GB"]:
        if num < 1024.0:
            return f"{num:3.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} TB"

def format_duration(num, unit):
    if unit == "us":
        units = ["us", "ms"]
    elif unit == "ms":
        units = ["ms"]
    else:
        raise Exception(f"Unknown time unit {unit}")

    for unit in units:
        if num < 1000.0:
            return f"{num:3.2f} {unit}"
        num /= 1000.0

    if num < 60.0:
        return f"{num:.1f} s"

    minutes, seconds = divmod(num, 60)
    if minutes < 60.0:
        return f"{minutes:.0f} min {seconds:.1f} s ({num:.1f} s)"
    hours, minutes = divmod(minutes, 60)
    return f"{hours:.0f} h {minutes:.0f} min {seconds:.1f} s ({num:.1f} s)"

####################
# Read order analysis
####################
def analyzeReadOrder(logEntries):
    read_total = 0
    read_backward = 0
    readv_total = 0
    readv_backward = 0
    prevOffset = 0
    for entry in logEntries:
        if isinstance(entry, Entries.Read):
            read_total += 1
            if entry.offset < prevOffset:
                read_backward += 1
            prevOffset = entry.offset + entry.requested
        elif isinstance(entry, Entries.Readv):
            readv_total += 1
            if entry.elements[0].offset < prevOffset:
                readv_backward += 1
            prevOffset = entry.elements[-1].offset + entry.elements[-1].requested
    print(f"Read order analysis")
    if read_total > 0:
        print(f"Singular reads")
        print(f" All reads {read_total}")
        print(f" Reads with smaller offset than previous {read_backward}")
        print(f" Backward fraction {read_backward/float(read_total)*100} %")
    if readv_total > 0:
        print(f"Vector reads")
        print(f" All reads {readv_total}")
        print(f" Reads with smaller offset than previous {readv_backward}")
        print(f" Backward fraction {readv_backward/float(readv_total)*100} %")

####################
# Read overlaps analysis
####################
def processReadOverlaps(read_chunks):
    """Takes a list of Chunks

    Returns an OverlapResult, that has the following members
    - total_bytes: Total number of bytes read
    - overlap_bytes: Number of bytes that whose reading could theoretically be avoided
    - total_count: Number of singular reads plus number of vector read elements
    - overlap_count: Number of reads that could theoretically be avoided

    The unique amount of bytes read can be obtained as total_bytes - overlap_bytes

    N reads that overlap with each other, total_count is increased by N,
    and overlap_count by N-1.
    """
    # smallest begin first, and among them, largest end first
    read_chunks.sort(key=lambda x: (x.begin, -x.end))

    read_total_bytes = 0
    read_unique_bytes = 0

    prev = read_chunks[0]
    read_total_bytes = prev.end-prev.begin
    read_total_count = 1

    read_unique_bytes = 0
    read_overlap_count = 0

    for chunk in read_chunks[1:]:
        read_total_bytes += chunk.end-chunk.begin
        read_total_count += 1
        if chunk.begin >= prev.end:
            read_unique_bytes += prev.end-prev.begin
            prev = chunk
        else:
            read_overlap_count += 1
            if chunk.end > prev.end:
                prev = Chunk(prev.begin, chunk.end)
    read_unique_bytes += prev.end-prev.begin

    OverlapResult = namedtuple("OverlapResult", ("total_bytes", "overlap_bytes", "total_count", "overlap_count"))
    return OverlapResult(read_total_bytes, read_total_bytes-read_unique_bytes, read_total_count, read_overlap_count)

def analyzeReadOverlaps(logEntries, args):
    read_chunks = []
    for entry in logEntries:
        if isinstance(entry, Entries.Read):
            read_chunks.append(Chunk(entry.offset, entry.offset+entry.requested))
        elif isinstance(entry, Entries.Readv):
            for element in entry.elements:
                read_chunks.append(Chunk(element.offset, element.offset+element.requested))

    result = processReadOverlaps(read_chunks)
    print(f"Analysis of overlapping reads")
    print(f"Total")
    print(f" Number of reads (singular or vector elements) {result.total_count}")
    print(f" Bytes read {format_bytes(result.total_bytes)}")
    print(f"Overlaps")
    print(f" Number of reads that could overlapped with another read {result.overlap_count}")
    print(f"  Fraction of all reads {result.overlap_count/float(result.total_count)*100} %")
    print(f" Bytes read that had been already read {format_bytes(result.overlap_bytes)}")
    print(f"  Fraction of all bytes {result.overlap_bytes/float(result.total_bytes)*100} %")

####################
# Summary
####################
class Counter(object):
    def __init__(self, typ):
        self._type = typ
        self.count = 0
        self.duration = 0
        self.requested = 0
        self.actual = 0
        self.elements = 0

    def type(self):
        return self._type

    def accumulate(self, entry):
        if isinstance(entry, self._type):
            self.count += 1
            for f in ["duration", "requested", "actual"]:
                if hasattr(entry, f):
                    setattr(self, f, getattr(self, f) + getattr(entry, f))
            if hasattr(entry, "elements"):
                self.elements += len(entry.elements)

def print_summary(header, counter):
    if counter.count == 0:
        return

    print(header)
    print(f" Number            {counter.count}")
    if hasattr(counter.type(), "elements"):
        print(f" Elements          {counter.elements}")
    if hasattr(counter.type(), "requested"):
        print(f" Requested         {format_bytes(counter.requested)}")
        if hasattr(counter.type(), "actual"):
            print(f" Actual            {format_bytes(counter.actual)}")

    print(f" Duration          {format_duration(counter.duration, 'us')}")

    if hasattr(counter.type(), "requested"):
        print(f" Average size      {format_bytes(counter.requested/float(counter.count))}")
        print(f" Average bandwidth {format_bytes(counter.requested/(counter.duration/float(1000000)))}/s")

    print(f" Average latency   {format_duration(counter.duration/float(counter.count), 'us')}")


def summary(logEntries):
    quantities = [
        ("Singular reads", Counter(Entries.Read)),
        ("Vector reads", Counter(Entries.Readv)),
        ("Singular writes", Counter(Entries.Write)),
        ("Vector writes", Counter(Entries.Writev)),
        ("Seeks", Counter(Entries.Position)),
        ("Prefetches", Counter(Entries.Prefetch)),
        ("Flushes", Counter(Entries.Flush))
    ]
    for entry in logEntries:
        if isinstance(entry, Entries.Open):
            print(f"Summary for file {entry.filename}")

        for h, q in quantities:
            q.accumulate(entry)

    for h, q in quantities:
        print_summary(h, q)

####################
# Read ranges
####################
def printReadRanges(logEntries):
    for entry in logEntries:
        if isinstance(entry, Entries.Read):
            print(f"# id {entry.id}")
            print(f"{entry.offset} {entry.offset+entry.requested}")
        elif isinstance(entry, Entries.Readv):
            print(f"# id {entry.id}")
            for element in entry.elements:
                print(f"{element.offset} {element.offset+element.requested}")

####################
# Map read ranges
####################
MapChunk = namedtuple("MapChunk", ("begin", "end", "type", "content"))
def searchMapChunk(chunks, offset, size):
    if len(chunks) == 0:
        return []

    offset_end = offset+size
    import bisect
    i = bisect.bisect_left(chunks, MapChunk(offset, 0, "", ""))
    ret = []
    if i > 0 and offset < chunks[i-1].end:
        ret.append(i-1)
    while i < len(chunks) and chunks[i].begin < offset_end:
        ret.append(i)
        i += 1

    return ret

def printMapReadRanges(logEntries, mapFileName):
    chunks = []
    with open(mapFileName) as f:
        import re
        # Extract the offset (At:...), size (N=...), type, and content
        # (with name and possibly title) of an element in
        # TFile::Map("extended") printout
        line_re = re.compile("At:(?P<offset>\d+)\s*N=(?P<size>\d+)\s*(?P<type>\w+).*(?P<content>name:.*$)")
        for line in f:
            m = line_re.search(line)
            if m:
                offset = int(m.group("offset"))
                chunks.append(MapChunk(offset, offset+int(m.group("size")), m.group("type"), m.group("content")))

    for entry in logEntries:
        if isinstance(entry, Entries.Read):
            print(f"# id {entry.id}")
            for i in searchMapChunk(chunks, entry.offset, entry.requested):
                ch = chunks[i]
                print(f"{ch.begin} {ch.end} {ch.type} {ch.content}")
        elif isinstance(entry, Entries.Readv):
            print(f"# id {entry.id}")
            for element in entry.elements:
                for i in searchMapChunk(chunks, element.offset, element.requested):
                    ch = chunks[i]
                    print(f"{ch.begin} {ch.end} {ch.type} {ch.content}")

####################
# Main function
####################
def main(logEntries, args):
    if args.summary:
        summary(logEntries)
        print()
    if args.readOrder:
        analyzeReadOrder(logEntries)
        print()
    if args.readOverlaps:
        analyzeReadOverlaps(logEntries, args)
        print()
    if args.readRanges:
        printReadRanges(logEntries)
    if args.mapReadRanges:
        printMapReadRanges(logEntries, args.mapReadRanges)
    pass

####################
# Unit tests
####################
import unittest
class TestHelper(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(format_bytes(10), "10.00 B")
        self.assertEqual(format_bytes(1023), "1023.00 B")
        self.assertEqual(format_bytes(1024), "1.00 kB")
        self.assertEqual(format_bytes(1024*1023), "1023.00 kB")
        self.assertEqual(format_bytes(1024*1024), "1.00 MB")
        self.assertEqual(format_bytes(1024*1024*1023), "1023.00 MB")
        self.assertEqual(format_bytes(1024*1024*1024), "1.00 GB")

    def test_format_duration(self):
        self.assertEqual(format_duration(10, "us"), "10.00 us")
        self.assertEqual(format_duration(999, "us"), "999.00 us")
        self.assertEqual(format_duration(1000, "us"), "1.00 ms")
        self.assertEqual(format_duration(1, "ms"), "1.00 ms")
        self.assertEqual(format_duration(999, "ms"), "999.00 ms")
        self.assertEqual(format_duration(1000, "ms"), "1.0 s")
        self.assertEqual(format_duration(59*1000, "ms"), "59.0 s")
        self.assertEqual(format_duration(60*1000, "ms"), "1 min 0.0 s (60.0 s)")
        self.assertEqual(format_duration(59*60*1000, "ms"), "59 min 0.0 s (3540.0 s)")
        self.assertEqual(format_duration(60*60*1000, "ms"), "1 h 0 min 0.0 s (3600.0 s)")
        self.assertEqual(format_duration(90*60*1000, "ms"), "1 h 30 min 0.0 s (5400.0 s)")
        self.assertEqual(format_duration(90*60*1000+345, "ms"), "1 h 30 min 0.3 s (5400.3 s)")

    def test_processReadOverlaps(self):
        chunks = [
            Chunk(0, 5),
            Chunk(5, 10),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 10)
        self.assertEqual(result.overlap_bytes, 0)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.overlap_count, 0)

        chunks = [
            Chunk(0, 10),
            Chunk(5, 10),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 15)
        self.assertEqual(result.overlap_bytes, 5)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.overlap_count, 1)

        chunks = [
            Chunk(0, 10),
            Chunk(5, 10),
            Chunk(0, 5),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 20)
        self.assertEqual(result.overlap_bytes, 10)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.overlap_count, 2)

        chunks = [
            Chunk(0, 10),
            Chunk(5, 15),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 20)
        self.assertEqual(result.overlap_bytes, 5)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.overlap_count, 1)

        chunks = [
            Chunk(0, 5),
            Chunk(2, 10),
            Chunk(9, 12),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 16)
        self.assertEqual(result.overlap_bytes, 4)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.overlap_count, 2)

        chunks = [
            Chunk(0, 5),
            Chunk(2, 10),
            Chunk(9, 12),
            Chunk(5, 7),
            Chunk(12, 13),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 19)
        self.assertEqual(result.overlap_bytes, 6)
        self.assertEqual(result.total_count, 5)
        self.assertEqual(result.overlap_count, 3)

        chunks = [
            Chunk(2, 4),
            Chunk(6, 8),
            Chunk(10, 12),
            Chunk(0, 20),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 26)
        self.assertEqual(result.overlap_bytes, 6)
        self.assertEqual(result.total_count, 4)
        self.assertEqual(result.overlap_count, 3)

        chunks = [
            Chunk(0, 20),
            Chunk(19, 21),
            Chunk(20, 25),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 27)
        self.assertEqual(result.overlap_bytes, 2)
        self.assertEqual(result.total_count, 3)
        # Value 2 here is debatable
        self.assertEqual(result.overlap_count, 2)

    def test_searchMapChunk(self):
        self.assertEqual(searchMapChunk([], 0, 10), [])

        chunks = []
        for i in range(0, 100):
            chunks.append(MapChunk(i*100, i*100+50, "", i))
        self.assertEqual(searchMapChunk(chunks, 0, 10), [0])
        self.assertEqual(searchMapChunk(chunks, 0, 50), [0])
        self.assertEqual(searchMapChunk(chunks, 0, 100), [0])
        self.assertEqual(searchMapChunk(chunks, 10, 50), [0])
        self.assertEqual(searchMapChunk(chunks, 10, 90), [0])
        self.assertEqual(searchMapChunk(chunks, 9900, 50), [99])
        self.assertEqual(searchMapChunk(chunks, 9900, 100), [99])
        self.assertEqual(searchMapChunk(chunks, 9900, 100), [99])

        self.assertEqual(searchMapChunk(chunks, -10, 5), [])
        self.assertEqual(searchMapChunk(chunks, 50, 40), [])
        self.assertEqual(searchMapChunk(chunks, 50, 50), [])
        self.assertEqual(searchMapChunk(chunks, 9950, 10), [])

        self.assertEqual(searchMapChunk(chunks, 0, 200), [0, 1])
        self.assertEqual(searchMapChunk(chunks, 49, 101-49), [0, 1])
        self.assertEqual(searchMapChunk(chunks, 149, 301-149), [1, 2, 3])

def test():
    import sys
    unittest.main(argv=sys.argv[:1])

####################
# Command line arguments
####################
def printHelp():
    return """The storage traces can be obtained by adding StorageTracerProxy to
the storage proxies of the TFileAdaptor Service, for example as
----
process.add_(cms.Service("TFileAdaptor",
    storageProxies = cms.untracked.VPSet(
        cms.PSet(type = cms.untracked.string("StorageTracerProxy"))
))
----
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze storage trace",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                    epilog=printHelp())
    parser.add_argument("filename", type=str, nargs="?", default=None, help="file to process")
    parser.add_argument("--summary", action="store_true", help="Print high-level summary of storage operations")
    parser.add_argument("--readOrder", action="store_true", help="Analyze ordering of reads")
    parser.add_argument("--readOverlaps", action="store_true", help="Analyze overlaps of reads")
    parser.add_argument("--readRanges", action="store_true", help="Print offset ranges of each read element")
    parser.add_argument("--mapReadRanges", type=str, default=None, help="Like --readRanges, but uses the output of TFile::Map() to map the file regions to TFile content. The argument should be a file containing the output of 'edmFileUtil --map'.")
    parser.add_argument("--test", action="store_true", help="Run internal tests")

    args = parser.parse_args()
    if args.test:
        test()
    else:
        if args.readRanges and args.mapReadRanges:
            parser.error("Only one of --readRanges and --mapReadRanges can be given")
        if args.filename is None:
            parser.error("filename argument is missing")
        with open(args.filename) as f:
            logEntries = readlog(f)
        main(logEntries, args)

#!/usr/bin/env python3

from enum import Enum
import argparse
from collections import namedtuple

class Entries:
    common = ["id", "timestamp"]
    Open = namedtuple("Open", common+["filename", "traceid"])
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
ChunkWithTime = namedtuple("ChunkWithTime", ("begin", "end", "begintime", "endtime"))

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
class OverlapType(Enum):
    UNIQUE = "unique"
    PARTIAL_OVERLAP = "partial overlap"
    FULL_OVERLAP = "full overlap"

overlapColors = {
    OverlapType.UNIQUE: "#276827",
    OverlapType.PARTIAL_OVERLAP: "#C6BA13",
    OverlapType.FULL_OVERLAP: "#FF4400"
}

def addAndMergeRange(chunk, seen_ranges):
    """Add chunk to seen_ranges, merging with any overlapping ranges

    seen_ranges is a list of (start, end) tuples, sorted by start and properly merged.
    The function modifies seen_ranges in-place by adding the chunk and merging
    it with any overlapping existing ranges.

    Returns:
        OverlapResult with:
        - overlap_type: OverlapType.UNIQUE/PARTIAL_OVERLAP/FULL_OVERLAP
        - overlap_bytes: number of bytes in chunk that overlapped with existing ranges
    """
    OverlapResult = namedtuple("OverlapResult", ("overlap_type", "overlap_bytes"))
    if len(seen_ranges) == 0:
        seen_ranges.append((chunk.begin, chunk.end))
        return OverlapResult(OverlapType.UNIQUE, 0)

    import bisect

    # Find the first range that could potentially overlap with chunk
    # We need to find ranges where range.end > chunk.begin
    # Since ranges are sorted by start, we use bisect to find insertion point
    left_idx = bisect.bisect_left(seen_ranges, (chunk.begin, 0))

    # Check if chunk overlaps with the previous range (if any)
    if left_idx > 0 and seen_ranges[left_idx - 1][1] > chunk.begin:
        left_idx -= 1

    # Find the rightmost range that overlaps with chunk
    # We need ranges where range.start < chunk.end
    right_idx = left_idx
    while right_idx < len(seen_ranges) and seen_ranges[right_idx][0] < chunk.end:
        right_idx += 1

    # No overlapping ranges found
    if left_idx == right_idx:
        # Insert chunk at the correct position
        seen_ranges.insert(left_idx, (chunk.begin, chunk.end))
        return OverlapResult(OverlapType.UNIQUE, 0)

    # Calculate overlapping bytes
    overlap_bytes = 0
    for i in range(left_idx, right_idx):
        start, end = seen_ranges[i]
        # Find intersection of chunk and current range
        intersection_start = max(chunk.begin, start)
        intersection_end = min(chunk.end, end)
        if intersection_start < intersection_end:
            overlap_bytes += intersection_end - intersection_start

    # Check if chunk is fully contained within a single range
    for i in range(left_idx, right_idx):
        start, end = seen_ranges[i]
        if chunk.begin >= start and chunk.end <= end:
            # Chunk is fully contained, no modification needed
            return OverlapResult(OverlapType.FULL_OVERLAP, chunk.end - chunk.begin)

    # Partial overlap: merge chunk with all overlapping ranges
    merge_start = chunk.begin
    merge_end = chunk.end

    # Extend merge bounds to include all overlapping ranges
    for i in range(left_idx, right_idx):
        start, end = seen_ranges[i]
        merge_start = min(merge_start, start)
        merge_end = max(merge_end, end)

    # Remove all overlapping ranges (in reverse order to maintain indices)
    for i in range(right_idx - 1, left_idx - 1, -1):
        del seen_ranges[i]

    # Insert the merged range at the correct position
    seen_ranges.insert(left_idx, (merge_start, merge_end))

    return OverlapResult(OverlapType.PARTIAL_OVERLAP, overlap_bytes)


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
    read_total_bytes = 0

    read_total_bytes = 0
    read_total_count = 0

    read_overlap_bytes = 0
    read_full_overlap_count = 0
    read_partial_overlap_count = 0

    seen_ranges = []
    for chunk in read_chunks:
        read_total_bytes += chunk.end-chunk.begin
        read_total_count += 1

        ret = addAndMergeRange(chunk, seen_ranges)
        read_overlap_bytes += ret.overlap_bytes
        if ret.overlap_type == OverlapType.UNIQUE:
            pass
        elif ret.overlap_type == OverlapType.PARTIAL_OVERLAP:
            read_partial_overlap_count += 1
        elif ret.overlap_type == OverlapType.FULL_OVERLAP:
            read_full_overlap_count += 1

    OverlapSummary = namedtuple("OverlapSummary", ("total_bytes", "overlap_bytes", "total_count", "full_overlap_count", "partial_overlap_count"))
    return OverlapSummary(read_total_bytes, read_overlap_bytes, read_total_count, read_full_overlap_count, read_partial_overlap_count)

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
    print(f" Number of reads that partially overlapped with an earlier read {result.partial_overlap_count}")
    print(f"  Fraction of all reads {result.partial_overlap_count/float(result.total_count)*100} %")
    print(f" Number of reads that fully overlapped with an earlier read {result.full_overlap_count}")
    print(f"  Fraction of all reads {result.full_overlap_count/float(result.total_count)*100} %")
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
        line_re = re.compile(r"At:(?P<offset>\d+)\s*N=(?P<size>\d+)\s*(?P<type>\w+).*(?P<content>name:.*$)")
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

#####################
# Plot read patterns
#####################
def plotReadPatterns(logEntries, outputFileName):
    if len(logEntries) == 0:
        return

    beginTime = logEntries[0].timestamp

    def makeChunkWithTime(offset, requested, timestamp, duration):
        timestamp = (timestamp-beginTime) / 1000.0 # convert ms to s
        duration = duration / 1_000_000.0 # convert us to s
        return ChunkWithTime(offset, offset+requested, timestamp, timestamp+duration)

    read_chunks = []
    for entry in logEntries:
        if isinstance(entry, Entries.Read):
            read_chunks.append(makeChunkWithTime(entry.offset, entry.requested, entry.timestamp, entry.duration))
        elif isinstance(entry, Entries.Readv):
            for element in entry.elements:
                read_chunks.append(makeChunkWithTime(element.offset, element.requested, entry.timestamp, entry.duration))

    if len(read_chunks) == 0:
        return

    import matplotlib.pyplot as plt

    # Create a plot with read patterns
    fig, ax = plt.subplots(figsize=(10, 6))

    seen_ranges = []
    for chunk in read_chunks:
        ret = addAndMergeRange(chunk, seen_ranges)
        ax.plot([chunk.begintime, chunk.endtime], [chunk.begin, chunk.end], color=overlapColors[ret.overlap_type], alpha=0.5)

    ax.set_xlabel('Time (from job start) (s)')
    ax.set_ylabel('Offset (B)')
    ax.set_title('Read operations')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(outputFileName)
    print(f"Read pattern plot saved to {outputFileName}")


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
    if args.plotReads:
        plotReadPatterns(logEntries, args.plotReads)
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
        self.assertEqual(result.full_overlap_count, 0)
        self.assertEqual(result.partial_overlap_count, 0)

        chunks = [
            Chunk(0, 10),
            Chunk(5, 10),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 15)
        self.assertEqual(result.overlap_bytes, 5)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.full_overlap_count, 1)
        self.assertEqual(result.partial_overlap_count, 0)

        chunks = [
            Chunk(0, 10),
            Chunk(5, 10),
            Chunk(0, 5),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 20)
        self.assertEqual(result.overlap_bytes, 10)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.full_overlap_count, 2)
        self.assertEqual(result.partial_overlap_count, 0)

        chunks = [
            Chunk(0, 10),
            Chunk(5, 15),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 20)
        self.assertEqual(result.overlap_bytes, 5)
        self.assertEqual(result.total_count, 2)
        self.assertEqual(result.full_overlap_count, 0)
        self.assertEqual(result.partial_overlap_count, 1)

        chunks = [
            Chunk(0, 5),
            Chunk(2, 10),
            Chunk(9, 12),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 16)
        self.assertEqual(result.overlap_bytes, 4)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.full_overlap_count, 0)
        self.assertEqual(result.partial_overlap_count, 2)

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
        self.assertEqual(result.full_overlap_count, 1)
        self.assertEqual(result.partial_overlap_count, 2)

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
        self.assertEqual(result.full_overlap_count, 0)
        self.assertEqual(result.partial_overlap_count, 1)

        chunks = [
            Chunk(0, 20),
            Chunk(19, 21),
            Chunk(20, 25),
        ]
        result = processReadOverlaps(chunks)
        self.assertEqual(result.total_bytes, 27)
        self.assertEqual(result.overlap_bytes, 2)
        self.assertEqual(result.total_count, 3)
        self.assertEqual(result.full_overlap_count, 0)
        self.assertEqual(result.partial_overlap_count, 2)

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

    def test_addAndMergeRange(self):
        # Test with empty seen_ranges
        seen_ranges = []
        result = addAndMergeRange(Chunk(0, 10), seen_ranges)
        self.assertEqual(result.overlap_type, OverlapType.UNIQUE)
        self.assertEqual(result.overlap_bytes, 0)
        self.assertEqual(seen_ranges, [(0, 10)])

        # Test UNIQUE cases - no overlap
        seen_ranges = [(10, 20), (30, 40), (50, 60)]
        original_ranges = seen_ranges.copy()

        # Before all ranges
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(0, 5), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.UNIQUE)
        self.assertEqual(result.overlap_bytes, 0)
        self.assertEqual(seen_ranges_copy, [(0, 5), (10, 20), (30, 40), (50, 60)])

        # Between ranges
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(25, 30), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.UNIQUE)
        self.assertEqual(result.overlap_bytes, 0)
        self.assertEqual(seen_ranges_copy, [(10, 20), (25, 30), (30, 40), (50, 60)])

        # After all ranges
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(65, 70), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.UNIQUE)
        self.assertEqual(result.overlap_bytes, 0)
        self.assertEqual(seen_ranges_copy, [(10, 20), (30, 40), (50, 60), (65, 70)])

        # Adjacent but not overlapping - should remain separate
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(0, 10), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.UNIQUE)
        self.assertEqual(result.overlap_bytes, 0)
        self.assertEqual(seen_ranges_copy, [(0, 10), (10, 20), (30, 40), (50, 60)])

        # Test FULL_OVERLAP cases - chunk completely contained within a seen range
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(12, 18), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.FULL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 6)  # 18 - 12 = 6 bytes fully overlapped
        self.assertEqual(seen_ranges_copy, original_ranges)  # No change expected

        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(10, 20), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.FULL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 10)  # 20 - 10 = 10 bytes fully overlapped
        self.assertEqual(seen_ranges_copy, original_ranges)  # No change expected

        # Test PARTIAL_OVERLAP cases - chunk partially overlaps with seen ranges

        # Overlaps beginning of first range
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(5, 15), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 5)  # 15 - 10 = 5 bytes overlapped with (10, 20)
        self.assertEqual(seen_ranges_copy, [(5, 20), (30, 40), (50, 60)])

        # Overlaps end of first range
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(15, 25), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 5)  # 20 - 15 = 5 bytes overlapped with (10, 20)
        self.assertEqual(seen_ranges_copy, [(10, 25), (30, 40), (50, 60)])

        # Spans multiple ranges
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(15, 35), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 10)  # (20-15) + (35-30) = 5 + 5 = 10 bytes
        self.assertEqual(seen_ranges_copy, [(10, 40), (50, 60)])

        # Encompasses a range
        seen_ranges_copy = seen_ranges.copy()
        result = addAndMergeRange(Chunk(5, 45), seen_ranges_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 20)  # (20-10) + (40-30) = 10 + 10 = 20 bytes
        self.assertEqual(seen_ranges_copy, [(5, 45), (50, 60)])

        # Test edge cases with single range
        single_range = [(20, 30)]

        # Overlaps beginning
        single_range_copy = single_range.copy()
        result = addAndMergeRange(Chunk(10, 25), single_range_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 5)  # 25 - 20 = 5 bytes overlapped
        self.assertEqual(single_range_copy, [(10, 30)])

        # Overlaps end
        single_range_copy = single_range.copy()
        result = addAndMergeRange(Chunk(25, 35), single_range_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 5)  # 30 - 25 = 5 bytes overlapped
        self.assertEqual(single_range_copy, [(20, 35)])

        # Fully contained
        single_range_copy = single_range.copy()
        result = addAndMergeRange(Chunk(22, 28), single_range_copy)
        self.assertEqual(result.overlap_type, OverlapType.FULL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 6)  # 28 - 22 = 6 bytes fully overlapped
        self.assertEqual(single_range_copy, [(20, 30)])  # No change

        # Encompasses the range
        single_range_copy = single_range.copy()
        result = addAndMergeRange(Chunk(15, 35), single_range_copy)
        self.assertEqual(result.overlap_type, OverlapType.PARTIAL_OVERLAP)
        self.assertEqual(result.overlap_bytes, 10)  # 30 - 20 = 10 bytes overlapped
        self.assertEqual(single_range_copy, [(15, 35)])

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
    parser.add_argument("--plotReads", type=str, default=None, help="Generate a plot of the read patterns. The argument should be the name of the output PDF file.")
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

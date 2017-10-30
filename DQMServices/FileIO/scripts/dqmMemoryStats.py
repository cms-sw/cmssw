#!/bin/env python

import sys

from DQMServices.FileIO.DQM import DQMReader
from collections import namedtuple

HistogramEntry = namedtuple('HistogramEntry', ['type', 'bin_size', 'bin_count', 'extra', 'total_bytes'])

class HistogramAnalyzer(object):
    BIN_SIZE = {
        'TH1S': 2, 'TH2S': 2,
        'TH1F': 4, 'TH2F': 4,
        'TH1D': 8, 'TH2D': 8,
        'TH3F': 4,
        'TProfile': 8, 'TProfile2D': 8,
    }

    def __init__(self):
        self._all = {}

    def analyze(self, fn, obj):
        name = fn.split("/")[-1]

        if hasattr(obj, 'ClassName'):
            # this is a root type
            t = str(obj.ClassName())
            bin_size = self.BIN_SIZE.get(t, None)
            if bin_size is None:
                sys.stderr.write("warning: unknown root type: %s\n" % t)
                sys.stderr.flush()
                bin_size = 8

            bin_count = obj.GetNcells()
            extra = len(fn)
            total_bytes = bin_count * bin_size + extra

            self._all[fn] = HistogramEntry(t, bin_size, bin_count, extra, total_bytes)
        else:
            t = str(type(obj))
            bin_count, bin_size, extra = 0, 0, len(str(obj)) + len(fn)
            total_bytes = bin_count * bin_size + extra

            self._all[fn] = HistogramEntry(t, bin_size, bin_count, extra, total_bytes)

    def group(self, level):
        group_stats = {}

        for k, v in self._all.items():
            group_key = "/".join(k.split("/")[:level])

            current = group_stats.get(group_key, 0)
            group_stats[group_key] = current + v.total_bytes

        return group_stats

    def difference(self, ref):
        results = HistogramAnalyzer()
        results._all = dict(self._all)

        zero = HistogramEntry("null", 0, 0, 0, 0)
        def cmp(a, b):
            return HistogramEntry(b.type, b.bin_size,
                a.bin_count - b.bin_count,
                a.extra - b.extra,
                a.total_bytes - b.total_bytes )

        for k, refv in ref._all.items():
            results._all[k] = cmp(self._all.get(k, zero), refv)

        return results


def kibisize(num):
    for prefix in ['KiB','MiB','GiB']:
        num /= 1024.0

        if num < 1024.0:
            return "%.2f %s" % (num, prefix)

def displayDirectoryStatistics(stats, args):
    group_stats = stats.group(args.depth)

    cutoff, display = args.cutoff * 1024, args.display

    as_list = [(v, k, ) for (k, v) in group_stats.items()]
    as_list.sort(reverse=True, key=lambda (v, k): abs(v))

    if cutoff is not None:
        as_list = filter(lambda (v, k): abs(v) > cutoff, as_list)

    if display is not None:
        as_list = as_list[:display]

    if args.human:
        print "*" * 80
        print (" DQM level %d folder breakdown " % args.depth).center(80, "*")
        if cutoff:
            print ("* Size cutoff: %s" % kibisize(cutoff)).ljust(79) + "*"
        if display:
            print ("* Showing top %d entries." % display).ljust(79) + "*"
        print "*" * 80

    for v, k in as_list:
        if args.human:
            print kibisize(v).ljust(16, " "), k
        else:
            print v, k  

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "Input DQM ROOT file")
    parser.add_argument("-r", "--ref", help = "Reference DQM ROOT file (to diff)")
    parser.add_argument("--summary", help = "Dump summary", action = "store_true")
    parser.add_argument("-x", "--human", help = "Human readable output.", action = "store_true")
    parser.add_argument("-n", "--display", help = "Max entries to display in --summary.", type = int, default = None)
    parser.add_argument("-c", "--cutoff", help = "Max cutoff to display in --summary.", type = float, default = 512, metavar="KiB")
    parser.add_argument("-d", "--depth", help = "Folder depth in --summary.", type = int, default = 2)

    args = parser.parse_args()

    stats = HistogramAnalyzer()
    reader = DQMReader(args.input)
    for (fn, v) in reader.read_objects():
        stats.analyze(fn, v)
    reader.close()

    if args.ref:
        reader = DQMReader(args.ref)
        ref_stats = HistogramAnalyzer()
        for (fn, v) in reader.read_objects():
            ref_stats.analyze(fn, v)
        reader.close()

        stats = stats.difference(ref_stats)

    if args.summary:
        displayDirectoryStatistics(stats, args)

    total = stats.group(0)
    if args.human:
        print "Total bytes: %s" % kibisize(total[""])
    else:
        print total[""]

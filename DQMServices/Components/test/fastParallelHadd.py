#!/usr/bin/env python

from threading import Thread, Lock, activeCount
from optparse import OptionParser
import time
import commands
import os

class SingleMerge(Thread):
    def __init__(self, sm, output_filename, debug, *args):
        self.sm_ = sm
        self.output_filename_ = output_filename
        self.debug_ = debug
        self.files_ = args
        Thread.__init__(self)

    def run(self):
        com = "fastHadd add "
        com += "-o %s " % self.output_filename_
        for f in self.files_:
            com += "%s " % f
        if self.debug_ > 0:
            print "Going to execute: %s" % com
        start = time.time()
        (status, output) = commands.getstatusoutput(com)
        stop = time.time()
        if self.debug_ > 1:
            print "Process took %f secs. [%f, %f]" % (stop-start, start, stop)
        if status != 0:
            print "Merge job failed for %s" % com
        self.sm_.appendFile(self.output_filename_)

class SampleManager(object):
    def __init__(self, opt, files):
        self.lock_ = Lock()
        self.files_ = files
        self.nukefiles_ = []
#         for i in range(0, 50):
#             self.files_.append("%d.root" % i)
        print self.files_
        self.low_index_ = 0
        self.up_index_ = len(self.files_)
        self.abs_index_ = len(self.files_)
        self.group_by_ = max(2, opt.group_by) # no point in adding 1 file.
        self.num_threads_ = opt.num_threads
        self.output_file_ = opt.output_file
        self.debug_ = opt.debug
        self.step_ = 0

    def appendFile(self, input_file):
        self.lock_.acquire()
        self.files_.append(input_file)
        self.nukefiles_.append(input_file)
        self.up_index_ = self.up_index_ + 1
        if self.debug_ > 0:
            print "Inserting %s\n" % input_file
        self.lock_.release()

    def processFiles(self):
        while True:
            while self.up_index_ - self.low_index_ >= self.group_by_:
                while activeCount() > self.num_threads_:
                    pass
                if self.debug_ > 0:
                    print "Step: %d" % self.step_
                f = []
                for j in range(0, self.group_by_):
                    if self.debug_ > 1:
                        print "low_index: %d, up_index:%d" % (self.low_index_,
                                                              self.up_index_)
                    f.append(self.files_[self.low_index_])
                    self.low_index_ += 1
                if len(f) > 0:
                    t = SingleMerge(self, "%d.pb" % (self.abs_index_),
                                    self.debug_, *f)
                    t.start()
                    self.abs_index_ += 1
                    self.step_ += 1
            if activeCount() == 1:
                break

        # Consume whatever is left, in a single step, since we exited the
        # previous while
        if self.up_index_ > self.low_index_ + 1:
            f = []
            while (self.low_index_ + 1) <= self.up_index_:
                if self.debug_ > 2:
                    print "Currently at ", self.low_index_, self.up_index_, len(self.files_)
                f.append(self.files_[self.low_index_])
                self.low_index_ += 1
            if len(f) > 0 :
                if self.debug_ > 0:
                    print "Final Step: %d" % self.step_
                t = SingleMerge(self, "%d.pb" % (self.abs_index_),
                                self.debug_, *f)
                self.abs_index_ += 1
                t.start()

        while activeCount() > 1:
            pass
        self.finalise()

    def finalise(self):
        os.rename(self.files_[-1], self.output_file_)
        for f in self.nukefiles_[:-1]:
            os.remove(f)



if __name__ == '__main__':
    usage = "usage: %prog [options] -o output_file file1 file2 ... fileN"
    op = OptionParser(usage = usage)
    op.add_option("-d", "--debug", dest = "debug",
                  action = "count", default = 0,
                  help = "Debug information level. [default: %default]")
    op.add_option("-j", dest = "num_threads", type = "int",
                  action = "store", metavar = "NUM_THREADS",
                  default = 2,
                  help = "Maximum number of concurrent threads. [default: %default]")
    op.add_option("-g", dest = "group_by", type = "int",
                  action = "store", metavar = "GROUP_BY",
                  default = 2,
                  help = "Maximum number of files to be merged in a single job. [default: %default]")
    op.add_option("-o", dest = "output_file", type = "string",
                  action = "store", metavar = "OUTPUT_FILE",
                  default = None,
                  help = "Output filename.")
    (options, files) = op.parse_args()
    if options.output_file == None:
        op.error("Missing mandatory parameter -o output_file")
    if len(files) <= 1:
        op.error("You must supply at least 2 files to be merged.")
    sm = SampleManager(options, files)
    sm.processFiles()

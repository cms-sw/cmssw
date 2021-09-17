#!/usr/bin/env python3

"""Usage: test_fastHaddMerge.py [-t produce|check] [-n files]

Produce a custom number of identical ROOT files and check that their
final merged output matches what is expected.
"""
from __future__ import print_function

from builtins import range
from optparse import OptionParser
import sys
import commands
import re

word2num = {'One': 1, 'Two': 2, 'Ten': 10, 'Twenty': 20, 'Fifty': 50}

class Histo(object):
    def __init__(self, name, entries,
                 bins, xmin, xmax,
                 value, folder):
        self.name_ = name
        self.bins_ = bins
        self.xmin_ = xmin
        self.xmax_ = xmax
        self.entries_ = entries
        self.value_ = value
        self.folder_ = folder

    def book_and_fill(self, tfile):
        import ROOT as r
        r.gDirectory.cd('/')
        fullpath = ''
        for level in self.folder_.split('/'):
            fullpath += '/%s' % level
            if not level == '':
                if not r.gDirectory.GetDirectory(level):
                    r.gDirectory.mkdir(level)
                r.gDirectory.cd(fullpath)
        histo = r.TH1F(self.name_, self.name_,
                       self.bins_, self.xmin_, self.xmax_)
        for e in range(0, self.entries_):
            histo.Fill(self.value_)
        histo.Write()

class FileProducer(object):
    def __init__(self, prefix, numFiles, folders, histo_per_folder):
        self.prefix_ = prefix
        self.numFiles_ = numFiles
        self.folders_ = folders
        self.histo_per_folder_ = histo_per_folder
        self.histos_ = []
        assert numFiles > 0

    def checkCumulative(self, rootfile):
        import ROOT as r
        self.prepareHistos()
        f = r.TFile(rootfile)
        num = 0
        for histo in self.histos_:
            num += 1
            if num%10000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
            h = f.Get(histo.folder_+'/'+histo.name_)
            h.SetDirectory(0)
            assert h
            m = re.match('(.*)Entr.*_(\d+)$', h.GetName())
            assert m.groups()
            assert h.GetEntries() == word2num[m.group(1)] * self.numFiles_
            assert h.GetMean() == float(m.group(2))
        print()
        f.Close()
        print()

    def createIdenticalFiles(self):
        import ROOT as r
        self.prepareHistos()
        f = r.TFile("%s_%d.root" % (self.prefix_, 0), "RECREATE")
        num = 0
        for h in self.histos_:
            num = num + 1
            if num%1000 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
            h.book_and_fill(f)
        f.Write()
        f.Close()
        print('Wrote %d histograms in %d folders' % (len(self.histos_), len(self.folders_)))
        for i in range(1, self.numFiles_):
            commands.getoutput("cp %s_0.root %s_%d.root" % (self.prefix_,
                                                            self.prefix_,
                                                            i))
            print('x')

    def prepareHistos(self):
        for folder in self.folders_:
            sys.stdout.write("|")
            sys.stdout.flush()
            for i in range(0, self.histo_per_folder_):
                if i%100 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
                self.histos_.append(Histo("OneEntry_%d" %i,
                                          1, i+1, 0, i+1, i, folder))
                self.histos_.append(Histo("TwoEntries_%d" %i,
                                          2, i+1, 0, i+1, i, folder))
                self.histos_.append(Histo("TenEntries_%d" %i,
                                          10, i+1, 0, i+1, i, folder))
                self.histos_.append(Histo("TwentyEntries_%d" %i,
                                          20, i+1, 0, i+1, i, folder))
                self.histos_.append(Histo("FiftyEntries_%d" %i,
                                          50, i+1, 0, i+1, i, folder))
        print()

op = OptionParser(usage = __doc__)
op.add_option("-a", "--action", dest = "action",
              type = "string", action = "store", metavar = "ACTION",
              default = "produce",
              help = "Either produce or check ROOT file(s).")
op.add_option("-c", "--check", dest = "file_to_check",
              type = "string", action = "store", metavar = "FILE",
              default = '',
              help = "Check the content of FILE against expectations.")
op.add_option("-n", "--numfiles", dest = "numfiles",
              type = "int", action = "store", metavar = "NUM",
              default = 10, help = "Create NUM identical files")
options, args = op.parse_args()

if __name__ == '__main__':
    fp = FileProducer("MergePBTest", options.numfiles, ["/First_level",
                                                        "/Second/Level",
                                                        "/Third/Level/Folder",
                                                        "/Fourth/Of/Many/Folders",
                                                        "/Pixel/", "/Pixel/A", "/Pixel/A/B", "/Pixel/A/B/C", "/Pixel/A/B/C/D", "/Pixel/A/B/C/D/E",
                                                        "/SiStrip/", "/SiStrip/A", "/SiStrip/A/B", "/SiStrip/A/B/C", "/SiStrip/A/B/C/D", "/SiStrip/A/B/C/D/E",
                                                        "/RPC/", "/RPC/A", "/RPC/A/B", "/RPC/A/B/C", "/RPC/A/B/C/D", "/RPC/A/B/C/D/E",
                                                        "/HLT/", "/HLT/A", "/HLT/A/B", "/HLT/A/B/C", "/HLT/A/B/C/D", "/HLT/A/B/C/D/E",
                                                        "/EcalBarrel/", "/EcalBarrel/A", "/EcalBarrel/A/B", "/EcalBarrel/A/B/C", "/EcalBarrel/A/B/C/D", "/EcalBarrel/A/B/C/D/E",
                                                        "/EcalEndcap/", "/EcalEndcap/A", "/EcalEndcap/A/B", "/EcalEndcap/A/B/C", "/EcalEndcap/A/B/C/D", "/EcalEndcap/A/B/C/D/E",
                                                        "/Tracking/", "/Tracking/A", "/Tracking/A/B", "/Tracking/A/B/C", "/Tracking/A/B/C/D", "/Tracking/A/B/C/D/E",
                                                        "/Muon/", "/Muon/A", "/Muon/A/B", "/Muon/A/B/C", "/Muon/A/B/C/D", "/Muon/A/B/C/D/E",
                                                        "/EGamma/", "/EGamma/A", "/EGamma/A/B", "/EGamma/A/B/C", "/EGamma/A/B/C/D", "/EGamma/A/B/C/D/E",
                                                        "/Tau/", "/Tau/A", "/Tau/A/B", "/Tau/A/B/C", "/Tau/A/B/C/D", "/Tau/A/B/C/D/E"],
                      100)
    if options.action == 'produce':
        fp.createIdenticalFiles()
    else:
        if not options.action == 'check':
            print("Option -a|--action takes only 'produce|check' options.", file=sys.stderr)
            sys.exit(1)
        else:
            if options.file_to_check == '':
                print("Option -c|--check required to check the content of a file.", file=sys.stderr)
                sys.exit(1)
            fp.checkCumulative(options.file_to_check)

# Local Variables:
# show-trailing-whitespace: t
# truncate-lines: t
# End:

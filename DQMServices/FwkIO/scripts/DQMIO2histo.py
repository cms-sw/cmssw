#!/bin/env python3

"""
Script converting DQM I/O format input file into folder structured ROOT file.
Ouput files historgrams are easy browseable by ROOT. When there are more than 1 run
in input file it creates a Run X named folder for each run.
Thanks for Marco Rovere for giving example script/class needed to browse DQM I/O 
formatted input.
"""
from __future__ import print_function

from builtins import range
import ROOT as R
import sys
import re
import os
import argparse

class DQMIO:
    """
    Class responsible for browsing the content of a DQM file produced
    with the DQMIO I/O framework of CMSSW
    """
    types=["Ints","Floats","Strings",  ##defined DQMIO types
         "TH1Fs","TH1Ss","TH1Ds",
         "TH2Fs", "TH2Ss", "TH2Ds",
         "TH3Fs", "TProfiles","TProfile2Ds", "kNIndicies"]

    def __init__(self, input_filename, output_filename):
        self._filename = input_filename
        self._canvas = None
        self.f = R.TFile(output_filename, "RECREATE")
        self.already_defined = {"TProfiles" : False, "TProfile2Ds" : False,
                "TH2Fs" : False, "TH2Ds" : False}

        if os.path.exists(self._filename): #we try to open input file if fail
            self._root_file = R.TFile.Open(self._filename) #-> close script
            if args.debug:
                print("## DEBUG ##:")
                print("    Input: %s\n    Output: %s" % (input_filename, 
                    output_filename))

        else:
            print("File %s does not exists" % self._filename)
            sys.exit(1)
    
    def print_index(self):
        """
        Loop over the complete index and dump it on the screen.
        """
        indices = self._root_file.Get("Indices")
        if args.debug:
            print("## DEBUG ##:")
            print("Run,\tLumi,\tType,\t\tFirstIndex,\tLastIndex")
            for i in range(indices.GetEntries()):
                indices.GetEntry(i)
                print('{0:4d}\t{1:4d}\t{2:4d}({3:s})\t\t{4:4d}\t{5:4d}'.format(
                    indices.Run, indices.Lumi, indices.Type, 
                    DQMIO.types[indices.Type], indices.FirstIndex, indices.LastIndex))

        for i in range(indices.GetEntries()):
            indices.GetEntry(i)
            if indices.Type < len(DQMIO.types):
                self.write_to_file(self.types[indices.Type],
                    [indices.FirstIndex,indices.LastIndex], str(indices.Run))

            else:
                print("Unknown histogram type. Type numer: %s" % (indices.Type))
        self.f.Close()

    def write_to_file(self, hist_type, index_range, run):
        """
        Method looping over entries for specified histogram type and 
        writing to FullName path to output ROOT File
        """
        print("Working on: %s indexes: %s..%s" % (hist_type ,index_range[0],
            index_range[1]))
        t_tree = self._root_file.Get(hist_type)
        __run_dir = "Run %s" % (run)
        ###we set Branch for the needed type
        if hist_type == "TProfiles":
            if not self.already_defined["TProfiles"]:
                R.gROOT.ProcessLine("TProfile* _tprof;")
                self.already_defined["TProfiles"] = True
            t_tree.SetBranchAddress("Value", R._tprof)
            t_tree.GetEntry(index_range[0])
        elif hist_type == "TProfile2Ds":
            if not self.already_defined["TProfile2Ds"]:
                R.gROOT.ProcessLine("TProfile2D* _tprof2d;")
                self.already_defined["TProfile2Ds"] = True
            t_tree.SetBranchAddress("Value", R._tprof2d)
            t_tree.GetEntry(index_range[0])
        elif hist_type == "TH2Fs":
            if not self.already_defined["TH2Fs"]:
                R.gROOT.ProcessLine("TH2F* _th2f;")
                self.already_defined["TH2Fs"] = True
            t_tree.SetBranchAddress("Value", R._th2f)
            t_tree.GetEntry(index_range[0])
        elif hist_type == "TH2Ds":
            if not self.already_defined["TH2Ds"]:
                R.gROOT.ProcessLine("TH2D* _th2d;")
                self.already_defined["TH2Ds"] = True
            t_tree.SetBranchAddress("Value", R._th2d)
            t_tree.GetEntry(index_range[0])

        for i in range(0,t_tree.GetEntries()+1): ##iterate on entries for specified type
            if i >= index_range[0] and i <= index_range[1]: ##if entries as in range:
                t_tree.GetEntry(i)
                name = str(t_tree.FullName)
               # print "  %s:  %s" % (i, name)
                file_path = name.split("/")[:-1]            ##  same run/lumi histograms
                __directory = "%s/%s" % (os.path.join("DQMData", __run_dir),
                    "/".join(file_path))
                directory_ret = self.f.GetDirectory(__directory)
                if not directory_ret:
                    self.f.mkdir(os.path.join(__directory))
                self.f.cd(os.path.join(__directory))
                if hist_type == "Strings":
                    construct_str = '<%s>s=%s</%s>' % (name.split("/")[-1:][0],
                        t_tree.Value, name.split("/")[-1:][0])
                    tmp_str = R.TObjString(construct_str)
                    tmp_str.Write()
                elif hist_type == "Ints":
                    construct_str = '<%s>i=%s</%s>' % (name.split("/")[-1:][0],
                        t_tree.Value, name.split("/")[-1:][0])
                    tmp_str = R.TObjString(construct_str)
                    tmp_str.Write()
                elif hist_type == "Floats":
                    construct_str = '<%s>f=%s</%s>' % (name.split("/")[-1:][0],
                        t_tree.Value, name.split("/")[-1:][0])
                    tmp_str = R.TObjString(construct_str)
                    tmp_str.Write()
                else:
                    if hist_type in ["TProfiles", "TProfile2Ds", "TH2Fs", "TH2Ds"]: 
                        if hist_type == "TProfiles": #if type is specific we write it.
                            R._tprof.Write()
                        elif hist_type == "TProfile2Ds":
                            R._tprof2d.Write()
                        elif hist_type == "TH2Fs":
                            R._th2f.Write()
                        elif hist_type == "TH2Ds":
                            R._th2d.Write()
                    else: #else we wirte Leafs Value which is a histogram
                        t_tree.Value.Write()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-in", "--input", help = "Input DQMIO ROOT file")
    parser.add_argument("-o", "--output", help = "Output filename",
        default = "DQMIO_converter_output.root")
    parser.add_argument("--debug", help = "Debug mode to spam you console",
        action = "store_true")

    args = parser.parse_args()
    __in_file = args.input
    __out_file = args.output
    dqmio = DQMIO(__in_file, __out_file)
    dqmio.print_index()

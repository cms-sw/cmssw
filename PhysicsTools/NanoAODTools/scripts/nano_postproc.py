#!/usr/bin/env python3
from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from importlib import import_module
import os
import sys
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-s", "--postfix", dest="postfix", type=str, default=None,
                        help="Postfix which will be appended to the file name (default: _Friend for friends, _Skim for skims)")
    parser.add_argument("-J", "--json", dest="json", type=str,
                        default=None, help="Select events using this JSON file")
    parser.add_argument("-c", "--cut", dest="cut", type=str,
                        default=None, help="Cut string")
    parser.add_argument("-b", "--branch-selection", dest="branchsel",
                        type=str, default=None, help="Branch selection")
    parser.add_argument("--bi", "--branch-selection-input", dest="branchsel_in",
                        type=str, default=None, help="Branch selection input")
    parser.add_argument("--bo", "--branch-selection-output", dest="branchsel_out",
                        type=str, default=None, help="Branch selection output")
    parser.add_argument("--friend", dest="friend", action="store_true", default=False,
                        help="Produce friend trees in output (current default is to produce full trees)")
    parser.add_argument("--full", dest="friend", action="store_false", default=False,
                        help="Produce full trees in output (this is the current default)")
    parser.add_argument("--noout", dest="noOut", action="store_true",
                        default=False, help="Do not produce output, just run modules")
    parser.add_argument("-P", "--prefetch", dest="prefetch", action="store_true", default=False,
                        help="Prefetch input files locally instead of accessing them via xrootd")
    parser.add_argument("--long-term-cache", dest="longTermCache", action="store_true", default=False,
                        help="Keep prefetched files across runs instead of deleting them at the end")
    parser.add_argument("-N", "--max-entries", dest="maxEntries", type=int, default=None,
                        help="Maximum number of entries to process from any single given input tree")
    parser.add_argument("--first-entry", dest="firstEntry", type=int, default=0,
                        help="First entry to process in the three (to be used together with --max-entries)")
    parser.add_argument("--justcount", dest="justcount", default=False,
                        action="store_true", help="Just report the number of selected events")
    parser.add_argument("-I", "--import", dest="imports", type=str, default=[], action="append",
                        nargs=2, help="Import modules (python package, comma-separated list of ")
    parser.add_argument("-z", "--compression", dest="compression", type=str,
                        default=("LZMA:9"), help="Compression: none, or (algo):(level) ")
    parser.add_argument("outputDir", type=str)
    parser.add_argument("inputFile", type=str, nargs='+')
    options = parser.parse_args()

    if options.friend:
        if options.cut or options.json:
            raise RuntimeError(
                "Can't apply JSON or cut selection when producing friends")

    modules = []
    for mod, names in options.imports:
        import_module(mod)
        obj = sys.modules[mod]
        selnames = names.split(",")
        mods = dir(obj)
        for name in selnames:
            if name in mods:
                print("Loading %s from %s " % (name, mod))
                if type(getattr(obj, name)) == list:
                    for mod in getattr(obj, name):
                        modules.append(mod())
                else:
                    modules.append(getattr(obj, name)())
    if options.noOut:
        if len(modules) == 0:
            raise RuntimeError(
                "Running with --noout and no modules does nothing!")
    if options.branchsel != None:
        options.branchsel_in = options.branchsel
        options.branchsel_out = options.branchsel
    p = PostProcessor(options.outputDir, options.inputFile,
                      cut=options.cut,
                      branchsel=options.branchsel_in,
                      modules=modules,
                      compression=options.compression,
                      friend=options.friend,
                      postfix=options.postfix,
                      jsonInput=options.json,
                      noOut=options.noOut,
                      justcount=options.justcount,
                      prefetch=options.prefetch,
                      longTermCache=options.longTermCache,
                      maxEntries=options.maxEntries,
                      firstEntry=options.firstEntry,
                      outputbranchsel=options.branchsel_out)
    p.run()

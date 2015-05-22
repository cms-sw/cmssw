#!/usr/bin/env python
#from mcPlots import *
from CMGTools.TTHAnalysis.plotter.mcAnalysis import *

import ROOT

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] mc.txt cuts.txt outputDir")
    parser.add_option("-D", "--drop",  dest="drop", type="string", default=[], action="append",  help="Branches to drop, as per TTree::SetBranchStatus") 
    parser.add_option("-K", "--keep",  dest="keep", type="string", default=[], action="append",  help="Branches to keep, as per TTree::SetBranchStatus") 
    parser.add_option("--oldstyle",    dest="oldstyle", default=False, action="store_true",  help="Oldstyle naming (e.g. file named as <analyzer>_tree.root)") 
    addMCAnalysisOptions(parser)
    (options, args) = parser.parse_args()
    options.weight = False
    options.final = True
    mca  = MCAnalysis(args[0],options)
    cut = CutsFile(args[1],options)
    outdir = args[2]

    print "Will write selected trees to "+outdir
    if not os.path.exists(outdir):
        os.system("mkdir -p "+outdir)

    for proc in mca.listProcesses():
        print "Process %s" % proc
        for tty in mca._allData[proc]:
            print "\t component %-40s" % tty.cname(),
            mytree = tty.getTree()
            yields = tty.getYields(cut, noEntryLine=False) 
            ntot  = mytree.GetEntries() 
            npass = yields[0][1][0]
            print ": %8d/%8d" % (npass,ntot)
            myoutpath = outdir+"/"+tty.cname()
            mysource  = options.path+"/"+tty.cname()
            mycut = tty.adaptExpr(cut.allCuts(),cut=True)
            if options.doS2V: mycut  = scalarToVector(mycut)
            # now we do
            os.system("mkdir -p "+myoutpath)
            os.system("cp -r %s/skimAnalyzerCount %s/" % (mysource,myoutpath))
            os.system("mkdir -p %s/%s" % (myoutpath,options.tree))
            histo = ROOT.gROOT.FindObject("Count")
            if not options.oldstyle:
                fout = ROOT.TFile("%s/%s/tree.root" % (myoutpath,options.tree), "RECREATE");
            else:
                fout = ROOT.TFile("%s/%s/%s_tree.root" % (myoutpath,options.tree,options.tree), "RECREATE");
            fout = ROOT.TFile("%s/%s/tree.root" % (myoutpath,options.tree), "RECREATE");
            # drop and keep branches
            for drop in options.drop: mytree.SetBranchStatus(drop,0)
            for keep in options.keep: mytree.SetBranchStatus(keep,1)
            out = mytree.CopyTree(mycut)
            friends = out.GetListOfFriends() or []
            for tf in friends:
                    out.RemoveFriend(tf.GetTree())
            fout.WriteTObject(out,options.tree if options.oldstyle else "tree")
            fout.WriteTObject(out,"tree")
            if histo: histo.Write()
            fout.Close()


import sys
import numpy as np

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
from optparse import OptionParser
from RecoTauTag.Configuration.tools.DisplayManager import DisplayManager

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

colours = [1, 2, 3, 6, 8]
styles = [1, 2, 3, 4, 5]

def findTree(f):
    for key in f.GetListOfKeys():
        if key.GetName() == "Events":
            tree = f.Get(key.GetName())
            if isinstance(tree, ROOT.TTree):
                return tree
            elif isinstance(tree, ROOT.TDirectory):
                return findTree(tree)
    print 'Failed to find TTree Events in file', f
    return None

def applyHistStyle(h, i):
    h.SetLineColor(colours[i])
    h.SetLineStyle(styles[i])
    h.SetLineWidth(3)
    h.SetStats(False)

def comparisonPlots(u_names, trees, titles, pname='compareTauVariables.pdf', ratio=True):
    display = DisplayManager(pname, ratio)
    for branch in u_names:
        nbins = 50
        min_x = min(t.GetMinimum(branch) for t in trees)
        max_x = max(t.GetMaximum(branch) for t in trees)
        title_x = branch
        if min_x == max_x or all(t.GetMinimum(branch) == t.GetMaximum(branch) for t in trees):
            continue
        if min_x < -900 and max_x < -min_x * 1.5:
            min_x = - max_x
        min_x = min(0., min_x)
        hists = []
        for i, t in enumerate(trees):
            h_name = branch+t.GetName()+str(i)
            h = ROOT.TH1F(h_name, branch, nbins, min_x, max_x + (max_x - min_x) * 0.01)
            h.Sumw2()
            h.GetXaxis().SetTitle(title_x)
            h.GetYaxis().SetTitle('Entries')
            applyHistStyle(h, i)
            t.Project(h_name, branch, '1') # Should introduce weight...
            hists.append(h)
        display.Draw(hists, titles)

def interSect(tree1, tree2, var='ull_dumpTauVariables_EventNumber_DUMP.obj', common=False, save=False,  titles=[]):
    tlist1 = ROOT.TEntryList()
    tlist2 = ROOT.TEntryList()
    tree1.Draw(var)
    r_evt1 = tree1.GetV1()
    if len(titles) > 0 and titles[0] == 'Reference':
        evt1 = np.array([int(r_evt1[i]) & 0xffffffff for i in xrange(tree2.GetEntries())], dtype=int)
    else:
        evt1 = np.array([r_evt1[i] for i in xrange(tree1.GetEntries())], dtype=int)
    tree2.Draw(var)
    r_evt2 = tree2.GetV1()
    if len(titles) > 1 and titles[1] == 'Reference':
        evt2 = np.array([int(r_evt2[i]) & 0xffffffff for i in xrange(tree2.GetEntries())], dtype=int)
    else:
        evt2 = np.array([int(r_evt2[i]) for i in xrange(tree2.GetEntries())], dtype=int)
    if common:
        indices1 = np.nonzero(np.in1d(evt1, evt2))
        indices2 = np.nonzero(np.in1d(evt2, evt1))
    else:
        indices1 = np.nonzero(np.in1d(evt1, evt2) == 0)
        indices2 = np.nonzero(np.in1d(evt2, evt1) == 0)
    if save:
        if len(titles) < 2:
            titles = ['tree1', 'tree2']
        evt1[indices1].tofile(titles[0]+'.csv', sep=',', format='%d')
        evt2[indices2].tofile(titles[1]+'.csv', sep=',', format='%d')
    for ind1 in indices1[0]:
        tlist1.Enter(ind1)
    for ind2 in indices2[0]:
        tlist2.Enter(ind2)
    return tlist1, tlist2

def scanForDiff(tree1, tree2, branch_names, scan_var='floats_dumpTauVariables_pt_DUMP.obj', index_var='ull_dumpTauVariables_EventNumber_DUMP.obj'):
    tree2.BuildIndex(index_var)
    diff_events = []
    for entry_1 in tree1:
        ind = int(getattr(tree1, index_var))
        tree2.GetEntryWithIndex(ind)
        var1 = getattr(tree1, scan_var)
        var2 = getattr(tree2, scan_var)
        if tree1.evt != tree2.evt:
            continue
        if round(var1, 6) != round(var2, 6): 
            diff_events.append(ind)
            print 'Event', ind
            for branch in branch_names:
                v1 = getattr(tree1, branch)
                v2 = getattr(tree2, branch)
                if round(v1, 6) != round(v2, 6) and v1 > -99.:
                    print '{b:>43}: {v1:>8.4f}, {v2:>8.4f}'.format(b=branch, v1=v1, v2=v2)
            print
    print 'Found', len(diff_events), 'events with differences in', scan_var
    print diff_events


if __name__ == '__main__':
        
    usage = '''
%prog [options] arg1 arg2 ... argN
    Compares first found trees in N different root files; 
    in case of two root files, additional information about the event overlap
    can be calculated.
    Example run commands:
> python compareTauVariables.py dumpTauVariables_CMSSW_8_0_X.root dumpTauVariables_CMSSW_8_1_X.root -t CMSSW_8_0_X,CMSSW_8_1_X
'''

    parser = OptionParser(usage=usage)

    parser.add_option('-t', '--titles', type='string', dest='titles', default='Reference,Test', help='Comma-separated list of titles for the N input files (e.g. CMSSW_8_0_X,CMSSW_8_1_X)')
    parser.add_option('-i', '--do-intersection', dest='do_intersect', action='store_true', default=False, help='Produce plots for events not in common')
    parser.add_option('-c', '--do-common', dest='do_common', action='store_true', default=False, help='Produce plots for events in common')
    parser.add_option('-r', '--do-ratio', dest='do_ratio', action='store_true', default=False, help='Show ratio plots')
    parser.add_option('-d', '--diff', dest='do_diff', action='store_true', default=False, help='Print events where single variable differs')
    parser.add_option('-v', '--var-diff', dest='var_diff', default='floats_dumpTauVariables_pt_DUMP.obj', help='Variable for printing single event diffs')

    (options,args) = parser.parse_args()

    if len(args) < 2:
        print 'provide at least 2 input root files'
        sys.exit(1)

    titles = options.titles.split(',')    
    if len(titles) < len(args):
        print 'Provide at least as many titles as input files'
        sys.exit(1)

    for i, arg in enumerate(args):
        if arg.endswith('.txt'):
            f_txt = open(arg)
            for line in f_txt.read().splitlines():
                line.strip()
                if line.startswith('/afs'):
                    args[i] = line
                    break

    tfiles = [ROOT.TFile(arg) for arg in args]    
    trees = [findTree(f) for f in tfiles]
    b_names = []
    b_event_number = ''
    for t in trees:
        bs = []
        for b in t.GetListOfBranches():
            if "DUMP" in b.GetName():
                bs.append(b.GetName().replace("'","\'")+"obj")
        b_names.append(set(bs))
        if "EventNumber" in bs :
            b_event_number = bs
    u_names = set.intersection(*b_names)
    u_names = sorted(u_names)

    print 'Making plots for all common branches (', len(u_names), ')'
    comparisonPlots(u_names, trees, titles, 'compareTauVariables.pdf', options.do_ratio)

    if len(trees) == 2 and options.do_intersect:
        intersect = interSect(trees[0], trees[1], var=b_event_number, save=True, titles=titles)
        trees[0].SetEntryList(intersect[0])
        trees[1].SetEntryList(intersect[1])
        if not all(l.GetN() == 0 for l in intersect):
            comparisonPlots(u_names, trees, titles, 'compareTauVariables_intersect.pdf', options.do_ratio)

    if len(trees) == 2 and options.do_common:
        intersect = interSect(trees[0], trees[1], var=b_event_number, common=True)
        trees[0].SetEntryList(intersect[0])
        trees[1].SetEntryList(intersect[1])
        comparisonPlots(u_names, trees, titles, 'compareTauVariables_common.pdf', options.do_ratio)

    if len(trees) == 2 and options.do_diff:
        scanForDiff(trees[0], trees[1], u_names, scan_var=options.var_diff, index_var=b_event_number)

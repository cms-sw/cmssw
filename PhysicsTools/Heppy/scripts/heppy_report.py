#!/usr/bin/env python
from optparse import OptionParser
import json

def root2map(dir,ana,treename):
    import ROOT
    tfile = ROOT.TFile.Open("%s/%s/%s.root"%(dir,ana,treename))
    if not tfile:
        print "Error: dir %s does not contain %s/%s.root" % (dir,ana,treename)
        return None
    tree = tfile.Get(treename)
    if not tree:
        print "Error: rootfile %s/%s/%s.root does not contain a TTree %s" % (dir,ana,treename,treename)
        return None
    jsonind = {}
    for e in xrange(tree.GetEntries()):
        tree.GetEntry(e)
        run,lumi = tree.run, tree.lumi
        if run not in jsonind:
            jsonind[run] = [lumi]
        else:
            jsonind[run].append(lumi)
    # remove duplicates
    for run in jsonind:
        jsonind[run] =  list(set(jsonind[run]))

    nruns = len(jsonind)
    nlumis = sum(len(v) for v in jsonind.itervalues())
    jsonmap = {}
    for r,lumis in jsonind.iteritems():
        if len(lumis) == 0: continue # shouldn't happen
        lumis.sort()
        ranges = [ [ lumis[0], lumis[0] ] ]
        for lumi in lumis[1:]:
            if lumi == ranges[-1][1] + 1:
                ranges[-1][1] = lumi
            else:
                ranges.append([lumi,lumi])
        jsonmap[r] = ranges
    return (jsonmap, nruns, nlumis)

if __name__ == '__main__':
    parser = OptionParser(usage='%prog <target_directories> [options]',
                          description='Check the output of the JSONAnalyzer and produce a json file of the processed runs and lumisections')
    parser.add_option("-a", "--analyzer", dest="jsonAnalyzer", default="JSONAnalyzer", help="Name of the JSONAnalyzer")
    parser.add_option("-t", "--tree", dest="treeName", default="RLTInfo", help="Name of the TTree produced by the JSONAnalyzer")
    parser.add_option("-o", "--out", dest="outputFile", default="lumiSummary.json", help="Name of the output file")
    (options,args) = parser.parse_args()
    if len(args)==0:
        print 'provide at least one directory in argument. Use -h to display help'
        exit()
    for a in args:
        summary = root2map(a,options.jsonAnalyzer,options.treeName)
        if summary:
            oname = "%s/%s" % (a,options.outputFile)
            jmap, runs, lumis = summary
            json.dump(jmap,open(oname,'w'))
            print "Saved %s (%d runs, %d lumis)" % (oname, runs, lumis)

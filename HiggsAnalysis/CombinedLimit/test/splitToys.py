# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
from sys import argv
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
argv.remove( '-b-' )

from optparse import OptionParser
parser = OptionParser(usage="usage: %prog [options] [ -o outfile.root ]  infiles ")
parser.add_option("-c", "--category", dest="cat",     type="string", default="CMS_channel", help="Category to split channels along")
parser.add_option("-b", "--binned",   dest="binned",    default=False, action="store_true",  help="produce a RooDataHist")
parser.add_option("-C", "--counting", dest="counting",  default=False, action="store_true",  help="Input data is a counting experiment")
parser.add_option("-S", "--manual-split", dest="manualSplit",  default=False, action="store_true",  help="Split datasets manually (necessary for binned-unbinned mixes)")
parser.add_option("-v", "--verbose",  dest="verbose",  default=0,  type="int",    help="Verbosity level (0 = quiet, 1 = verbose, 2+ = more)")
parser.add_option("-n", "--name",     dest="outn",     type="string", default="{channel}_bData_{number}", help="Naming convention for output")
parser.add_option("-o", "--out",      dest="outf",     type="string", default="splittedToys.root", help="Name of the output file")
parser.add_option("--first", dest="first", default=0,       type="int",    help="First toy to include in the output file (note: numbers start from 1)")
parser.add_option("--last",  dest="last",  default=999999,  type="int",    help="Last toy to include in output file")

(options, args) = parser.parse_args()
if len(args) == 0:
    parser.print_usage()
    exit(1)


if   options.verbose == 0: ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
elif options.verbose == 1: ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

outw    = ROOT.RooWorkspace("w","w")
impw    = getattr(outw, 'import')

if options.counting:
    outw._wv = outw.factory("_weight_[1]")
    outw._fv = outw.factory("CMS_fakeObs[0,1]")
    outw.var("CMS_fakeObs").setBins(1);
elif options.manualSplit:
    outw._wv = outw.factory("_weight_[1]")

dupCheck = {}

iToy = -1
for fname in args:
    infile  = ROOT.TFile(fname)
    iToyInFile = 0;
    while True:
        iToyInFile += 1; iToy += 1; dataset = infile.Get("toys/toy_%d" % iToyInFile);
        if dataset == None: break
        print "Processing dataset ",iToy," (", iToyInFile," in file ", fname, ")"
        if iToy < options.first: continue
        if iToy > options.last: break
        if options.verbose > 1:
            dataset.get().Print("V")
        obs = ROOT.RooArgSet(dataset.get()); datasets = []
        if options.counting:
            obsin  = ROOT.RooArgList(obs)
            obsout = ROOT.RooArgSet(outw.var("CMS_fakeObs"))
            entry = dataset.get(0)
            for i in range(obsin.getSize()):
                name = obsin.at(i).GetName().replace("n_obs_bin","")
                data = ROOT.RooDataSet(name,name,ROOT.RooArgSet(outw.var("CMS_fakeObs"),outw.var("_weight_")), "_weight_");
                val = entry.find(obsin.at(i).GetName()).getVal()
                data.add(obsout, val)
                datasets.append(data)
            obs = obsout 
        else:
            cat = dataset.get().find(options.cat)
            if cat == None: raise RuntimeError, "Cannot find category %s in dataset." % options.cat
            obs.remove(cat)
            if options.verbose > 1: 
                print " observables in reduced dataset: "; obs.Print("V");
            datasets = []
            if not options.manualSplit:
                list = dataset.split(cat)
                datasets = [ list.At(i) for i in xrange(list.GetSize()) ]
            else:
                datamap = {}
                obsPlusW = ROOT.RooArgSet(obs); obsPlusW.add(outw._wv);
                for ic in xrange(cat.numBins("")):
                    cat.setBin(ic)
                    datamap[cat.getLabel()] = ROOT.RooDataSet(cat.getLabel(), cat.getLabel(), obsPlusW, "_weight_");
                    if options.verbose > 1: print "Category ",cat.getLabel()
                for i in xrange(dataset.numEntries()):
                    entry = dataset.get(i)
                    if options.verbose > 2:
                        print "  input entry %d of weight %g" % (i, dataset.weight())
                        entry.Print("V")
                    obs.assignValueOnly(entry)
                    datamap[entry.getCatLabel(options.cat)].add(obs, dataset.weight())
                datasets = datamap.values()
            if options.verbose > 2:
                for d in datasets:
                    print "Dumping dataset %15s, %6d entries, %8.1f events" %  (d.GetName(),d.numEntries(),d.sumEntries())
                    for i in xrange(d.numEntries()):
                        entry = d.get(i)
                        print "  entry %d of weight %g" % (i, d.weight())
                        entry.Print("V")
            population = ";".join(["%s=%g" % (d.GetName(),d.sumEntries()) for d in datasets])
            if population in dupCheck:
                print "DUPLICATE population: %s (toys %s)" % (population, ", ".join(dupCheck[population]))
                dupCheck[population].append(str(iToy))
            else:
                dupCheck[population] = [ str(iToy) ]
        if options.binned:
            datasets = [ ROOT.RooDataHist(d.GetName(),d.GetTitle(),obs,d) for d in datasets ]
        if options.verbose: 
            print " splitted datasets:"
            for i,d in enumerate(datasets):
                print "   dataset %d: %15s, %6d entries, %8.1f events"%(i,d.GetName(),d.numEntries(),d.sumEntries())
                if options.verbose > 1: d.get().Print("V")
        for d in datasets:
            d.SetName(options.outn.format(channel=d.GetName(),number=iToy))
            impw(d)
    infile.Close()
    if iToy > options.last: break
if options.verbose:
    outw.Print("V")
outw.writeToFile(options.outf)

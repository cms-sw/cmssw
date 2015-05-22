#!/usr/bin/env python
from CMGTools.TTHAnalysis.plotter.mcAnalysis import *
import re, sys, os, os.path
systs = {}

from optparse import OptionParser
parser = OptionParser(usage="%prog [options] mc.txt cuts.txt var bins systs.txt ")
addMCAnalysisOptions(parser)
parser.add_option("-o",   "--out",    dest="outname", type="string", default=None, help="output name") 
parser.add_option("--od", "--outdir", dest="outdir", type="string", default=None, help="output name") 
parser.add_option("-v", "--verbose",  dest="verbose",  default=0,  type="int",    help="Verbosity level (0 = quiet, 1 = verbose, 2+ = more)")
parser.add_option("--masses", dest="masses", default=None, type="string", help="produce results for all these masses")
parser.add_option("--mass-int-algo", dest="massIntAlgo", type="string", default="sigmaBR", help="Interpolation algorithm for nearby masses") 
parser.add_option("--asimov", dest="asimov", action="store_true", help="Asimov")

(options, args) = parser.parse_args()
options.weight = True
options.final  = True
options.allProcesses  = True

mca  = MCAnalysis(args[0],options)
cuts = CutsFile(args[1],options)

truebinname = os.path.basename(args[1]).replace(".txt","") if options.outname == None else options.outname
binname = truebinname if truebinname[0] not in "234" else "ttH_"+truebinname
outdir  = options.outdir+"/" if options.outdir else ""

masses = [ 125.7 ]
if options.masses:
    masses = [ float(x) for x in open(options.masses) ]

def file2map(x):
    ret = {}; headers = []
    for x in open(x,"r"):
        cols = x.split()
        if len(cols) < 2: continue
        if "BR2" in x: # skip the errors
            cols = [cols[0]] + [c for (i,c) in enumerate(cols) if i % 3 == 1]
        if "mH" in x:
            headers = [i.strip() for i in cols[1:]]
        else:
            fields = [ float(i) for i in cols ]
            ret[fields[0]] = dict(zip(headers,fields[1:]))
    return ret
YRpath = os.environ['CMSSW_RELEASE_BASE']+"/src/HiggsAnalysis/CombinedLimit/data/lhc-hxswg/sm/";
XStth = file2map(YRpath+"xs/8TeV/8TeV-ttH.txt")
BRhvv = file2map(YRpath+"br/BR2bosons.txt")
BRhff = file2map(YRpath+"br/BR2fermions.txt")
def mkspline(table,column,sf=1.0):
    pairs = [ (x,c[column]/sf) for (x,c) in table.iteritems() ]
    pairs.sort()
    x,y = ROOT.std.vector('double')(), ROOT.std.vector('double')()
    for xi,yi in pairs:
        x.push_back(xi) 
        y.push_back(yi) 
    spline = ROOT.ROOT.Math.Interpolator(x,y);
    spline._x = x
    spline._y = y
    return spline
splines = {
    'ttH' : mkspline(XStth, "XS_pb",   0.1271 * 1.00757982823 ), ## get 1 at 125.7
    'hww' : mkspline(BRhvv, "H_WW",    0.2262 ),
    'hzz' : mkspline(BRhvv, "H_ZZ",    0.0281 ),
    'htt' : mkspline(BRhff, "H_tautau",0.0620 ),
}
def getYieldScale(mass,process):
    if "ttH_" not in process: return 1.0
    scale = splines['ttH'].Eval(mass)
    for dec in "hww","hzz","htt":
        if dec in process: 
            scale *= splines[dec].Eval(mass)
            if 'efficiency_'+dec in splines:
                scale *= splines['efficiency_'+dec].Eval(mass)
            break
    return scale 

report = mca.getPlotsRaw("x", args[2], args[3], cuts.allCuts(), nodata=options.asimov)

if options.asimov:
    tomerge = []
    for p in mca.listSignals() + mca.listBackgrounds():
        if p in report: tomerge.append(report[p])
    report['data_obs'] = mergePlots("x_data_obs", tomerge) 
else:
    report['data_obs'] = report['data'].Clone("x_data_obs") 

allyields = dict([(p,h.Integral()) for p,h in report.iteritems()])
procs = []; iproc = {}
for i,s in enumerate(mca.listSignals()):
    if allyields[s] == 0: continue
    procs.append(s); iproc[s] = i-len(mca.listSignals())+1
for i,b in enumerate(mca.listBackgrounds()):
    if allyields[b] == 0: continue
    procs.append(b); iproc[b] = i+1

systs = {}
systsEnv = {}
for sysfile in args[4:]:
    for line in open(sysfile, 'r'):
        if re.match("\s*#.*", line): continue
        line = re.sub("#.*","",line).strip()
        if len(line) == 0: continue
        field = [f.strip() for f in line.split(':')]
        if len(field) < 4:
            raise RuntimeError, "Malformed line %s in file %s"%(line.strip(),sysfile)
        elif len(field) == 4 or field[4] == "lnN":
            (name, procmap, binmap, amount) = field[:4]
            if re.match(binmap,truebinname) == None: continue
            if name not in systs: systs[name] = []
            systs[name].append((re.compile(procmap),amount))
        elif field[4] in ["envelop","shapeOnly","templates","alternateShapeOnly"]:
            (name, procmap, binmap, amount) = field[:4]
            if re.match(binmap,truebinname) == None: continue
            if name not in systs: systsEnv[name] = []
            systsEnv[name].append((re.compile(procmap),amount,field[4]))
        else:
            raise RuntimeError, "Unknown systematic type %s" % field[4]
    if options.verbose:
        print "Loaded %d systematics" % len(systs)
        print "Loaded %d envelop systematics" % len(systsEnv)


for name in systs.keys():
    effmap = {}
    for p in procs:
        effect = "-"
        for (procmap,amount) in systs[name]:
            if re.match(procmap, p): effect = amount
        if mca._projection != None and effect not in ["-","0","1"]:
            if "/" in effect:
                e1, e2 = effect.split("/")
                effect = "%.3f/%.3f" % (mca._projection.scaleSyst(name, float(e1)), mca._projection.scaleSyst(name, float(e1)))
            else:
                effect = str(mca._projection.scaleSyst(name, float(effect)))
        effmap[p] = effect
    systs[name] = effmap

for name in systsEnv.keys():
    effmap0  = {}
    effmap12 = {}
    for p in procs:
        effect = "-"
        effect0  = "-"
        effect12 = "-"
        for (procmap,amount,mode) in systsEnv[name]:
            if re.match(procmap, p): effect = float(amount) if mode not in ["templates","alternateShape", "alternateShapeOnly"] else amount
        if mca._projection != None and effect not in ["-","0","1",1.0,0.0] and type(effect) == type(1.0):
            effect = mca._projection.scaleSyst(name, effect)
        if effect == "-" or effect == "0": 
            effmap0[p]  = "-" 
            effmap12[p] = "-" 
            continue
        if mode in ["envelop","shapeOnly"]:
            nominal = report[p]
            p0up = nominal.Clone(nominal.GetName()+"_"+name+"0Up"  ); p0up.Scale(effect)
            p0dn = nominal.Clone(nominal.GetName()+"_"+name+"0Down"); p0dn.Scale(1.0/effect)
            p1up = nominal.Clone(nominal.GetName()+"_"+name+"1Up"  );
            p1dn = nominal.Clone(nominal.GetName()+"_"+name+"1Down");
            p2up = nominal.Clone(nominal.GetName()+"_"+name+"2Up"  );
            p2dn = nominal.Clone(nominal.GetName()+"_"+name+"2Down");
            nbin = nominal.GetNbinsX()
            xmin = nominal.GetBinCenter(1)
            xmax = nominal.GetBinCenter(nbin)
            for b in xrange(1,nbin+1):
                x = (nominal.GetBinCenter(b)-xmin)/(xmax-xmin)
                c1 = 2*(x-0.5)         # straight line from (0,-1) to (1,+1)
                c2 = 1 - 8*(x-0.5)**2  # parabola through (0,-1), (0.5,~1), (1,-1)
                p1up.SetBinContent(b, p1up.GetBinContent(b) * pow(effect,+c1))
                p1dn.SetBinContent(b, p1dn.GetBinContent(b) * pow(effect,-c1))
                p2up.SetBinContent(b, p2up.GetBinContent(b) * pow(effect,+c2))
                p2dn.SetBinContent(b, p2dn.GetBinContent(b) * pow(effect,-c2))
            if mode != "shapeOnly":
                report[p+"_"+name+"0Up"]   = p0up
                report[p+"_"+name+"0Down"] = p0dn
                effect0 = "1"
            report[p+"_"+name+"1Up"]   = p1up
            report[p+"_"+name+"1Down"] = p1dn
            report[p+"_"+name+"2Up"]   = p2up
            report[p+"_"+name+"2Down"] = p2dn
            effect12 = "1"
            # useful for plotting
            for h in p0up, p0dn, p1up, p1dn, p2up, p2dn: 
                h.SetFillStyle(0); h.SetLineWidth(2)
            for h in p1up, p1dn: h.SetLineColor(4)
            for h in p2up, p2dn: h.SetLineColor(2)
        elif mode in ["templates"]:
            nominal = report[p]
            p0Up = report["%s_%s_Up" % (p, effect)]
            p0Dn = report["%s_%s_Dn" % (p, effect)]
            if not p0Up or not p0Dn: 
                raise RuntimeError, "Missing templates %s_%s_(Up,Dn) for %s" % (p,effect,name)
            p0Up.SetName("%s_%sUp"   % (nominal.GetName(),name))
            p0Dn.SetName("%s_%sDown" % (nominal.GetName(),name))
            report[str(p0Up.GetName())[2:]] = p0Up
            report[str(p0Dn.GetName())[2:]] = p0Dn
            effect0  = "1"
            effect12 = "-"
            if mca._projection != None:
                mca._projection.scaleSystTemplate(name,nominal,p0Up)
                mca._projection.scaleSystTemplate(name,nominal,p0Dn)
        elif mode in ["alternateShape", "alternateShapeOnly"]:
            nominal = report[p]
            alternate = report[effect]
            if mca._projection != None:
                mca._projection.scaleSystTemplate(name,nominal,alternate)
            alternate.SetName("%s_%sUp" % (nominal.GetName(),name))
            if mode == "alternateShapeOnly":
                alternate.Scale(nominal.Integral()/alternate.Integral())
            mirror = nominal.Clone("%s_%sDown" % (nominal.GetName(),name))
            for b in xrange(1,nominal.GetNbinsX()+1):
                y0 = nominal.GetBinContent(b)
                yA = alternate.GetBinContent(b)
                yM = y0
                if (y0 > 0 and yA > 0):
                    yM = y0*y0/yA
                elif yA == 0:
                    yM = 2*y0
                mirror.SetBinContent(b, yM)
            if mode == "alternateShapeOnly":
                # keep same normalization
                mirror.Scale(nominal.Integral()/mirror.Integral())
            else:
                # mirror normalization
                mnorm = (nominal.Integral()**2)/alternate.Integral()
                mirror.Scale(mnorm/alternate.Integral())
            report[alternate.GetName()] = alternate
            report[mirror.GetName()] = mirror
            effect0  = "1"
            effect12 = "-"
        effmap0[p]  = effect0 
        effmap12[p] = effect12 
    systsEnv[name] = (effmap0,effmap12,mode)

## create efficiency scale factors, if needed
if len(masses) > 1 and options.massIntAlgo not in [ "sigmaBR","noeff"]:
    x = ROOT.std.vector('double')()
    y = { "hww": ROOT.std.vector('double')(), "hzz": ROOT.std.vector('double')(), "htt": ROOT.std.vector('double')() }
    for m in 110,115,120,130,135,140:
        x.push_back(m)
        for d in "htt","hww","hzz":
            p = "ttH_%s_%d" % (d,m)
            h0 = report[p] 
            h  = report["ttH_%s" % d] 
            #print "efficiency scale factor %s @ %d = %.3f" % (d,m,h.Integral()/h0.Integral())
            y[d].push_back(h.Integral()/h0.Integral())
        if m == 120:
            x.push_back(125)
            y["hww"].push_back(1)
            y["htt"].push_back(1)
            y["hzz"].push_back(1)
    for d in "htt","hww":#,"hzz": h->ZZ is bad
        splines["efficiency_"+d] = ROOT.ROOT.Math.Interpolator(x,y[d])

if len(masses) > 1:
    for mass in masses:
        smass = str(mass).replace(".0","")
        for p in "ttH_hww ttH_hzz ttH_htt".split():
            scale = getYieldScale(mass,p)
            posts = ['']
            for name,(effmap0,effmap12,mode) in systsEnv.iteritems():
                if mode == "envelop" and effmap0[p] != "-": 
                    posts += [ "_%s%d%s" % (name,i,d) for (i,d) in [(0,'Up'),(0,'Down'),(1,'Up'),(1,'Down'),(2,'Up'),(2,'Down')]]
                elif effmap0[p]  != "-":
                    posts += [ "_%s%s" % (name,d) for d in ['Up','Down']]
                elif effmap12[p] != "-":
                    posts += [ "_%s%d%s" % (name,i,d) for (i,d) in [(1,'Up'),(1,'Down'),(2,'Up'),(2,'Down')]]
            for post in posts:
                template = report["%s%s" % (p,post)].Clone("x_%s%s%s" % (p,smass,post))
                template.Scale(scale)
                if options.massIntAlgo == "full":
                    pythias = [ 110,115,120,125,130,135,140]
                    pythias.sort(key = lambda x : abs(x-mass))
                    mpythia1 = pythias[0]
                    mpythia2 = pythias[1]
                    if mpythia1 > mpythia2: (mpythia1, mpythia2) = (mpythia2, mpythia1)
                    #print "for mass %s, take closest pythias from %d, %d" % (mass, mpythia1, mpythia2)
                    norm = template.Integral()
                    h0_m0 = report[p]
                    h0_m1  = report[("%s_%d" % (p,mpythia1)) if mpythia1 != 125 else p]
                    h0_m2  = report[("%s_%d" % (p,mpythia2)) if mpythia2 != 125 else p]
                    w1 = abs(mass-mpythia2)/abs(mpythia1-mpythia2)
                    w2 = abs(mass-mpythia1)/abs(mpythia1-mpythia2)
                    for b in xrange(1,template.GetNbinsX()+1):
                        avg = w1*h0_m1.GetBinContent(b) + w2*h0_m2.GetBinContent(b)
                        ref = h0_m0.GetBinContent(b)
                        if avg > 0 and ref > 0: 
                            template.SetBinContent(b, template.GetBinContent(b) * avg/ref)
                        #print "bin %d: m1 %7.4f   m2 %7.4f  avg %7.4f   ref %7.4f " % (b, h0_m1.GetBinContent(b), h0_m2.GetBinContent(b), avg, ref)
                    template.Scale(norm/template.Integral())
                    #exit(0)
                report["%s%s%s" % (p,smass,post)] = template
                #print "created x_%s%s%s" % (p,mass,post)
if len(masses) > 1: 
    if not os.path.exists(outdir+"/common"): os.mkdir(outdir+"/common")
for mass in masses:
    smass = str(mass).replace(".0","")
    myout = outdir
    if len(masses) > 1:
        myout += "%s/" % smass 
        if not os.path.exists(myout): os.mkdir(myout)
        myyields = dict([(k,getYieldScale(mass,k)*v) for (k,v) in allyields.iteritems()]) 
        datacard = open(myout+binname+".card.txt", "w"); 
        datacard.write("## Datacard for cut file %s (mass %s)\n"%(args[1],mass))
        datacard.write("shapes *        * ../common/%s.input.root x_$PROCESS x_$PROCESS_$SYSTEMATIC\n" % binname)
        datacard.write("shapes ttH_hww  * ../common/%s.input.root x_$PROCESS$MASS x_$PROCESS$MASS_$SYSTEMATIC\n" % binname)
        datacard.write("shapes ttH_hzz  * ../common/%s.input.root x_$PROCESS$MASS x_$PROCESS$MASS_$SYSTEMATIC\n" % binname)
        datacard.write("shapes ttH_htt  * ../common/%s.input.root x_$PROCESS$MASS x_$PROCESS$MASS_$SYSTEMATIC\n" % binname)
    else:
        myyields = dict([(k,v) for (k,v) in allyields.iteritems()]) 
        datacard = open(myout+binname+".card.txt", "w"); 
        datacard.write("## Datacard for cut file %s\n"%args[1])
        datacard.write("shapes *        * %s.input.root x_$PROCESS x_$PROCESS_$SYSTEMATIC\n" % binname)
    datacard.write('##----------------------------------\n')
    datacard.write('bin         %s\n' % binname)
    datacard.write('observation %s\n' % myyields['data_obs'])
    datacard.write('##----------------------------------\n')
    klen = max([7, len(binname)]+[len(p) for p in procs])
    kpatt = " %%%ds "  % klen
    fpatt = " %%%d.%df " % (klen,3)
    datacard.write('##----------------------------------\n')
    datacard.write('bin             '+(" ".join([kpatt % binname  for p in procs]))+"\n")
    datacard.write('process         '+(" ".join([kpatt % p        for p in procs]))+"\n")
    datacard.write('process         '+(" ".join([kpatt % iproc[p] for p in procs]))+"\n")
    datacard.write('rate            '+(" ".join([fpatt % myyields[p] for p in procs]))+"\n")
    datacard.write('##----------------------------------\n')
    for name,effmap in systs.iteritems():
        datacard.write(('%-12s lnN' % name) + " ".join([kpatt % effmap[p]   for p in procs]) +"\n")
    for name,(effmap0,effmap12,mode) in systsEnv.iteritems():
        if mode == "templates":
            datacard.write(('%-10s shape' % name) + " ".join([kpatt % effmap0[p]  for p in procs]) +"\n")
        if mode == "envelop":
            datacard.write(('%-10s shape' % (name+"0")) + " ".join([kpatt % effmap0[p]  for p in procs]) +"\n")
        if mode in ["envelop", "shapeOnly"]:
            datacard.write(('%-10s shape' % (name+"1")) + " ".join([kpatt % effmap12[p] for p in procs]) +"\n")
            datacard.write(('%-10s shape' % (name+"2")) + " ".join([kpatt % effmap12[p] for p in procs]) +"\n")
if len(masses) > 1:
    myout = outdir
    myyields = dict([(k,-1 if "ttH" in k else v) for (k,v) in allyields.iteritems()]) 
    datacard = open(myout+binname+".card.txt", "w"); 
    datacard.write("## Datacard for cut file %s (all massess, taking signal normalization from templates)\n")
    datacard.write("shapes *        * common/%s.input.root x_$PROCESS x_$PROCESS_$SYSTEMATIC\n" % binname)
    datacard.write("shapes ttH_hww  * common/%s.input.root x_$PROCESS$MASS x_$PROCESS$MASS_$SYSTEMATIC\n" % binname)
    datacard.write("shapes ttH_hzz  * common/%s.input.root x_$PROCESS$MASS x_$PROCESS$MASS_$SYSTEMATIC\n" % binname)
    datacard.write("shapes ttH_htt  * common/%s.input.root x_$PROCESS$MASS x_$PROCESS$MASS_$SYSTEMATIC\n" % binname)
    datacard.write('##----------------------------------\n')
    datacard.write('bin         %s\n' % binname)
    datacard.write('observation %s\n' % myyields['data_obs'])
    datacard.write('##----------------------------------\n')
    klen = max([7, len(binname)]+[len(p) for p in procs])
    kpatt = " %%%ds "  % klen
    fpatt = " %%%d.%df " % (klen,3)
    datacard.write('##----------------------------------\n')
    datacard.write('bin             '+(" ".join([kpatt % binname  for p in procs]))+"\n")
    datacard.write('process         '+(" ".join([kpatt % p        for p in procs]))+"\n")
    datacard.write('process         '+(" ".join([kpatt % iproc[p] for p in procs]))+"\n")
    datacard.write('rate            '+(" ".join([fpatt % myyields[p] for p in procs]))+"\n")
    datacard.write('##----------------------------------\n')
    for name,effmap in systs.iteritems():
        datacard.write(('%-12s lnN' % name) + " ".join([kpatt % effmap[p]   for p in procs]) +"\n")
    for name,(effmap0,effmap12,mode) in systsEnv.iteritems():
        if mode == "templates":
            datacard.write(('%-10s shape' % name) + " ".join([kpatt % effmap0[p]  for p in procs]) +"\n")
        if mode == "envelop":
            datacard.write(('%-10s shape' % (name+"0")) + " ".join([kpatt % effmap0[p]  for p in procs]) +"\n")
        if mode in ["envelop", "shapeOnly"]:
            datacard.write(('%-10s shape' % (name+"1")) + " ".join([kpatt % effmap12[p] for p in procs]) +"\n")
            datacard.write(('%-10s shape' % (name+"2")) + " ".join([kpatt % effmap12[p] for p in procs]) +"\n")
    datacard.close()
    print "Wrote to ",myout+binname+".card.txt"
    if options.verbose:
        print "="*120
        os.system("cat %s.card.txt" % (myout+binname));
        print "="*120

myout = outdir+"/common/" if len(masses) > 1 else outdir;
workspace = ROOT.TFile.Open(myout+binname+".input.root", "RECREATE")
for n,h in report.iteritems():
    if options.verbose: print "\t%s (%8.3f events)" % (h.GetName(),h.Integral())
    workspace.WriteTObject(h,h.GetName())
workspace.Close()

print "Wrote to ",myout+binname+".input.root"


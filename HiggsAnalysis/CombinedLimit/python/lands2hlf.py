import re
from sys import argv, stdout, stderr
from optparse import OptionParser
import ROOT
parser = OptionParser()
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-a", "--asimov", dest="asimov",  default=False, action="store_true")
parser.add_option("-c", "--compiled", dest="cexpr", default=False, action="store_true")
parser.add_option("-b", "--binary",   dest="bin",   default=False, action="store_true", help="produce a Workspace in a rootfile instead of an HLF file")
parser.add_option("-o", "--out",      dest="out",   type="string", default=None,  help="output file (if none, it will print to stdout). Required for binary mode.")

(options, args) = parser.parse_args()

file = open(args[0], "r")
ROOFIT_EXPR = "cexpr" if options.cexpr else "expr"  # change to cexpr to use compiled expressions (faster, but takes more time to start up)

N_OBS_MAX = 10000
nbins      = 1; bins = []
nprocesses = 1; processes = []
nuisances = -1;
obs = []; exp = []; systs = []
keyline = [];  # line that maps each column into bin and process. list of pairs (bin,process,signalT)
binline = []; processline = []; sigline = []
isSignal = {}; signals = [];
for l in file:
    f = l.split();
    if len(f) < 1: continue
    if f[0] == "imax": 
        nbins = int(f[1]) if f[1] != "*" else -1
    if f[0] == "jmax": 
        nprocesses = int(f[1])+1 if f[1] != "*" else -1
    if f[0] == "kmax": 
        nuisances = int(f[1]) if f[1] != "*" else -1
    if f[0] == "Observation" or f[0] == "observation": 
        obs = [ float(x) for x in f[1:] ]
        if nbins == -1: nbins = len(obs)
        if len(obs) != nbins: raise RuntimeError, "Found %d observations but %d bins have been declared" % (len(obs), nbins)
        if binline != []:
            if len(binline) != len(obs): raise RuntimeError, "Found %d bins (%s) but %d bins have been declared" % (len(bins), bins, nbins)
            bins = binline
            obs = dict([(b,obs[i]) for i,b in enumerate(bins)])
            binline = []
    if f[0] == "bin": 
        binline = f[1:] 
    if f[0] == "process": 
        if processline == []: # first line contains names
            processline = f[1:]
            if len(binline) != len(processline): raise RuntimeError, "'bin' line has a different length than 'process' line."
            continue
        sigline = f[1:] # second line contains ids
        if len(sigline) != len(processline): raise RuntimeError, "'bin' line has a different length than 'process' line."
        hadBins = (len(bins) > 0)
        for i,b in enumerate(binline):
            p = processline[i];
            s = (int(sigline[i]) <= 0) # <=0 for signals, >0 for backgrounds
            keyline.append((b, processline[i], s))
            if hadBins:
                if b not in bins: raise RuntimeError, "Bin %s not among the declared bins %s" % (b, bins)
            else:
                if b not in bins: bins.append(b)
            if p not in processes: processes.append(p)
        if nprocesses == -1: nprocesses = len(processes)
        if nbins      == -1: nbins      = len(bins)
        if nprocesses != len(processes): raise RuntimeError, "Found %d processes (%s), declared jmax = %d" % (len(processes),processes,nprocesses)
        if nbins      != len(bins):      raise RuntimeError, "Found %d bins (%s), declared jmax = %d" % (len(bins),bins,nbins)
        exp = dict([(b,{}) for b in bins])
        isSignal = dict([(p,None) for p in processes])
        if obs != [] and type(obs) == list: # still as list, must change into map with bin names
            obs = dict([(b,obs[i]) for i,b in enumerate(bins)])
        for (b,p,s) in keyline:
            if isSignal[p] == None: 
                isSignal[p] = s
            elif isSignal[p] != s:
                raise RuntimeError, "Process %s is declared as signal in some bin and as background in some other bin" % p
        signals = [p for p,s in isSignal.items() if s == True]
        if len(signals) == 0: raise RuntimeError, "You must have at least one signal process (id <= 0)"
    if f[0] == "rate":
        if processline == []: raise RuntimeError, "Missing line with process names before rate line" 
        if sigline == []:     raise RuntimeError, "Missing line with process id before rate line" 
        if len(f[1:]) != len(keyline): raise RuntimeError, "Malformed rate line: length %d, while bins and process lines have length %d" % (len(f[1:]), len(keyline))
        for (b,p,s),r in zip(keyline,f[1:]):
            exp[b][p] = float(r)
        for b in bins:
            np_bin = sum([exp[b][p] for (b1,p,s) in keyline if b1 == b])
            ns_bin = sum([exp[b][p] for (b1,p,s) in keyline if b1 == b and s == True])
            nb_bin = sum([exp[b][p] for (b1,p,s) in keyline if b1 == b and s != True])
            if np_bin == 0: raise RuntimeError, "Bin %s has no processes contributing to it" % b
            if ns_bin == 0: raise RuntimeError, "Bin %s has no signal processes contributing to it" % b
            if nb_bin == 0: raise RuntimeError, "Bin %s has no background processes contributing to it" % b
        break # rate is the last line before nuisances
    
for l in file:
    if l.startswith("--"): continue
    l = re.sub("\\s-+(\\s|$)"," 0\\1",l);
    f = l.split();
    lsyst = f[0]; pdf = f[1]; args = []; numbers = f[2:];
    if pdf == "lnN" or pdf == "gmM":
        pass # nothing special to do
    elif pdf == "gmN":
        args = [int(f[2])]; numbers = f[3:];
    else:
        raise RuntimeError, "Unsupported pdf %s" % pdf
    if len(numbers) < len(keyline): raise RuntimeError, "Malformed systematics line %s of length %d: while bins and process lines have length %d" % (lsyst, len(numbers), len(keyline))
    errline = dict([(b,{}) for b in bins])
    for (b,p,s),r in zip(keyline,numbers):
        if "/" in r: # "number/number"
            if pdf != "lnN": raise RuntimeError, "Asymmetric errors are allowed only for Log-normals"
            errline[b][p] = [ float(x) for x in r.split("/") ]
        else:
            errline[b][p] = float(r) 
    systs.append((lsyst,pdf,args,errline))

if options.stat: 
    nuisances = 0
    systs = []

if options.asimov:
    obs = dict([(b,0) for b in bins])
    for (b,p,s) in keyline: 
        if s == False: obs[b] += exp[b][p]

if nuisances == -1: 
    nuisances = len(systs)
elif len(systs) != nuisances: 
    raise RuntimeError, "Found %d systematics, expected %d" % (len(systs), nuisances)

out = stdout; 
if options.bin:
    if options.out != None:
        ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit.so")
        out = ROOT.RooWorkspace("w","w");
        out.dont_delete = []
    else:
        raise RuntimeException, "You need to specify an output file when using binary mode";
elif options.out != None:
    stderr.write("Will save workspace to HLF file %s" % options.out)
    out = open(options.out, "w");

def factory_(X):
    global out
    ret = out.factory(X);
    if ret: out.dont_delete.append(ret)
    else:
        print "ERROR parsing '%s'" % X
        out.W.Print("V");
        raise RuntimeError, "Error in factory statement" 

def doComment(X):
    global out
    if not options.bin: out.write("// "+X+"\n");
def doVar(vardef):
    global out
    if options.bin: factory_(vardef);
    else: out.write(vardef+";\n");
def doSet(name,vars):
    global out
    if options.bin: out.defineSet(name,vars)
    else: out.write("%s = set(%s);\n" % (name,vars));
def doObj(name,type,X):
    global out
    if options.bin: factory_("%s::%s(%s)" % (type, name, X));
    else: out.write("%s = %s(%s);\n" % (name, type, X))
            
if len(obs):
    doComment(" ----- observables (already set to asimov values) -----")
    for b in bins: doVar("n_obs_bin%s[%f,0,%d]" % (b,obs[b],N_OBS_MAX))
else:
    doComment(" ----- observables -----")
    for b in bins: doVar("n_obs_bin%s[0,%d]" % (b,N_OBS_MAX))
doSet("observables", ",".join(["n_obs_bin%s" % b for b in bins]))

doComment(" ----- parameters of interest -----");
doComment(" --- Signal Strength --- ");
doVar("r[0,20];");
doComment(" --- set of all parameters of interest --- ");
doSet("POI","r");

if nuisances: 
    doComment(" ----- nuisances -----")
    globalobs = []
    for (n,pdf,args,errline) in systs: 
        if pdf == "lnN":
            #print "thetaPdf_%s = Gaussian(theta_%s[-5,5], 0, 1);" % (n,n)
            doObj("thetaPdf_%s" % n, "Gaussian", "theta_%s[-5,5], thetaIn_%s[0], 1" % (n,n));
            globalobs.append("thetaIn_%s" % n)
        elif pdf == "gmM":
            val = 0;
            for c in errline.values(): #list channels
              for v in c.values():     # list effects in each channel 
                if v != 0:
                    if val != 0 and v != val: 
                        raise RuntimeError, "Error: line %s contains two different uncertainties %g, %g, which is not supported for gmM" % (n,v,val)
                    val = v;
            if val == 0: raise RuntimeError, "Error: line %s contains all zeroes"
            theta = val*val; kappa = 1/theta
            doObj("thetaPdf_%s" % n, "Gamma", "theta_%s[1,%f,%f], %g, %g, 0" % (n, max(0.01,1-5*val), 1+5*val, kappa, theta))
        elif pdf == "gmN":
            doObj("thetaPdf_%s" % n, "Poisson", "thetaIn_%s[%d], theta_%s[0,%d]" % (n,args[0],n,2*args[0]+5))
            globalobs.append("thetaIn_%s" % n)
    doSet("nuisances", ",".join(["theta_%s"    % n for (n,p,a,e) in systs]))
    doObj("nuisancePdf", "PROD", ",".join(["thetaPdf_%s" % n for (n,p,a,e) in systs]))
    if globalobs:
        doSet("globalObservables", ",".join(globalobs))

doComment(" --- Expected events in each bin, for each process ----")
for b in bins:
    for p in exp[b].keys(): # so that we get only processes contributing to this bin
        # collect multiplicative corrections
        strexpr = ""; strargs = ""
        gammaNorm = None; iSyst=-1
        for (n,pdf,args,errline) in systs:
            if not errline[b].has_key(p): continue
            if errline[b][p] == 0.0: continue
            if pdf == "lnN" and errline[b][p] == 1.0: continue
            iSyst += 1
            if pdf == "lnN":
                if type(errline[b][p]) == list:
                    elow, ehigh = errline[b][p];
                    strexpr += " * @%d" % iSyst
                    strargs += ", AsymPow(%f,%f,theta_%s)" % (elow, ehigh, n)
                else:
                    strexpr += " * pow(%f,@%d)" % (errline[b][p], iSyst)
                    strargs += ", theta_%s" % n
            elif pdf == "gmM":
                strexpr += " * @%d " % iSyst
                strargs += ", theta_%s" % n
            elif pdf == "gmN":
                strexpr += " * @%d " % iSyst
                strargs += ", theta_%s" % n
                if abs(errline[b][p] * args[0] - exp[b][p]) > max(0.02 * max(exp[b][p],1), errline[b][p]):
                    raise RuntimeError, "Values of N = %d, alpha = %g don't match with expected rate %g for systematics %s " % (
                                            args[0], errline[b][p], exp[b][p], n)
                if gammaNorm != None:
                    raise RuntimeError, "More than one gmN uncertainty for the same bin and process (second one is %s)" % n
                gammaNorm = "%g" % errline[b][p]
        # set base term (fixed or gamma)
        if gammaNorm != None:
            strexpr = gammaNorm + strexpr
        else:
            strexpr = str(exp[b][p]) + strexpr
        # optimize constants
        if strargs != "":
            doObj("n_exp_bin%s_proc_%s" % (b,p), ROOFIT_EXPR, "'%s'%s" % (strexpr, strargs));
        else:
            doVar("n_exp_bin%s_proc_%s[%g]" % (b, p, exp[b][p]))
    doObj("n_exp_bin%s_bonly" % b, "sum", ", ".join(["n_exp_bin%s_proc_%s" % (b,p) for p in exp[b].keys() if isSignal[p] == False]) )
    if len(signals) == 1:
        doObj("n_exp_bin%s" % b, "sum", "prod(r, n_exp_bin%s_proc_%s), n_exp_bin%s_bonly" % (b,signals[0],b))
    else:
        sigsum = ", ".join(["n_exp_bin%s_proc_%s" % (b,p) for p in exp[b].keys() if isSignal[p] == True])  
        doObj("n_exp_bin%s" % b, "sum", "prod(r, sum(%s)), n_exp_bin%s_bonly" % (sigsum,b))

doComment(" --- Expected events in each bin, total (S+B and B) ----")
for b in bins:
    doObj("pdf_bin%s"       % b, "Poisson", "n_obs_bin%s, n_exp_bin%s"       % (b,b))
    doObj("pdf_bin%s_bonly" % b, "Poisson", "n_obs_bin%s, n_exp_bin%s_bonly" % (b,b))

prefix = "modelObs"
if not nuisances: prefix = "model" # we can make directly the model
if nbins > 50:
    from math import ceil
    nblocks = int(ceil(nbins/10.))
    for i in range(nblocks):
        doObj("%s_s_%d" % (prefix,i), "PROD", ",".join(["pdf_bin%s"       % bins[j] for j in range(10*i,min(nbins,10*i+10))]))
        doObj("%s_b_%d" % (prefix,i), "PROD", ",".join(["pdf_bin%s_bonly" % bins[j] for j in range(10*i,min(nbins,10*i+10))]))
    doObj("%s_s" % prefix, "PROD", ",".join([prefix+"_s_%d" % i for i in range(nblocks)]))
    doObj("%s_b" % prefix, "PROD", ",".join([prefix+"_b_%d" % i for i in range(nblocks)]))
else: 
    doObj("%s_s" % prefix, "PROD", ",".join(["pdf_bin%s"       % b for b in bins]))
    doObj("%s_b" % prefix, "PROD", ",".join(["pdf_bin%s_bonly" % b for b in bins]))

if nuisances: # multiply by nuisances if needed
    doObj("model_s", "PROD", "modelObs_s, nuisancePdf")
    doObj("model_b", "PROD", "modelObs_b, nuisancePdf")

if options.bin:
    if options.out != None: 
        mc_s = ROOT.RooStats.ModelConfig("ModelConfig",   out)
        mc_b = ROOT.RooStats.ModelConfig("ModelConfig_b", out)
        for (l,mc) in [ ('s',mc_s), ('b',mc_b) ]:
            mc.SetPdf(out.pdf("model_"+l))
            mc.SetParametersOfInterest(out.set("POI"))
            mc.SetObservables(out.set("observables"))
            if nuisances:  mc.SetNuisanceParameters(out.set("nuisances"))
            if out.set("globalObservables"): mc.SetGlobalObservables(out.set("globalObservables"))
            getattr(out,"import")(mc, mc.GetName())
        out.writeToFile(options.out)
    else: raise RuntimeException, "You need to specify an output file when using binary mode";


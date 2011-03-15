import re
from sys import argv
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true")  # ignore systematic uncertainties to consider statistical uncertainties only
parser.add_option("-a", "--asimov", dest="asimov",  default=False, action="store_true")
parser.add_option("-c", "--compiled", dest="cexpr", default=False, action="store_true")
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
        if bins != []:
            obs = dict([(b,obs[i]) for i,b in enumerate(bins)])
    if f[0] == "bin": 
        if obs == []: # (optional) bin line before observation
            bins = f[1:]
            if nbins == -1: nbins = len(bins)
            if len(bins) != nbins: raise RuntimeError, "Found %d bins (%s) but %d bins have been declared" % (len(bins), bins, nbins)
        else: 
            binline = f[1:] # binline before processes
        #if len(f[1:]) != nbins * nprocesses: raise RuntimeError, "Malformed bin line: len %d, while nbins*nprocesses = %d*%d" % (len(f[1:]), nbins,nprocesses)
        #for (i,x) in enumerate(f[1:]):
        #    if x != str(i/nprocesses+1): raise RuntimeError, "Malformed bin line for %d nprocesses: %s" % (nprocesses, line)
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
        if type(obs) == list: # still as list, must change into map with bin names
            obs = dict([(b,obs[i]) for i,b in enumerate(bins)])
        for (b,p,s) in keyline:
            if isSignal[p] == None: 
                isSignal[p] = s
            elif isSignal[p] != s:
                raise RuntimeError, "Process %s is declared as signal in some bin and as background in some other bin" % p
        signals = [p for p,s in isSignal.items() if s == True]
        if len(signals) == 0: raise RuntimeError, "You must have at least one signal process (id <= 0)"
    if f[0] == "rate":
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


if len(obs):
    print "/// ----- observables (already set to asimov values) -----"
    for b in bins: print "n_obs_bin%s[%f,0,%d];" % (b,obs[b],N_OBS_MAX)
else:
    print "/// ----- observables -----"
    for b in bins: print "n_obs_bin%s[0,%d];" % (b,N_OBS_MAX)
print "observables = set(", ",".join(["n_obs_bin%s" % b for b in bins]),");"

print """
/// ----- parameters of interest -----
// signal strength
r[0,20];
// set of all parameters of interest
POI = set(r);
"""

if nuisances: 
    print "/// ----- nuisances -----"
    globalobs = []
    for (n,pdf,args,errline) in systs: 
        if pdf == "lnN":
            #print "thetaPdf_%s = Gaussian(theta_%s[-5,5], 0, 1);" % (n,n)
            print "thetaPdf_%s = Gaussian(theta_%s[-5,5], thetaIn_%s[0], 1);" % (n,n,n)
            globalobs.append("thetaIn_%s" % n)
        elif pdf == "gmM":
            val = 0;
            for v in sum(errline,[]): # concatenate all numbers
                if v != 0:
                    if val != 0 and v != val: 
                        raise RuntimeError, "Error: line %s contains two different uncertainties %g, %g, which is not supported for gmM" % (n,v,val)
                    val = v;
            if val == 0: raise RuntimeError, "Error: line %s contains all zeroes"
            theta = val*val; kappa = 1/theta
            print "thetaPdf_%s = Gamma(theta_%s[1,%f,%f], %g, %g, 0);" % (n, n, max(0.01,1-5*val), 1+5*val, kappa, theta)
        elif pdf == "gmN":
            print "thetaPdf_%s = Poisson(thetaIn_%s[%d], theta_%s[0,%d]);" % (n,n,args[0],n,2*args[0]+5)
            globalobs.append("thetaIn_%s" % n)
    print "nuisances   =  set(", ",".join(["theta_%s"    % n for (n,p,a,e) in systs]),");"
    print "nuisancePdf = PROD(", ",".join(["thetaPdf_%s" % n for (n,p,a,e) in systs]),");"
    if globalobs:
        print "globalObservables =  set(", ",".join(globalobs),");"

print "/// --- Expected events in each bin, for each process ----"
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
            print "n_exp_bin%s_proc_%s = %s('%s'%s);" % (b, p, ROOFIT_EXPR, strexpr, strargs)
        else:
            print "n_exp_bin%s_proc_%s[%g];" % (b, p, exp[b][p])
    print "n_exp_bin%s_bonly  = sum(" % b + ", ".join(["n_exp_bin%s_proc_%s" % (b,p) for p in exp[b].keys() if isSignal[p] == False]) + ");";
    if len(signals) == 1:
        print "n_exp_bin%s        = sum(prod(r, n_exp_bin%s_proc_%s), n_exp_bin%s_bonly);" % (b,b,signals[0],b);
    else:
        sigsum = ", ".join(["n_exp_bin%s_proc_%s" % (b,p) for p in exp[b].keys() if isSignal[p] == True])  
        print "n_exp_bin%s        = sum(prod(r, sum(%s)), n_exp_bin%s_bonly);" % (b,sigsum,b);

print "/// --- Expected events in each bin, total (S+B and B) ----"
for b in bins:
    print "pdf_bin%s       = Poisson(n_obs_bin%s, n_exp_bin%s);"       % (b,b,b);
    print "pdf_bin%s_bonly = Poisson(n_obs_bin%s, n_exp_bin%s_bonly);" % (b,b,b);

prefix = "modelObs"
if not nuisances: prefix = "model" # we can make directly the model
if nbins > 50:
    from math import ceil
    nblocks = int(ceil(nbins/10.))
    for i in range(nblocks):
        print prefix+"_s_%d = PROD("%i, ",".join(["pdf_bin%s"       % bins[j] for j in range(10*i,min(nbins,10*i+10))]),");"
        print prefix+"_b_%d = PROD("%i, ",".join(["pdf_bin%s_bonly" % bins[j] for j in range(10*i,min(nbins,10*i+10))]),");"
    print prefix+"_s = PROD(", ",".join([prefix+"_s_%d" % i for i in range(nblocks)]),");"
    print prefix+"_b = PROD(", ",".join([prefix+"_b_%d" % i for i in range(nblocks)]),");"
else: 
    print prefix+"_s = PROD({", ",".join(["pdf_bin%s"       % b for b in bins]),"});"
    print prefix+"_b = PROD({", ",".join(["pdf_bin%s_bonly" % b for b in bins]),"});"

if nuisances: # multiply by nuisances if needed
    print "model_s = PROD(modelObs_s, nuisancePdf);"
    print "model_b = PROD(modelObs_b, nuisancePdf);"

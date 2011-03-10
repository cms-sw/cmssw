import re
from sys import argv
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--stat",   dest="stat",    default=False, action="store_true")
parser.add_option("-a", "--asimov", dest="asimov",  default=False, action="store_true")
parser.add_option("-c", "--compiled", dest="cexpr", default=False, action="store_true")
(options, args) = parser.parse_args()

file = open(args[0], "r")
ROOFIT_EXPR = "cexpr" if options.cexpr else "expr"  # change to cexpr to use compiled expressions (faster, but takes more time to start up)

N_OBS_MAX = 10000
bins      = 1
processes = 1
nuisances = -1;
obs = []; exp = []; systs = []
for l in file:
    f = l.split();
    if len(f) < 1: continue
    if f[0] == "imax": bins      = int(f[1])
    if f[0] == "jmax": processes = int(f[1])+1
    if f[0] == "kmax": nuisances = int(f[1]) if f[1] != "*" else -1;
    if f[0] == "Observation": 
        obs = [ float(x) for x in f[1:] ]
        if len(obs) != bins: raise RuntimeError, "Found %d observations but %d bins" % (len(obs), bins)
    if f[0] == "bin": 
        if len(f[1:]) != bins * processes: raise RuntimeError, "Malformed bin line: len %d, while bins*processes = %d*%d" % (len(f[1:]), bins,processes)
        for (i,x) in enumerate(f[1:]):
            if x != str(i/processes+1): raise RuntimeError, "Malformed bin line for %d processes: %s" % (processes, line)
    if f[0] == "process": 
        if len(f[1:]) < bins * processes: raise RuntimeError, "Malformed process line: len %d, while bins*processes = %d*%d" % (len(f[1:]), bins,processes)
    if f[0] == "rate":
        if len(f[1:]) < bins * processes: raise RuntimeError, "Malformed rate line: %d, while bins*processes = %d*%d" % (len(f[1:]), bins,processes)
        for b in range(bins):
            exp.append([float(f[1+(b*processes + p)]) for p in range(processes)])
        break
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
    if len(numbers) < bins * processes: raise RuntimeError, "Malformed rate line: %d, while bins*processes = %d*%d" % (len(f[1:]), bins,processes)
    errline = []
    for b in range(bins):
        errline.append([float(numbers[b*processes + p]) for p in range(processes)])
    systs.append((lsyst,pdf,args,errline))

if options.stat: 
    nuisances = 0
    systs = []

if options.asimov:
    obs = [sum(r[1:]) for r in exp]

if nuisances == -1: 
    nuisances = len(systs)
elif len(systs) != nuisances: 
    raise RuntimeError, "Found %d systematics, expected %d" % (len(systs), nuisances)


if len(obs):
    print "/// ----- observables (already set to asimov values) -----"
    for b in range(bins): print "n_obs_bin%d[%f,0,%d];" % (b,obs[b],N_OBS_MAX)
else:
    print "/// ----- observables -----"
    for b in range(bins): print "n_obs_bin%d[0,%d];" % (b,N_OBS_MAX)
print "observables = set(", ",".join(["n_obs_bin%d" % b for b in range(bins)]),");"

print """
/// ----- parameters of interest -----
// signal strenght
r[0,20];
// set of all paramers of interest
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
for b in range(bins):
    for p in range(processes):
        # collect multiplicative corrections
        strexpr = ""; strargs = ""
        gammaNorm = None
        for (n,pdf,args,errline) in systs:
            if errline[b][p] == 0.0: continue
            if pdf == "lnN" and errline[b][p] == 1.0: continue
            strargs += ", theta_%s" % n
            if pdf == "lnN":
                strexpr += " * pow(%f,theta_%s)" % (errline[b][p], n)
            elif pdf == "gmM":
                strexpr += " * theta_%s" % n
            elif pdf == "gmN":
                strexpr += " * %g " % errline[b][p]
                if abs(errline[b][p] * args[0] - exp[b][p]) > 1e-3:
                    raise RuntimeError, "Values of N = %d, alpha = %g don't match with expected rate %g for systematics %s " % (
                                            args[0], errline[b][p], exp[b][p], n)
                if gammaNorm != None:
                    raise RuntimeError, "More than one gmN uncertainty for the same bin and process (theta_%s, %s) " % (n, gammaNorm)
                gammaNorm = "theta_"+n
        # set base term (fixed or gamma)
        if gammaNorm != None:
            strexpr = gammaNorm + strexpr
        else:
            strexpr = str(exp[b][p]) + strexpr
        # optimize constants
        if strargs != "":
            print "n_exp_bin%d_proc%d = %s('%s'%s);" % (b, p, ROOFIT_EXPR, strexpr, strargs)
        else:
            print "n_exp_bin%d_proc%d[%g];" % (b, p, exp[b][p])
    expr_sb = "+".join(["n_exp_bin%d_proc%d" % (b,p) for p in range(0,processes)])
    expr_b  = "+".join(["n_exp_bin%d_proc%d" % (b,p) for p in range(1,processes)])
    args_sb = ",".join(["n_exp_bin%d_proc%d" % (b,p) for p in range(0,processes)]) 
    args_b  = ",".join(["n_exp_bin%d_proc%d" % (b,p) for p in range(1,processes)]) 
    print "n_exp_bin%d       = %s('@0*%s',r, %s);" % (b, ROOFIT_EXPR, expr_sb, args_sb);
    print "n_exp_bin%d_bonly = %s('   %s',   %s);" % (b, ROOFIT_EXPR, expr_b,  args_b );

print "/// --- Expected events in each bin, total (S+B and B) ----"
for b in range(bins):
    print "pdf_bin%d       = Poisson(n_obs_bin%d, n_exp_bin%d);"       % (b,b,b);
    print "pdf_bin%d_bonly = Poisson(n_obs_bin%d, n_exp_bin%d_bonly);" % (b,b,b);

prefix = "modelObs"
if not nuisances: prefix = "model" # we can make directly the model
if bins > 50:
    from math import ceil
    nblocks = int(ceil(bins/10.))
    for i in range(nblocks):
        print prefix+"_s_%d = PROD("%i, ",".join(["pdf_bin%d"       % b for b in range(10*i,min(bins,10*i+10))]),");"
        print prefix+"_b_%d = PROD("%i, ",".join(["pdf_bin%d_bonly" % b for b in range(10*i,min(bins,10*i+10))]),");"
    print prefix+"_s = PROD(", ",".join([prefix+"_s_%d" % b for b in range(nblocks)]),");"
    print prefix+"_b = PROD(", ",".join([prefix+"_b_%d" % b for b in range(nblocks)]),");"
else: 
    print prefix+"_s = PROD({", ",".join(["pdf_bin%d"       % b for b in range(min(50,bins))]),"});"
    print prefix+"_b = PROD({", ",".join(["pdf_bin%d_bonly" % b for b in range(min(50,bins))]),"});"

if nuisances: # multiply by nuisances if needed
    print "model_s = PROD(modelObs_s, nuisancePdf);"
    print "model_b = PROD(modelObs_b, nuisancePdf);"

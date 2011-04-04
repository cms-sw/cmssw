import re

class Datacard():
    def __init__(self):
        self.bins = []
        self.obs  = [] # empty or map bin -> value
        self.processes = []; self.signals = []; self.isSignal = {}
        self.keyline = []
        self.exp     = {}  # map bin -> (process -> value)
        self.systs   = []  # list (name, pdf, args, errline)
                           # errline: map bin -> (process -> value)
        self.shapeMap = {} # map channel -> (process -> [fname, hname, hname_syst])
        self.hasShape = False

def parseCard(file, options):
    ret = Datacard()
    #
    nbins      = -1; 
    nprocesses = -1; 
    nuisances  = -1;
    binline = []; processline = []; sigline = []
    for l in file:
        f = l.split();
        if len(f) < 1: continue
        if f[0] == "imax": 
            nbins = int(f[1]) if f[1] != "*" else -1
        if f[0] == "jmax": 
            nprocesses = int(f[1])+1 if f[1] != "*" else -1
        if f[0] == "kmax": 
            nuisances = int(f[1]) if f[1] != "*" else -1
        if f[0] == "shapes":
            if not options.bin: raise RuntimeError, "Can use shapes only with binary output mode"
            if len(f) < 5: raise RuntimeError, "Malformed shapes line"
            if not ret.shapeMap.has_key(f[2]): ret.shapeMap[f[2]] = {}
            if ret.shapeMap[f[2]].has_key(f[1]): raise RuntimeError, "Duplicate definition for process '%s', channel '%s'" % (f[1], f[2])
            ret.shapeMap[f[2]][f[1]] = f[3:]
        if f[0] == "Observation" or f[0] == "observation": 
            ret.obs = [ float(x) for x in f[1:] ]
            if nbins == -1: nbins = len(ret.obs)
            if len(ret.obs) != nbins: raise RuntimeError, "Found %d observations but %d bins have been declared" % (len(ret.obs), nbins)
            if binline != []:
                if len(binline) != len(ret.obs): raise RuntimeError, "Found %d bins (%s) but %d bins have been declared" % (len(ret.bins), ret.bins, nbins)
                ret.bins = binline
                ret.obs = dict([(b,ret.obs[i]) for i,b in enumerate(ret.bins)])
                binline = []
        if f[0] == "bin": 
            binline = f[1:] 
        if f[0] == "process": 
            if processline == []: # first line contains names
                processline = f[1:]
                if len(binline) != len(processline): raise RuntimeError, "'bin' line has a different length than 'process' line."
                continue
            sigline = f[1:] # second line contains ids
            if re.match("-?[0-9]+", processline[0]) and not re.match("-?[0-9]+", sigline[0]):
                (processline,sigline) = (sigline,processline)
            if len(sigline) != len(processline): raise RuntimeError, "'bin' line has a different length than 'process' line."
            hadBins = (len(ret.bins) > 0)
            for i,b in enumerate(binline):
                p = processline[i];
                s = (int(sigline[i]) <= 0) # <=0 for signals, >0 for backgrounds
                ret.keyline.append((b, processline[i], s))
                if hadBins:
                    if b not in ret.bins: raise RuntimeError, "Bin %s not among the declared bins %s" % (b, ret.bins)
                else:
                    if b not in ret.bins: ret.bins.append(b)
                if p not in ret.processes: ret.processes.append(p)
            if nprocesses == -1: nprocesses = len(ret.processes)
            if nbins      == -1: nbins      = len(ret.bins)
            if nprocesses != len(ret.processes): raise RuntimeError, "Found %d processes (%s), declared jmax = %d" % (len(ret.processes),ret.processes,nprocesses)
            if nbins      != len(ret.bins):      raise RuntimeError, "Found %d bins (%s), declared jmax = %d" % (len(ret.bins),ret.bins,nbins)
            ret.exp = dict([(b,{}) for b in ret.bins])
            ret.isSignal = dict([(p,None) for p in ret.processes])
            if ret.obs != [] and type(ret.obs) == list: # still as list, must change into map with bin names
                ret.obs = dict([(b,ret.obs[i]) for i,b in enumerate(ret.bins)])
            for (b,p,s) in ret.keyline:
                if ret.isSignal[p] == None: 
                    ret.isSignal[p] = s
                elif ret.isSignal[p] != s:
                    raise RuntimeError, "Process %s is declared as signal in some bin and as background in some other bin" % p
            ret.signals = [p for p,s in ret.isSignal.items() if s == True]
            if len(ret.signals) == 0: raise RuntimeError, "You must have at least one signal process (id <= 0)"
        if f[0] == "rate":
            if processline == []: raise RuntimeError, "Missing line with process names before rate line" 
            if sigline == []:     raise RuntimeError, "Missing line with process id before rate line" 
            if len(f[1:]) != len(ret.keyline): raise RuntimeError, "Malformed rate line: length %d, while bins and process lines have length %d" % (len(f[1:]), len(ret.keyline))
            for (b,p,s),r in zip(ret.keyline,f[1:]):
                ret.exp[b][p] = float(r)
            break # rate is the last line before nuisances
    # parse nuisances   
    for l in file:
        if l.startswith("--"): continue
        l  = re.sub("\\s*#.*","",l)
        l = re.sub("(?<=\\s)-+(\\s|$)"," 0\\1",l);
        f = l.split();
        if len(f) <= 1: continue
        lsyst = f[0]; pdf = f[1]; args = []; numbers = f[2:];
        if re.match("[0-9]+",lsyst): lsyst = "theta"+lsyst
        if pdf == "lnN" or pdf == "gmM" or pdf.startswith("shape"):
            pass # nothing special to do
        elif pdf == "gmN":
            args = [int(f[2])]; numbers = f[3:];
        elif pdf == "param":
            # for parametric uncertainties, there's no line to account per bin/process effects
            # just assume everything else is an argument and move on
            args = f[2:]
            if len(args) <= 1: raise RuntimeError, "Uncertainties of type 'param' must have at least two arguments (mean and sigma)"
            ret.systs.append((lsyst,pdf,args,[]))
            continue
        else:
            raise RuntimeError, "Unsupported pdf %s" % pdf
        if len(numbers) < len(ret.keyline): raise RuntimeError, "Malformed systematics line %s of length %d: while bins and process lines have length %d" % (lsyst, len(numbers), len(ret.keyline))
        errline = dict([(b,{}) for b in ret.bins])
        nonNullEntries = 0 
        for (b,p,s),r in zip(ret.keyline,numbers):
            if "/" in r: # "number/number"
                if pdf != "lnN": raise RuntimeError, "Asymmetric errors are allowed only for Log-normals"
                errline[b][p] = [ float(x) for x in r.split("/") ]
            else:
                errline[b][p] = float(r) 
            # set the rate to epsilon for backgrounds with zero observed sideband events.
            if pdf == "gmN" and ret.exp[b][p] == 0 and float(r) != 0: ret.exp[b][p] = 1e-6
        ret.systs.append((lsyst,pdf,args,errline))
    # check if there are bins with no rate
    for b in ret.bins:
        np_bin = sum([(ret.exp[b][p] != 0) for (b1,p,s) in ret.keyline if b1 == b])
        ns_bin = sum([(ret.exp[b][p] != 0) for (b1,p,s) in ret.keyline if b1 == b and s == True])
        nb_bin = sum([(ret.exp[b][p] != 0) for (b1,p,s) in ret.keyline if b1 == b and s != True])
        if np_bin == 0: raise RuntimeError, "Bin %s has no processes contributing to it" % b
        if ns_bin == 0: raise RuntimeError, "Bin %s has no signal processes contributing to it" % b
        if nb_bin == 0: raise RuntimeError, "Bin %s has no background processes contributing to it" % b
    # cleanup systematics that have no effect to avoid zero derivatives
    syst2 = []
    for lsyst,pdf,args,errline in ret.systs:
        nonNullEntries = 0 
        if pdf == "param": # this doesn't have an errline
            syst2.append((lsyst,pdf,args,errline))
            continue
        for (b,p,s) in ret.keyline:
            r = errline[b][p]
            nullEffect = (r == 0.0 or (pdf == "lnN" and r == 1.0))
            if not nullEffect and ret.exp[b][p] != 0: nonNullEntries += 1 # is this a zero background?
        if nonNullEntries != 0: syst2.append((lsyst,pdf,args,errline))
        elif nuisances != -1: nuisances -= 1 # remove from count of nuisances, since qe skipped it
    ret.systs = syst2
    # remove them if options.stat asks so
    if options.stat: 
        nuisances = 0
        ret.systs = []
    # check number of nuisances
    if nuisances == -1: 
        nuisances = len(ret.systs)
    elif len(ret.systs) != nuisances: 
        raise RuntimeError, "Found %d systematics, expected %d" % (len(ret.systs), nuisances)
    # set boolean to know about shape
    ret.hasShapes = (len(ret.shapeMap) > 0)
    # return result
    return ret

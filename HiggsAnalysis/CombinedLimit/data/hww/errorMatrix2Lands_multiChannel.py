import re
from sys import argv
from math import *
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--stat",     dest="stat",     default=False, action="store_true")
parser.add_option("-S", "--signal",   dest="signal",   default=False, action="store_true")
parser.add_option("-a", "--asimov",   dest="asimov",   default=False, action="store_true")
parser.add_option("-F", "--floor",    dest="floor",    default=False, action="store_true")
parser.add_option("-o", "--optimize", dest="optimize", default=False, action="store_true", help="Remove processes or systematics that don't contribute")
parser.add_option("-g", "--gamma", dest="gamma", default=False, action="store_true", help="Use gammas for statistical uncertainties")
parser.add_option("--gmM",   dest="gmM",  type="string", action="append", default=[], help="Use multiplicative gamma for this systematics (can use multiple times)")
parser.add_option("-l", "--label",    dest="label",    type="string", default="hww")
parser.add_option("-L", "--label-syst",    dest="labelSyst", default=False, action="store_true", help="Give names to systematics")
parser.add_option("-4", "--4th-gen",  dest="sm4",      default=False, action="store_true")
parser.add_option("-n", "--nch",      dest="nch",      type="int",    default=1)
parser.add_option("-B", "--b-fluct",  dest="bfluct",   type="float",  default=0)
(options, args) = parser.parse_args()

if len(args) < 1: raise RuntimeError, "Usage: errorMatrix2Lands.py [options] errorMatrix.txt "

file = open(args[0], "r")

data = {} 

def iddify(x):
    id = x
    slang = { "[Cc]ross[ -][Ss]ection":"XS", 
              "[Tt]rigger(\s+[Ee]ff(s|ic(ienc(y|ies))?))?":"Trig",
              "([Ii]ntegrated\s+)?[Ll]umi(nosity)":"Lumi",
              "[Ee]ff(s|ic(ienc(y|ies))?)":"Eff",
              "[Nn]orm(alization)":"Norm",
              "[Rr]eco(nstruction)":"Reco",
              "[Ss]stat(istics?)":"Stat"}
    for s,r in slang.items(): id = re.sub(s,r,id)
    return re.sub("[^A-Za-z0-9_]", "", re.sub("\s+|-|/","_", re.sub("^\s+|\s+$","",id)))
     
header = file.readline() # skip header
processnames = header.split()[2:]
nproc = len(processnames)
if (options.nch > 1):
    processnames = header.split()[1:]
    nproc = len(processnames)/options.nch-1
    processnames = processnames[1:(nproc+1)]
    print processnames
# read yields
for l in file:
    l = re.sub("--+","0",l)
    m = re.match(r"(\d+) Yield\s+((\d+\.?\d*(E[+\-]\d+)?\s+)+)", l)
    if not m: break
    mh = m.group(1)
    yields = [float(x) for x in m.group(2).split()];
    line = []
    if len(yields) != (options.nch * (nproc+1)): 
        raise RuntimeError, "len(yields) = %d != options.nch * (nproc + 1) = %d * (%d+1)" % (len(yields),options.nch,nproc)
    data[mh] = { 'nch':0, 'obs':[], 'exp':[], 'processnames':[], 'nuis':[], 'nuisname':[]}
    for i in range(options.nch):
        start = i*(nproc+1); end = start + nproc + 1
        data[mh]['nch'] += 1
        data[mh]['obs'] += [yields[start]]
        data[mh]['exp'] +=  yields[start+1:end]
        data[mh]['processnames'] += processnames[:]

# read nuisances
if not options.stat:
    for l in file:
        if l.rstrip().rstrip() == "": continue
        l = re.sub("--+","0",l)
        m = re.match(r"(.*?)\s+(0\s+(\d+\.?\d*(E[+\-]\d+)?\s+)+)", l)
        if m == None: raise ValueError, "Missing line "+l
        sysname = m.group(1)
        syseff  = [float(x) for x in m.group(2).split()]
        if len(syseff) != (options.nch * (nproc+1)): 
            raise RuntimeError, "len(syseff) = %d != options.nch * (nproc + 1) = %d * (%d+1)" % (len(syseff),options.nch,nproc)
        # decide which selections are affected
        mhs = data.keys()
        if re.match(r"\d{3}\s+.*?", sysname) and data.has_key(sysname[0:3]):
            mhs = [ sysname[0:3] ]
        for mh in mhs:
            nuisline = []
            for i in range(options.nch):
                start = i*(nproc+1); end = start + nproc + 1
                nuisline += syseff[start+1:end] 
            # special case: the stats line have to be expanded in N independent systematics
            if sysname != mh+" Stats":
                data[mh]['nuis'].append(nuisline)
                data[mh]['nuisname'].append(sysname)
            else:   
                for k in range(0, nproc * options.nch):
                    if nuisline[k] < 0.005: continue
                    data[mh]['nuis'].append([(x if j == k else 0) for j,x in enumerate(nuisline)])
                    data[mh]['nuisname'].append('Statistics in MC or control sample')

if options.asimov:
    for mh in data.keys():
        for i in range(options.nch):
            start = i*(nproc); end = start + nproc
            data[mh]['obs'][i] = sum(data[mh]['exp'][start+1:end])
            if options.bfluct != 0:
                data[mh]['obs'][i] = max(0, data[mh]['obs'][i] + options.bfluct * sqrt(data[mh]['obs'][i]))
            if options.signal: data[mh]['obs'][i] += data[mh]['exp'][start]
            if options.floor:  data[mh]['obs'][i] = floor(data[mh]['obs'][i])
    

if options.optimize:
    for mh in data.keys():
        # step 1: strip processes with no rate
        hasrate = [0 for p in range(nproc)]
        for i in range(options.nch): 
            for p in range(nproc):  
                if data[mh]['exp'][p] != 0: hasrate[p] = 1
        for p in range(nproc):
            if hasrate[p] == 0: print "For mH = ", mh , " process ", processnames[p], " does not contribute."
        data[mh]['nuis']         = [ [x for i,x in enumerate(n) if hasrate[i%nproc]] for n in data[mh]['nuis'] ]
        data[mh]['processnames'] = [  x for i,x in enumerate(data[mh]['processnames']) if hasrate[i%nproc] ]
        data[mh]['exp']          = [  x for i,x in enumerate(data[mh]['exp'])          if hasrate[i%nproc] ]
        # step 2: strip nuisances with no non-zero value
        data[mh]['nuisname'] = [ data[mh]['nuisname'][i] for i,n in enumerate(data[mh]['nuis']) if sum(n) > 0 ]
        data[mh]['nuis']     = [ n                       for i,n in enumerate(data[mh]['nuis']) if sum(n) > 0 ]
        
print "Generating datacards: " 
for mh,D in  data.items():
        # prepare variables
        # open file
        filename = "%s-mH%s.txt" % (options.label,mh)
        fout = open(filename, "w")
        if fout == None: raise RuntimeError, "Cannot open %s for writing" % filename
        print " - "+filename
        nproc = len(data[mh]['exp'])/data[mh]['nch']
        # write datacard
        fout.write( "Limit (%s), mH = %s GeV\n" % (options.label,mh) )
        fout.write( "date 2010.11.30\n" )
        fout.write( "\n" )
        fout.write( "imax %d number of channels\n"            % D['nch'] )
        fout.write( "jmax %d number of background\n"          % (nproc-1) )
        fout.write( "kmax %d number of nuisance parameters\n" % len(D['nuis']) )
        fout.write( "---------------------------------------------------------------------------------------------------------------\n" );
        fout.write( "Observation " + (" ".join( "%7.3f" % X for X in D['obs'])) + "\n" )
        fout.write( "---------------------------------------------------------------------------------------------------------------\n" );
        pad = " "*27 if options.labelSyst else "";
        fout.write( "bin      " + pad + ("  ".join("%6d" % (X/nproc+1) for X in range(nproc*options.nch)) ) + "\n" )
        fout.write( "process  " + pad + ("  ".join("%6.6s" % X for X in D['processnames'])) + "\n" )
        fout.write( "process  " + pad + ("  ".join("%6d" % (i%nproc) for i in range(nproc*options.nch))) + "\n")
        fout.write( "rate     " + pad + ("  ".join("%6.2f" % f for f in D['exp'])) + "\n")
        fout.write( "---------------------------------------------------------------------------------------------------------------\n" );
        used_syst_labels = {}
        for (inuis,N) in enumerate(D['nuis']):
            name = "%-3d"  % (inuis+1); id = name
            isStat = (D['nuisname'][inuis] == "Statistics in MC or control sample")
            if options.labelSyst: 
                id = iddify(D['nuisname'][inuis])
                if isStat: id = "Stat%d" % inuis 
                if id in used_syst_labels: id += "_%d" % inuis
                used_syst_labels[id] = True
                name = "%-30s" % id
            if options.gamma and isStat:
                # make some room for label
                name = "%-22s" % id
                # have to guess N
                sumN = sum([1.0/(X*X) for X in N if X > 0]); ## people filling error matrix uses the naive 1/sqrt(N) errors
                nN   = sum([1         for X in N if X > 0]);
                realN = int(floor(sumN/nN+1.5)) if nN else 1
                alphas = [ (D['exp'][i]/(realN) if X else 0) for i,X in enumerate(N) ]
                fout.write( name+"  gmN %7d " % realN  + (" ".join([("%7.5f" % X if X else "   0   ") for X in alphas])) + "     " + D['nuisname'][inuis] +  "\n")
            else:
                if id in options.gmM:
                    fout.write( name+"  gmM "  + ("  ".join(["%6.3f" % (0.0+X) for X in N])) + "     " + D['nuisname'][inuis] +  "\n")
                else:
                    fout.write( name+"  lnN "  + ("  ".join(["%6.3f" % (1.0+X) for X in N])) + "     " + D['nuisname'][inuis] +  "\n")

import re
from sys import argv
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-s", "--stat",     dest="stat",     default=False, action="store_true")
parser.add_option("-S", "--signal",   dest="signal",   default=False, action="store_true")
parser.add_option("-a", "--asimov",   dest="asimov",   default=False, action="store_true")
parser.add_option("-o", "--optimize", dest="optimize", default=False, action="store_true")
parser.add_option("-l", "--label",    dest="label",    type="string", default="hww")
parser.add_option("-4", "--4th-gen",  dest="sm4",      default=False, action="store_true")
(options, args) = parser.parse_args()

if len(args) < 1: raise RuntimeError, "Usage: errorMatrix2Lands.py [options] errorMatrix.txt "

file = open(args[0], "r")

data = {}

header = file.readline() # skip header
processnames = header.split()[2:]
# read yields
for l in file:
    l = l.replace("---","0")
    m = re.match(r"(\d+) Yield\s+((\d+\.?\d*(E[+\-]\d+)?\s+)+)", l)
    if not m: break
    mh = m.group(1)
    yields = [float(x) for x in m.group(2).split()];
    if len(yields) != len(processnames)+1: raise RuntimeError, "Length of yields does not match with process names"
    data[mh] = { 'obs':yields[0], 'exp':yields[1:], 'processnames':processnames[:], 'nuis':[] }

# read nuisances
if not options.stat:
    for l in file:
        l = l.replace("---","0")
        m = re.match(r"(.*?)\s+0\s+((\d+\.?\d*(E[+\-]\d+)?\s+)+)", l)
        if m == None: raise ValueError, "Missing line "+l
        sysname = m.group(1)
        syseff  = [float(x) for x in m.group(2).split()]
        # decide which selections are affected
        mhs = data.keys()
        if re.match(r"\d{3}\s+.*?", sysname) and data.has_key(sysname[0:3]):
            mhs = [ sysname[0:3] ]
        for mh in mhs:
            if len(data[mh]['exp']) != len(syseff): raise RuntimeError, "Sysline %s: len(syseff) = %d, len(exp) = %d\n" % (l,len(syseff),len(data[mh]['exp']))
            # special case: the stats line have to be expanded in N independent systematics
            if sysname != mh+" Stats":
                data[mh]['nuis'].append(syseff)
            else:   
                nproc = len(syseff)-(1 if options.sm4 else 0)
                data[mh]['nuis'].append(syseff[0:(2 if options.sm4 else 1)] + [0] * (nproc-1))
                for i in range(1,nproc):
                    data[mh]['nuis'].append([(x if j == i+(1 if options.sm4 else 0) else 0) for j,x in enumerate(syseff)])

if options.optimize:
    for mh in data.keys():
        # step 1: strip processes with no rate
        data[mh]['nuis']         = [ [n[i] for i,k in enumerate(data[mh]['exp']) if k > 0] for n in data[mh]['nuis'] ]
        data[mh]['processnames'] = [ data[mh]['processnames'][i] for i,k in enumerate(data[mh]['exp']) if k > 0 ]
        data[mh]['exp']          = [ k for k in data[mh]['exp'] if k > 0 ]
        # step 2: strip nuisances with no non-zero value
        data[mh]['nuis'] = [ n for n in data[mh]['nuis'] if sum(n) > 0 ]
        
if options.asimov:
    for mh in data.keys():
        data[mh]['obs'] = sum(data[mh]['exp'][(2 if options.sm4 else 1):])
        if options.signal: data[mh]['obs'] += data[mh]['exp'][0]

print "Generating datacards: " 
models = [ 'SM', '4G' ] if options.sm4 else [ 'SM' ]
for (isig,name) in enumerate(models):
    for mh,D in  data.items():
        # prepare variables
        nproc = len(D['exp'])-(1 if options.sm4 else 0) # there's 1 more column, as it has both SM and 4G
        indices = [isig] + range(len(models),nproc+(1 if options.sm4 else 0))
        # open file
        filename = "%s-%s-mH%s.txt" % (options.label,name,mh)
        fout = open(filename, "w")
        if fout == None: raise RuntimeError, "Cannot open %s for writing" % filename
        print " - "+filename
        # write datacard
        fout.write( "%s limit (%s), mH = %s GeV\n" % (name,options.label,mh) )
        fout.write( "date 2010.11.30\n" )
        fout.write( "\n" )
        fout.write( "imax %d number of channels\n"           % 1 )
        fout.write( "jmax %d number of background\n"          % (nproc-1) )
        fout.write( "kmax %d number of nuisance parameters\n" % len(D['nuis']) )
        fout.write( "Observation %f\n" % D['obs'] )
        fout.write( "bin " + ("1 "*nproc) + "\n" )
        fout.write( "process " + (" ".join([D['processnames'][i] for i in indices])) + "\n" )
        fout.write( "process " + (" ".join([str(i) for i in range(nproc)])) + "\n")
        fout.write( "rate " + str(D['exp'][isig])+ " " + (" ".join([str(f) for f in D['exp'][(2 if options.sm4 else 1):]])) + "\n")
        for (inuis,N) in enumerate(D['nuis']):
            fout.write( str(inuis+1) + " lnN " + (" ".join([str(1.0+N[i]) for i in indices])) + "\n")

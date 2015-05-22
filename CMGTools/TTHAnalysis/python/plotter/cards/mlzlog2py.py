import re, sys
pat = re.compile(r"Best fit r:\s+(\S+)\s+(\S+)/(\S+)\s+\(68% CL\)")
for fname in sys.argv[1:]:
    ret = [-1] * 3
    for line in open(fname,'r'):
        m = re.match(pat, line.strip())
        if m:
            ret = [float(m.group(i)) for i in 1,2,3]
            break
    fname = fname.replace("BCat","")
    fname = fname.replace("MuSip4","")
    fname = fname.replace("SUS13","")
    cname = "comb"
    for X in "2lss 3l 4l".split():
        if X+"_" in fname: cname = X.replace("2lss","2l")
    for X in "ee em mumu".split():
        if "2lss_"+X+"_" in fname: cname = X
    print "      %-7s : %s,"  % ("'%s'" % cname, ret)

    

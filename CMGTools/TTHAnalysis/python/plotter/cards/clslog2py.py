import re, sys
res = [ re.compile("Observed Limit: r < (\\S+)"),
        re.compile("Expected  2.5%: r < (\\S+)"),
        re.compile("Expected 16.0%: r < (\\S+)"),
        re.compile("Expected 50.0%: r < (\\S+)"),
        re.compile("Expected 84.0%: r < (\\S+)"),
        re.compile("Expected 97.5%: r < (\\S+)")]
for fname in sys.argv[1:]:
    ret = [-1] * 6
    for line in open(fname,'r'):
        for i,r in enumerate(res):
           m = re.match(r,line.strip()) 
           if m: ret[i] = float(m.group(1))
    fname = fname.replace("BCat","")
    cname = "comb"
    for X in "2lss 3l 4l".split():
        if X+"_" in fname: cname = X.replace("2lss","2l")
    for X in "ee em mumu".split():
        if "2lss_"+X+"_" in fname: cname = X
    print "      %-7s : %s,"  % ("'%s'" % cname, ret)

    

import json

def isort(map):
    keys = map.keys()
    keys.sort()
    return [ (k, map[k]) for k in keys]
def textReport(report):
    tlength = max([len(t) for t in report.keys()])
    slength = max([max([len(r) for r in v['results']]) for v in report.values() if v.has_key('results')])
    tfmt = "%-"+str(tlength)+"s";
    sfmt = "%-"+str(slength)+"s";
    print (tfmt+"  "+sfmt+"  results") % ("test", "subtest")
    print (tfmt+"  "+sfmt+"  %s") % ("-"*tlength, "-"*slength, "-"*40)
    for tn,tv in isort(report):
        if not tv.has_key('results') or len(tv['results']) > 1:
            print (tfmt+"  %s: %s") % (tn, tv['status'].upper(), tv['comment'])
        if not tv.has_key('results'): continue
        one = (len(tv['results']) == 1)
        for rn, rv in isort(tv['results']):
            msg = (tv['status'] if one else rv['status'])+": "
            if rv['status'] == 'done':
                msg += "%6.3f +/- %5.3f  (%5.1f min)" % (rv['limit'], rv['limitErr'], rv['t_real'])
            comm = (tv['comment'] if one else rv['comment'])
            if comm != "": msg += " ("+comm+")"
            print (tfmt+"  "+sfmt+"  %s") % (tn if one else " ^", rn, msg)


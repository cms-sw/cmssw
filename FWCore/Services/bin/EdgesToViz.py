#!/usr/bin/env python

import sys

def readtable(flook):
    tab = {}
    for x in flook.xreadlines():
        s = x.split()
        tab[s[0]]=s
    return tab
        

def runme(infile,outfile,lookupfile,use_name):
    fin = open(infile,'r')
    flook = open(lookupfile,'r')
    fout = open(outfile,'w')

    table = readtable(flook)
    
    fout.write('digraph prof {')

    uni = {}

    for line in fin.xreadlines():
        count,from_node,to_node = line.split()
        uni[from_node] = 1
        uni[to_node] = 1
        print >>fout, '%s -> %s [label="%s"];' % (from_node,to_node,count)

    # print "blob",uni.keys

    for n in uni.keys():
        e = table[n]
        # print e
        node_label = e[0]
        if use_name: node_label = e[2]
        print >>fout,'%s [label="%s (%3.3f)\\nL%s:P%s:C%s"];' % (e[0],node_label,float(e[6]),e[3],e[5],e[4])

    fout.write('}')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "usage: ", sys.argv[0], " edge_input_file digraph_output_file func_names_lookup_file"
        sys.exit(1)
        
    infile = sys.argv[1]
    outfile = sys.argv[2]
    lookupfile = sys.argv[3]
    use_name = 0
    if len(sys.argv)>4: use_name=1
    runme(infile, outfile, lookupfile,use_name)

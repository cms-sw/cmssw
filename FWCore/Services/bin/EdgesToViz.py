#!/usr/bin/env python

import sys

def readtable(flook):
    tab = {}
    for line in flook.xreadlines():
        s = line.split()
        tab[s[0]]=s
    return tab
        

def runme(infile,outfile,lookupfile,use_name):
    fin   = open(infile,'r')
    flook = open(lookupfile,'r')
    fout  = open(outfile,'w')

    table = readtable(flook)
    
    fout.write('digraph prof {')

    uni = {}

    for line in fin.xreadlines():
        count,from_node,to_node = line.split()
        uni[from_node] = 1
        uni[to_node] = 1
        print >>fout, '%s -> %s [label="%s"];' % (from_node,to_node,count)

    # print "blob",uni.keys

    for function_id in uni.keys():
        function_data = table[function_id]
        # print e
        node_label = function_data[0]
        if use_name: node_label = function_data[7].strip('"')
        leaf_fraction      = float(function_data[5])
        recursive_fraction = float(function_data[6])
        print >>fout,'%s [label="ID: %s\\nL: %5.1f%%\\nB: %5.1f%%"];' % (node_label,node_label,leaf_fraction*100, recursive_fraction*100)

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

#!/usr/bin/env python

import sys
import os
import os.path

class Col:
    def __init__(me):
        me.h=0.
        me.s=.5
        me.b=.5

    def next(me):
        rc = "\"%f,%f,%f\""%(me.h,me.s,me.b)
        me.h += .1
        if me.h > 1.:
            me.h=0.
            me.s += .1
        if me.s> 1.:
            me.h=0.
            me.s = .5
            me.b += .1
        return rc

def readtable(flook):
    # field 7 is the library name
    tab = {}
    next = Col()
    cols = {}
    for line in flook.xreadlines():
        s = line.split()
        if not s[7] in cols:
            cols[s[7]] = next.next()
        s.append(cols[s[7]])
        tab[s[0]]=s
    return tab,cols
        

def runme(infile,outfile,lookupfile,use_name):
    fin   = open(infile,'r')
    flook = open(lookupfile,'r')
    fout  = open(outfile,'w')

    table,libcols = readtable(flook)
    
    fout.write('digraph prof {')

    uni = {}
#    d=1
#    for i in libcols.items():
#        print >>fout,'lib%d [label="%s",style=filled,color=%s,fontsize=18];' % (d,os.path.basename(i[0].strip('"')),i[1])
#        d += 1
        

    for line in fin.xreadlines():
        count,from_node,to_node = line.split()
        uni[from_node] = 1
        uni[to_node] = 1
        row_to = table[to_node]
        row_from = table[from_node]
        
        if row_from[-1] == row_to[-1]:
            color="\"#000000\""
        else:
            row=table[to_node]
            color=row[-1]
            
        print >>fout, '%s -> %s [label="%s",fontsize=18,color=%s];' % (from_node,to_node,count,color)

    # print "blob",uni.keys

    for function_id in uni.keys():
        function_data = table[function_id]
        # print e
        node_label = function_data[0]
        if use_name: node_label = function_data[-2].strip('"')
        leaf_fraction      = float(function_data[5])
        recursive_fraction = float(function_data[6])
        if recursive_fraction > .03 and recursive_fraction <.20: shape="box"
        else: shape="circle"
        print >>fout,'%s [label="ID: %s\\nL: %5.1f%%\\nB: %5.1f%%",style=filled,color=%s,shape=%s,fontsize=18];' % (node_label,node_label,leaf_fraction*100, recursive_fraction*100,function_data[-1],shape)

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

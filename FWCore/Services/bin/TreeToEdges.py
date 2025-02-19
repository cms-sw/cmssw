#!/usr/bin/env python

# go to the tree file and pick out all the paths that have hits greater
# than the cutoff and convert the entries to edge definitions.

import sys

class Int:
    def __init__(self,num):
        self.value = num

    def inc(self,num):
        self.value+=num

    def __repr__(self):
        return str(self.value)

def runme(infile,outfile,cutoff):
    fin = open(infile,'r')
    fout = open(outfile,'w')
    tree = {}
    count = 0
    
    for line in fin.xreadlines():

        a = line.split()
        id = int(a.pop(0))
        tot = int(a.pop(0))
        if tot < cutoff: break
        head = int(a.pop(0))
        
        for node in a:
            val = int(node)
            key = (head,val)
		
            n = tree.get(key)
            if n == None:
                tree[key] = Int(tot)
            else:
                n.inc(tot)
            head = val
            
        count += 1

    for node in tree.items():
        # print node
        print >>fout, node[1], ' ', node[0][0], ' ', node[0][1]
            
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print "usage: ", sys.argv[0], " in_tree_file out_edge_file cutoff"
        sys.exit(1)
        
    infile = sys.argv[1]
    outfile = sys.argv[2]
    cutoff = int(sys.argv[3])
    print "cutoff=",cutoff
    # if cutoff == 0: cutoff = 1000000000
    runme(infile, outfile, cutoff)

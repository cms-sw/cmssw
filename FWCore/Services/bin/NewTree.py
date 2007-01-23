#!/usr/bin/env python

# go to the tree file and pick out all the paths that have hits greater
# than the cutoff and convert the entries to edge definitions.

import sys
import re
import Parameters
import operator
import string

def runme2(fid,infile,outfile):
    fin = open(infile,'r')
    fout = open(outfile,'w')
    sub = ' ' + fid + '[ $]'
    r = re.compile(sub)
    
    for line in fin.xreadlines():
        i = line.index(' ')+1
        s = r.search(line[i:])
        if s != None:
            print >>fout,line[0:i+s.end()-1]

def runme(fid,infile,outfile):
    fin = open(infile,'r')
    fout = open(outfile,'w')
    tot_up = Parameters.parameters['view']['levels_up']
    tot_down = Parameters.parameters['view']['levels_down']
    print "up=",tot_up," down=",tot_down
    # trigger too difficult for now
    trigger = Parameters.parameters['find']['immediate_children_threshold']
    
    for line in fin.xreadlines():
        a=line.split()
        b=a[2:]
        # find fid in a
        try:
            i=operator.indexOf(b,fid)
            # write out stuff according to tot_up and tot_down
            if i < tot_up: c = i
            else: c = tot_up
            print >>fout,"%s %s %s"%(a[0],a[1],string.join(b[i-c:i+1+tot_down]))
        except ValueError:
            pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print "usage: ", sys.argv[0], " function_id input_file_prefix"
        sys.exit(1)
        
    fid = sys.argv[1]
    infile = sys.argv[2]
    outfile = fid + Parameters.parameters['view']['out_file_suffix']
    runme(fid, infile, outfile)


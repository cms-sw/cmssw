#!/usr/bin/env python

# go to the vertex file and pick a specific function or group of functions
# with hits less then cutoff.  then go to the path file and get all the
# paths that contain the functions.  produce a new path file from this.

import sys

class Int:
    def __init__(self):
        self.value = 1
    def inc(self):
        self.value+=1
    def __repr__(self):
        return str(self.value)

class NameLine:
    def __init__(self,line):
        self.attr = line.split()
        self.seen = int(self.attr[4])
        self.hits = int(self.attr[3])
        self.id = int(self.attr[0])
    def hits(self): return self.hits
    def seen(self): return self.seen
    def id(self):   return self.id
    def name(self): return self.attr[2]

class PathLine:
    def __init__(self,line):
        self.attr = line.split()
        self.hits = int(self.attr[1])
        self.id = int(self.attr[0])
    def hits(self): return self.hits
    def seen(self): return self.hits
    def id(self):   return self.id

class MatchId:
    def __init__(self,id):
        self.id = id
    def match(self,nline):
        return self.id == nline.id()

class MatchLessSeen:
    def __init__(self,count):
        self.count = count
    def match(self,nline):
        return self.count < nline.attr[4]

class MatchLessHit:
    def __init__(self,count):
        self.count = count
    def match(self,nline):
        return self.count < nline.attr[3]

class MatchIdSet:
    def __init__(self,idset):
        self.idset = idset
    def match(self,nline):
        return self.idset.get(nline.attr[0])!=None

class Match

class Parse:
    def __init__(self, pre_in, pre_out):
        self.in_names = pre_in + "names"
        self.in_paths = pre_in + "paths"
        self.out_names = pre_in + "names"
        self.out_paths = pre_in + "paths"
        self.out_edges = pre_out + "edges"
        self.out_totals = pre_out + "totals"

    def cut(matcher):
        fin_names = open(self.in_names,'r')
        fout_names = open(self.out_names,'w')
        names = []
        for line in fin.names.xreadlines():
            n = NameLine(line)
            b = matcher(n)
            if b<0: break
            if b:
                names[n.id()]
                fout.write(line)
        return names

    def selectOneName(value):
        fin_names = open(self.in_names,'r')
        fout_names = open(self.out_names,'w')
        names = {}
        for line in fin.names.xreadlines():
            a=line.split()
            if int(a[3])==value: break
        names[int(a[0])]=1
        fout.write(line)
        return names

    def trimNames(cutoff):
        fin_names = open(self.in_names,'r')
        fout_names = open(self.out_names,'w')
        names = []
        for line in fin.names.xreadlines():
            a=line.split()
            if int(a[3])<cuttoff: break
            names[int(a[0])]
            fout.write(line)
        return names

    def selectManyNames(ids):
        fin_names = open(self.in_names,'r')
        fout_names = open(self.out_names,'w')
        names = {}
        for line in fin.names.xreadlines():
            a=line.split()
            if ids.get(int(a[0]))!=None:
                fout.write(line)
        
    def trimPaths(cutoff):
        fin_paths = open(self.in_paths,'r')
        fout_paths = open(self.out_paths,'w')
        self.tot_paths

    def pathContaining(id):
        pass

def runme(in_nodefile, in_treefile, out_treefile, cutoff, cuttype)
    fin_nodes = open(in_nodefile,'r')
    fin_paths = open(in_treefile,'r')
    fout = open(out_treefile,'w')
    tree = {}
    
    for line in fin.xreadlines():
        a = line.split()
        id = int(a.pop(0))
        tot = int(a.pop(0))
        if tot < cutoff:
            print tot
            continue
        head = int(a.pop(0))
        
        for node in a:
            val = int(node)
            key = (head,val)
            n = tree.get(key)
            if n == None:
                tree[key] = Int()
            else:
                n.inc()
            head = val

    for node in tree.items():
        # print node
        print >>fout, node[1], ' ', node[0][0], ' ', node[0][1]
            
if __name__ == "__main__":
    if len(sys.argv) < 5:
        print "usage: ", sys.argv[0], " in_prefix out_prefix cutoff type"
        print " type = 0 means accept one exact match for cutoff value"
        print " type = 1 means accept anything >= cutoff value"
        sys.exit(1)
        
    in_nodefile = sys.argv[1]
    in_treefile = sys.argv[2]
    out_treefile = sys.argv[3]
    cutoff = int(sys.argv[4])
    cuttype = int(sys.argv[5])
    
    runme(in_nodefile, in_treefile, out_treefile, cutoff, cuttype)

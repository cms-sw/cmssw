#!/usr/bin/env python

'''
The output from the profiler has names that end in 'names' and 'paths'.
This script expects you run c++filt on the 'names' file and save
it will the same prefix, but with 'nice_names' on the end.
This will be improved on the release of simple profiler.

The output file from this script has the same prefix as the input
files, but ends in 'calltree'.  This file can be fed directly into
kcachegrind.
'''

import sys
import csv

if len(sys.argv) < 3:
    print "usage: ",sys.argv[0]," prefix_of_run_data tick_cutoff n|o"
    sys.exit(1)

prefix = sys.argv[1]
cutoff = int(sys.argv[2])
newold_flag = 'o'

if len(sys.argv) > 3:
    newold_flag = sys.argv[3]

outfile = open(prefix+"calltree","w")
pathfile = open(prefix+"paths","r")

names = {}
edges = {}

def tostring(l):
    x=""
    for i in l: x += "%d "%i
    return x

class Node:
    def __init__(me,fid,name):
        me.fid = int(fid)
        me.name = name
        me.ticks_as_parent = 0
        me.ticks_as_child = 0
        me.ticks_as_parent_recur = 0
        me.calls = 0
        me.children = {} # set() - want to use set, but not until 2.4

    def parentTicks(me,count):
        me.ticks_as_parent += int(count)
    
    def childTicks(me,count):
        c = int(count)
        me.ticks_as_parent += c
        me.ticks_as_child += c
        
    def recurTicks(me,count):
        me.ticks_as_parent_recur += int(count)

    def addChild(me,child_id):
        me.children[child_id]=0  # .add(child_id) -cannot use set functions yet

    def __str__(me):
        return "(%d,%s,%d,%d,%s)"%(me.fid,me.name,me.ticks_as_parent,me.ticks_as_child,tostring(me.children.keys()))

class EdgeCount:
    def __init__(me):
        me.count = 0

    def addTicks(me,count):
        me.count += count

    def __str__(me):
        return "%d"%me.count

if newold_flag == 'o':
    namefile = open(prefix+"nice_names","r")
    for entry in namefile:
        front=entry.split(' ',2)
        back=front[2].rsplit(' ',16)
        name=back[0]
        name=name.replace(' ','-')
        #print name
        #names[front[0]]=(front[0],name,back[1],back[2],back[3],back[4],back[5])
        names[int(front[0])] = Node(front[0],name)
else:
    namefile = csv.reader(open(prefix+"nice_names","rb"),delimiter='\t')
    for row in namefile:
        names[int(row[0])] = Node(row[0],row[-1])
    

print >>outfile,"events: ticks"

for entry in names.values():
    print >>outfile,"fn=(%s) %s"%(entry.fid,entry.name)
    # print entry[1]

print >>outfile

# all the 0 values below are line numbers (none in my case)
#------------------------------
# fn=func    <- parent function
# 0 100      <- cost, accumulated into "incl" and "self" for func
# cfn=func2  <- child function
# calls=2 0  <- times called from func
#               (accumulated into "called" for func2)
# 0 350      <- cost involved in call
#               (accumulated into "incl" for func2)

# "incl" for a function appears in the graphed box, it is the sum over
#        all incoming wires
# A wire contains cost value from cfn entry

for pe in pathfile:
    all = pe.split()
    l = len(all)
    #entry = names[all[-1:]]
    #print all

    ticks = int(all[1])
    if ticks < cutoff: continue
    last = int(all[l-1])
    names[last].childTicks(ticks)

    for i in range(2,l-1):
        edge = (int(all[i]), int(all[i+1]))
        node = names[edge[0]]
        node.recurTicks(ticks)
        if edge[0]!=edge[1]:
            node.parentTicks(ticks)
            node.addChild(edge[1])
            edges.setdefault(edge,EdgeCount()).addTicks(ticks)

#for x in names.values():
#    print x

for node in names.values():
    cost=0
    #if len(node.children) == 0:
    cost = node.ticks_as_child
    print >>outfile,"fn=(%d)"%node.fid
    print >>outfile,"0 %d"%cost
    #print >>outfile,"\n\n"
    
    for child in node.children.keys():
        count = edges[(node.fid,child)].count
        print >>outfile,"cfn=(%d)"%child
        print >>outfile,"calls=%d 0"%count
        print >>outfile,"0 %d"%count
        
    print >>outfile,"\n\n"
            

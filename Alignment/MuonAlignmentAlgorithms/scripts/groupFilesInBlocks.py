#! /usr/bin/env python

from __future__ import print_function
import re,os,sys,shutil,math
import optparse

copyargs = sys.argv[:]
for i in range(len(copyargs)):
    if copyargs[i] == "":
        copyargs[i] = "\"\""
    if copyargs[i].find(" ") != -1:
        copyargs[i] = "\"%s\"" % copyargs[i]
commandline = " ".join(copyargs)

prog = sys.argv[0]

usage='./%(prog)s NBLOCKS INFILE OUTFILE [options]\n'+\
  'takes list of files produced by findQualityFiles.py as INFILE,\n'+\
  'groups them into maximum NBLOCKS blocks with approximately similar #events.'


######################################################
# To parse commandline args


parser=optparse.OptionParser(usage)

parser.add_option("-v", "--verbose",
  help="debug verbosity level",
  type="int",
  default=0,
  dest="debug")

options,args=parser.parse_args()

if len(sys.argv) < 4:
    raise SystemError("Too few arguments.\n\n"+parser.format_help())

NBLOCKS = int(sys.argv[1])
INFILE = sys.argv[2]
OUTFILE = sys.argv[3]



def makeJobBlock(mylist, evtn):
    n = mylist[0][0]
    block = [mylist[0]]
    choosen = [0]
    while n<evtn:
    #print "n,evtn=",n,evtn
    # find the biggest unused #evt that would give n<evtn
        for i in range(len(mylist)):
            # get last not choosen i
            last_i=len(mylist)-1
            while last_i in choosen: last_i += -1
            if i==last_i:
        #print i,"last element reached"
                n += mylist[i][0]
                #print "   new last append: ",i, mylist[i][0], n
                block.append(mylist[i])
                choosen.append(i)
                break
            if i in choosen:
                #print i,"  in choosen, continue..."
                continue
            if n+mylist[i][0]<evtn:
                n += mylist[i][0]
                #print "   new append: ",i, mylist[i][0], n
                block.append(mylist[i])
                choosen.append(i)
                break
        if len(choosen)==len(mylist):
            #print " got everything"
            break
    # pick up unused elements
    newlist = []
    for i in range(len(mylist)):
        if not i in choosen:
            newlist.append(mylist[i])
    print("done makeJobBlock n =",n," len =",len(block))
    return block, newlist, n



comment1RE = re.compile (r'^#.+$')
fileLineRE = re.compile (r'^.*\'(.*)\'.+# (\d*).*$')
#fileLineRE = re.compile (r'^.*\'(.*)\'.+# (\d*),(\d*).*$')

if not os.access(INFILE, os.F_OK): 
    print("Cannot find input file ", INFILE)
    sys.exit()

fin = open(INFILE, "r")
lines = fin.readlines()
fin.close()


eventsFiles = []
ntotal = 0
commentLines=[]

for line in lines:
    #line = comment1RE.sub ('', line)
    #line = line.strip()
    #if not line: continue
    match = comment1RE.match(line)
    if match:
        commentLines.append(line)

    match = fileLineRE.match(line)
    if match:
        #print int(match.group(3)), str(match.group(1))
        #eventsFiles.append((int(match.group(3)), str(match.group(1)), str(match.group(2))))
        eventsFiles.append((int(match.group(2)), str(match.group(1))))
        ntotal += int(match.group(2))
    #else: print line,

if len(eventsFiles)==0:
    print("no file description strings found")
    sys.exit()

#print "len=", len(eventsFiles), ntotal
#tmp = set(eventsFiles)
#eventsFiles = list(tmp)
#ntotal = 0
#for ff in eventsFiles:  ntotal += ff[0]
#print "len=", len(eventsFiles), ntotal
#sys.exit()

eventsFiles.sort(reverse=True)
#print eventsFiles

evtPerJob = int(math.ceil(float(ntotal)/NBLOCKS))
print("Total = ",ntotal, "  per block =", evtPerJob,"(would give total of ", evtPerJob*NBLOCKS, ")", "  list length =",len(eventsFiles))
if eventsFiles[0][0] > evtPerJob:
    print("the biggest #evt is larger then #evt/block:",eventsFiles[0][0],">",evtPerJob)
    print("consider lowering NBLOCKS")


jobsBlocks=[]
temp = eventsFiles

tt = 0
for j in range(NBLOCKS):
    print(j)
    if len(temp)==0:
        print("done!")
        break
    block, temp, nn = makeJobBlock(temp,evtPerJob)
    tt+=nn
    if len(block)>0:
        jobsBlocks.append((block,nn))
        print(block)
    else:
        print("empty block!")

print(tt)
print(commandline)


fout = open(OUTFILE, mode="w")

fout.write("### job-split file list produced by:\n")
fout.write("### "+commandline+"\n")
fout.write("### Total #evt= "+str(ntotal)+"  #files ="+str(len(eventsFiles))+"  per job #evt="
           +str(evtPerJob)+" (would give total of"+str(evtPerJob*NBLOCKS)+")\n###\n")
fout.write("### previously produced by:\n")
fout.write("".join(commentLines))
fout.write("\nfileNamesBlocks = [\n")

commax = ","
for b in range(len(jobsBlocks)):
    fout.write('  [ # job '+str(b)+' with nevt='+str(jobsBlocks[b][1])+'\n')
    comma = ","
    for i in range(len(jobsBlocks[b][0])):
        if i==len(jobsBlocks[b][0])-1:
            comma=""
        #fout.write("    '"+ jobsBlocks[b][0][i][1] +"'"+comma+" # "+ str(jobsBlocks[b][0][i][2]) +','+ str(jobsBlocks[b][0][i][0]) + "\n")
        fout.write("    '"+ jobsBlocks[b][0][i][1] +"'"+comma+" # "+ str(jobsBlocks[b][0][i][0]) + "\n")
    if b==len(jobsBlocks)-1:
        commax=""
    fout.write('  ]'+commax+'\n')
fout.write(']\n')
fout.close()

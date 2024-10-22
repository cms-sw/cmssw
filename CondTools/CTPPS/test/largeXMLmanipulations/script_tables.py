import subprocess
from __future__ import print_function

infile   = open("alignment_export_2018_12_07.1.xml","rt")
xmllines = infile.readlines()
infile.close()
#tree = ET.parse(infilename)
#root = tree.getroot()
outfile =open("test.xml","wt")
iov = 0

firstline = xmllines.pop(0)
secondline = xmllines.pop(0)
lastline = xmllines.pop(-1)

for line in xmllines:
    if "iov" not in line:
        outfile.write(line)
    else:
        if "</iov>" in line:
            outfile.write(line)
            outfile.write(lastline)
            outfile.close()
            outfilename = "real_alignment_iov"+str(iov)+".xml"
            output = subprocess.run("mv test.xml "+outfilename, shell=True, check=True)
            outfile=open("test.xml","wt")
        else:
            outfile.write(firstline)
            outfile.write(secondline)
            outfile.write(line)
            iovfirstfield = line.split()[1].split("=\"")[1].split(":")[0]
            iov = int(iovfirstfield)
            print (iov)

if not  outfile.closed:
    outfile.close()

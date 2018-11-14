import fileinput
import re

def index(line,substr):
    result = line.index(substr)
    return result

def errorPrint(line,indices):
    print(line)
    ll = len(line)
    errstr="_"*ll
    for i in indices:
        errstr = errstr[:i] + '^' + errstr[i+1:]
    print errstr
    
def findValuesWithUnits(line,ln):
    numList = re.findall(r"\d*?[\s,.]?\d*\*\w*", line)
    errindices = []
    for match in re.finditer(r"\d*?[\s,.]?\d*\*\w*", line):
        errindices.append(match.start())
    l = len(numList)
    if l > 0:
        print 'Line #',ln,'Units defined: '
        errorPrint(line,errindices)
    return l

def findIndices(line,strList):
    indices=[]
    for x in strList:
        idx = index(line,x)
        indices.append(idx)
    print(indices)
    return indices

def findValuesWithoutUnits(line,ln):
    numList = re.findall(r"\d+?[\s,.]?\d+[\s\"]", line)
    errindices = []
    for match in re.finditer(r"\d+?[\s,.]?\d+[\s\"]", line):
        errindices.append(match.start())
    l = len(numList)
    if l > 0:
        print 'Line #', ln, 'WARNING: Numerical values without units: '
        errorPrint(line,errindices)
    return l

def lineNumber(lookup):
    with open(fileinput.filename()) as myfile:
        for num, line in enumerate(myfile, 1):
            if lookup in line:
                return num
    
def process(line):
    ln = lineNumber(line)
    l = findValuesWithUnits(line,ln)
    k = findValuesWithoutUnits(line,ln)
    if l > 0 or k > 0:
        print ' '

def check(line):
    return 0;

for line in fileinput.input():
    check(line)

with open(fileinput.filename()) as myfile:
    for num, line in enumerate(myfile, 1):
        process(line)



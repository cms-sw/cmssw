import fileinput
import re

def index(line,substr):
    result = line.index(substr)
    return result

def errorPrint(line,indices):
    print(line.replace('\t',' '))
    ll = len(line)
    errstr=" "*ll
    for i in indices:
        errstr = errstr[:i] + '^-----' + errstr[i+5:]
    print(errstr)
    
def findValuesWithUnits(line,ln):
    numList = re.findall(r"\d*?[\s,.]?\d*\*\w*", line)
    errindices = []
    for match in re.finditer(r"\d*?[\s,.]?\d*\*\w*", line):
        errindices.append(match.start())
    l = len(numList)
    if l > 0:
        text = fileinput.filename()+': line# '+str(ln)+' units defined: '
        print(text)
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
        if 'MaterialFraction' in line:
            return l
        if '<?xml' in line:
            return l
        text = fileinput.filename()+': line# '+str(ln)+' warning: numerical value without units: '
        print(text)
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

def check(line):
    return 0;

for line in fileinput.input():
    check(line)

with open(fileinput.filename()) as myfile:
    for num, line in enumerate(myfile, 1):
        process(line)



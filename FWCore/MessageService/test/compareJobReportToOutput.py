#!/usr/bin/env python3

import xml.etree.ElementTree as ET

quantities = ["Name", "ReducedConfigurationID", "ParameterSetID"]

def parsexml(fname):
    root = ET.parse(fname)
    process = root.find("Process")
    return {q: process.find(q).text for q in quantities}

def parselog(fname):
    ret = {}
    with open(fname) as f:
        for line in f:
            s = line.rstrip().split(":")
            if len(s) == 2:
                ret[s[0]] = s[1]
    return ret

def main(jobreport, log):
    xmldata = parsexml(jobreport)
    logdata = parselog(log)

    ret = 0
    for q in quantities:
        if xmldata[q] != logdata[q]:
            print(f"Quantity {q}: job report '{xmldata[q]}' != log '{logdata[q]}'")
            ret = 1
    return ret
    
if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1], sys.argv[2]))

#!/usr/bin/env python

import os
import re

def retrieveFedEntries(fedNumberingHeader):
    reFedEntry = re.compile('\s*(MIN|MAX)(?P<name>.*)FEDID\s*=\s*(?P<id>[0-9]+).*')
    entries = {}

    with open(fedNumberingHeader) as fedNumbering:
        for line in fedNumbering:
            match = reFedEntry.match(line)
            if match and match.group('name'):
                try:
                    entries[match.group('name')].append(int(match.group('id')))
                except KeyError:
                    entries[match.group('name')] = [int(match.group('id'))]
                entries[match.group('name')].sort()
    return sorted(entries.items(), key=lambda e: e[1][0])

def printHtmlTable(fedEntries):
    lastId = -1
    tableRow = "<tr style='color:%s'><td><div align='center'>%s</div></td><td><div align='center'>%s</div></td><td><div align='center'>%s</div></td></tr>"
    print("<table width='75%' border='1' align='center'>")
    print("<tr style='color:#FF0000'><th>Detector</th><th>Min FED id (decimal)</th><th>Max FED id (decimal)</th></tr>")
    for item in fedEntries:
        if lastId+1 < item[1][0]:
            print(tableRow%('#FF0000','Free IDs',lastId+1,item[1][0]-1))
        print(tableRow%('#000000',item[0],item[1][0],item[1][1]))
        lastId = item[1][1]
    print("</table>")

def printTwikiTable(fedEntries):
    lastId = -1
    tableRow = "|%(color)s !%(label)s|%(color)s %(minId)s|%(color)s %(maxId)s|"
    print("|*Detector*|*Min FED id (decimal)*|*Max FED id (decimal)*|")
    for item in fedEntries:
        if lastId+1 < item[1][0]:
            print(tableRow%{'color':'%RED%','label':'Free IDs','minId':lastId+1,'maxId':item[1][0]-1})
        print(tableRow%{'color':'%BLACK%','label':item[0],'minId':item[1][0],'maxId':item[1][1]})
        lastId = item[1][1]

if __name__ == "__main__":
    fedEntries = retrieveFedEntries(os.environ['CMSSW_BASE']+'/src/DataFormats/FEDRawData/interface/FEDNumbering.h')
    printHtmlTable(fedEntries)
#    printTwikiTable(fedEntries)

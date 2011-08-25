#!/usr/bin/python

print "Merging files"

filePart = ""
updatePart = ""

outputFile = open("full_run_updated.js", "w")

updateFile = open("update_runs.js")
for line in updateFile:
    if( line.find("data") != -1 ):
        if( len(line.split("[[")) == 1 ):
            print "No new runs to be added"
            updatePart = "]\n"
        else:
            updatePart = ", ["+line.split("[[")[1]


file = open("full_run.js")
for line in file:
    if( line.find("data") != -1 ):
        filePart = line.split("]]")[0]+"]"
        # print filePart + updatePart
        outputFile.write(filePart + updatePart)
    else:
        # print line
        outputFile.write(line)

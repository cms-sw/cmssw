import time

inputFile = open("full.js")

outputFile = open("oneMonth.js", "w")

firstIOV = 0
for line in inputFile:
    if "data" in line:
        splittedLine = line.split("],")
        for item in splittedLine:
            IOVtime = item.split(", ")[0].replace("[[", "[").split("[")[1]
            # one month in seconds = 31*24*60*60 = 2678400
            if (time.time() - int(IOVtime)/1000) < 2678400:
                firstIOV = IOVtime
                num = len(line.split(firstIOV))
                outputFile.write("    data: [[" + IOVtime + line.split(firstIOV)[num-2] + IOVtime + line.split(firstIOV)[num-1])
                break
    else:
        outputFile.write(line)

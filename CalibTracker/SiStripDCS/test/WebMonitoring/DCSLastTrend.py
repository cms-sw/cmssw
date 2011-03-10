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
                outputFile.write("    data: [[" + IOVtime + line.split(firstIOV)[1] + IOVtime + line.split(firstIOV)[2])
                break
    else:
        outputFile.write(line)

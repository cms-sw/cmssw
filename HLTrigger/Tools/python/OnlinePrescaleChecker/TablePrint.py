import sys
from colors import *
write = sys.stdout.write

def PrettyPrintTable(Headers,Data,ColWidths,WarningCol=[],border='*'):
   PrintHLine(ColWidths,border)
   PrintLine(Headers,ColWidths,False,border)
   PrintHLine(ColWidths,border)
   if WarningCol==[]:
      WarningCol=[False]*len(Data)
   for [line,Warn] in zip(Data,WarningCol):
       PrintLine(line,ColWidths,Warn,border)
   PrintHLine(ColWidths,border)

def PrintHLine(ColWidths,border): ## writes a horizontal line of the right width
    #write = sys.stdout.write
    for entry in ColWidths:
        write(border)
        for i in range(entry):
            write(border)
    write(border)
    write('\n')

def PrintLine(line,ColWidths,Warn,border):
    assert Warn in [True,False]
    try:
       assert len(line)==len(ColWidths)
    except:
       print line
       print ColWidths
       raise
    if Warn:
        write(bcolors.FAIL)
    for [width, entry] in zip(ColWidths,line):
        write(border)
        try:
            entry = str(entry)
        except:
            print "\n\n\n Weird Data .. Bailing out\n\n"
            sys.exit(0)
        for i in range(width):
            if i==0:
                write(' ')
            elif i<len(entry)+1:
                write(entry[i-1])
            else:
                write(' ')
    write(border)
    write('\n')
    write(bcolors.ENDC)

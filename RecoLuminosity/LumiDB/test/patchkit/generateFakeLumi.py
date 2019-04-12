from __future__ import print_function
from builtins import range
import sys
def main(*args):
    filename=args[1]
    startLS=args[2]
    stopLS=args[3]
    lumiVal=args[4]
    f=open(filename,'w')
    for i in range(int(startLS),int(stopLS)+1):
        value=str(i)+' '+str(lumiVal)+' '
        print(value)
        f.write(value)
    f.close()
if __name__=='__main__':
    sys.exit(main(*sys.argv))

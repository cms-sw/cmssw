import logging
import sys

'''
    Usage:
     Physics Debug output. Can write to file and/or to console. 
     is based on Python logging.

     To set it up
       import pdebug as pdebug
       from pdebug import pdebugger

     Use following 3 lines and comment out as needed to obtain desired behaviour
       #pdebugger.setLevel(logging.ERROR)  # turns off all output
       pdebugger.setLevel(logging.INFO) # turns on ouput
       pdebug.set_file("pdebug.log",level=logging.INFO) #optional writes to file
       pdebugger.set_stream(level=logging.ERROR)

    For example
     (1) file and console:
       pdebugger.setLevel(logging.INFO)
       pdebug.set_file("pdebug.log")

     (2) console only:
       pdebugger.setLevel(logging.INFO)

     (3) file only:
       pdebugger.setLevel(logging.INFO)
       pdebug.set_file("pdebug.log")
       pdebug.set_stream(level=logging.ERROR)

     (4) no output
       pdebugger.setLevel(logging.ERROR)
       or else no lines of code also gives same result

    to use in code
       from pdebug import pdebugger
       pdebugger.info("A message")

'''

#Note the first use of this header should come from the top level of the program
#If not the stream output may be missing
pdebugger = logging.getLogger('pdebug')
pdebugger.setLevel(logging.ERROR)
pdebugger.propagate = False

def set_file(filename = "pdebug.log", mode='w', level ="INFO"):
    #todo add checks
    cf = logging.FileHandler(filename, mode)
    cf.setLevel(level)
    pdebugger.addHandler(cf)

def set_stream(out=sys.stdout, level ="INFO"):
    ch = logging.StreamHandler(out)
    ch.setLevel(level)
    mformatter = logging.Formatter('%(message)s')
    ch.setFormatter(mformatter)
    pdebugger.addHandler(ch)

if __name__ == '__main__':

    pdebugger.setLevel(logging.INFO)
    set_stream(sys.stdout)
    set_file("pdebug.log")
    pdebugger.info('blah')

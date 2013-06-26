#!/usr/bin/env python
'''
GetBadEvents.py
v1.0
Jeff Temple
Oct. 19, 2012
'''

import sys,os,string
from optparse import OptionParser

def GetBadCrabEvents(crabdir=None,prefix=None,verbose=False):
    ''' Searches the res/*stdout files in the specified crab output directory "crabdir".  For each file, each line is read, and parsed as run:LS:event.  If "prefix" is set, any lines not starting with "prefix" are skipped.  The list of events "badlist" is returned at the end of the function.'''

    badlist=[]
    if not os.path.isdir(crabdir):
        print "<GetBadEvents> Sorry, directory '%s' does not exist!"%crabdir
        return badlist
    newdir=os.path.join(crabdir,'res')
    if not os.path.isdir(newdir):
        print "<GetBadEvents> Sorry, subdirectory '%s' does not exist!"%newdir
        return badlist
    # Search stdout files
    allfiles=os.listdir(newdir)
    allfiles.sort()
    for f in allfiles:
        if not f.endswith(".stdout"):
            continue
        ###print f
        temp=open(os.path.join(newdir,f),'r').readlines()
        for line in temp:
            if prefix<>None and not line.startswith(prefix):
                continue
            if prefix<>None:
                thisline=string.strip(string.split(line,prefix)[1])
            else:
                thisline=string.strip(line)
            try:
                # Check to make sure that string can be parsed as run:ls:event
                myevt=string.split(thisline,":")
                run=string.atoi(myevt[0])
                ls=string.atoi(myevt[1])
                evt=string.atoi(myevt[2])
                ##if (run<0 or ls<0 or evt<0):
                    ##print "<GetBadEvents> Error!  Run:LS:Event value less than 0 for '%s'!"%thisline
                    ##continue
            except:
                print "<GetBadEvents>  Error!  Cannot understand string '%s'"%thisline
                continue
            if thisline not in badlist:
                badlist.append(thisline)
            else:
                if verbose:
                    print "<GetBadEvents> Warning!  Event %s already in list!"%thisline
        
    return badlist


############################################################

if __name__=="__main__":
    parser=OptionParser()
    parser.add_option("-v","--verbose",
                      dest="verbose",
                      action="store_true",
                      default=False,
                      help="If specified, extra debug information is printed.")
    parser.add_option("-p","--prefix",
                      default=None,
                      dest="prefix",
                      help="If prefix specified, only lines beginning with specified prefix are parsed as Run:LS:Event")
    parser.add_option("-o","--outfile",
                      dest="outfile",
                      default="badevents.txt",
                      help="Specify name of output file.  Default is badevents.txt")
    parser.add_option("-n","--nocrab",
                      dest="nocrab",
                      default=False,
                      action="store_true",
                      help="If specified, combines all command-line input files into a single output, instead of assuming a CRAB output directory and parsing subdirs for *.stdout files.  Default is False (meaning that a CRAB structure is assumed).")
    (opts,args)=parser.parse_args()

    allbadevents=[]
    if (opts.nocrab==False):
        for dir in args:
            badlist=GetBadCrabEvents(crabdir=dir,
                                     prefix=opts.prefix,
                                     verbose=opts.verbose)
            print "Total bad events with prefix '%s' in directory '%s' is %i"%(opts.prefix,dir,len(badlist))
            # We could just all bad events, without first checking to see if already present, which would lead to double-counting, but wouldn't affect the actual filtering
            for b in badlist:
                if b not in allbadevents:
                    allbadevents.append(b)

    else:
        for infile in args:
            if not os.path.isfile(infile):
                print "Sorry, input file '%s' does not exist"%infile
                continue
            print "Reading input file '%s'"%infile
            events=open(infile,'r').readlines()
            for e in events:
                temp=string.strip(e)
                if temp not in allbadevents:
                    allbadevents.append(temp)

    print "Total bad events found = ",len(allbadevents)
    outfile=open(opts.outfile,'w')
    print "Sorting list of %i bad events"%len(allbadevents)
    allbadevents.sort()
    for b in allbadevents:
        outfile.write("%s\n"%b)
    outfile.close()
                             

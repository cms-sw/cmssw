# $Id$
#
# Author: Stefano Argiro'
#
# Script to load ECAL conditions to DB using PopCon
# Intended to be used with the drop-box mechanism, where an XML file
# containing Ecal conditions is sent to DB
#

from   elementtree.ElementTree import parse
import sys,os,getopt


def usage():

   print "Usage: "+sys.argv[0]+" Write me !"

   sys.exit(2)


def main():

   try:
      opts, args = getopt.getopt(sys.argv[1:], "f:d", ["file=","dryrun"])

   except getopt.GetoptError:
      #print help information and exit
      usage()
      sys.exit(2)

   file  = ''
   dryrun= False
   
   for opt, arg in opts:   
     if opt in ("-f", "--file"):
         file = arg
         if (not os.path.exists(file)) :
            print sys.argv[0]+" File not found: "+file
            sys.exit(2)

   if file=='':
       usage()
       exit(2)
   
   tag,since = readTagAndSince(file)
   print tag,since



def readTagAndSince(filename, headertag='EcalCondHeader'):
    '''Read tag and since from EcalCondHeader in XML file '''
    root   = parse(filename).getroot()
    header = root.find(headertag)
    since  = header.findtext('since') 
    tag    = header.findtext('tag')     

    return tag,since

    

if __name__ == "__main__":
    main()

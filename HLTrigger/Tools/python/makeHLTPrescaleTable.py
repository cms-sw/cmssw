#!/usr/bin/env python 
from sys import stderr, exit
import commands, os

from optparse import OptionParser
parser = OptionParser(usage=
"""
usage: %prog [options] csv_output_file

examples:

 %prog out.csv

     produces a table of ALL runs and ALL paths (can take quite some time)

 %prog --path='*ele*' --path='*photon*' out.csv

     select only paths containing 'ele' and 'photon'

""")
parser.add_option("--firstRun",  dest="firstRun",  help="first run", type="int", metavar="RUN", default="1")
parser.add_option("--lastRun",   dest="lastRun",   help="last run",  type="int", metavar="RUN", default="9999999")
parser.add_option("--groupName", dest="groupName", help="select runs of name like NAME", metavar="NAME", default="Collisions%")
parser.add_option("--overwrite", dest="overwrite", help="force overwriting of output CSV file", action="store_true", default=False)

parser.add_option("--path",
                 dest="pathPatterns",
                 default = [],
                 action="append",
                 metavar="PATTERN",
                 help="restrict paths to PATTERN. Note that this can be a single path name or a pattern " +
                      "using wildcards (*,?) similar to those used for selecting multiple files (see " +
                      "the documentation of the fnmatch module for details). Note also that this option " +
                      "can be specified more than once to select multiple paths or patterns. If this option " +
                      "is not specified, all paths are considered. Note that the comparison is done " +
                      "in a case-INsensitive manner. " +
                      "You may have to escape wildcards (with quotes or backslash) in order to avoid "+
                      "expansion by the shell"
                 )

# parser.add_option("--jsonOut",   dest="jsonOut",   help="dump prescales in JSON format on FILE", metavar="FILE")
(options, args) = parser.parse_args()
if len(args) != 1:
   parser.print_help()
   exit(2)

csv_output_file = args[0]

if os.path.exists(csv_output_file) and not options.overwrite:
   print >> stderr,"cowardly refusing to overwrite existing output file '" + csv_output_file + "'. Run this script without argument to see options for overriding this check."
   exit(1)

#----------------------------------------------------------------------
def getPrescaleTableFromProcessObject(process):
   """ returns a dict of hlt key to vector of prescales
   mapping """

   retval = {}
   for entry in process.PrescaleService.prescaleTable:
       retval[entry.pathName.value()] = entry.prescales.value()

   return retval    

#----------------------------------------------------------------------

def getProcessObjectFromConfDB(hlt_key):
   # print >> stderr,"\t%s ..." % hlt_key
   cmd = "edmConfigFromDB --orcoff --configName " + hlt_key
   # print >> stderr, "cmd=",cmd
   res = commands.getoutput(cmd)

   # potentially dangerous: we're running python code here
   # which we get from an external process (confDB).
   # we trust that there are no file deletion commands
   # in the HLT configurations...

   # for some keys, edmConfigFromDB seems to produce error messages.
   # just return None in this case
   try:
       exec(res)
   except:
       return None

   return process

#----------------------------------------------------------------------
from queryRR import queryRR

#----------------------------------------------------------------------
# main
#----------------------------------------------------------------------

# check whether we have a CMSSW environment initalized
if os.system("which edmConfigFromDB") != 0:
   print >> stderr,"could not find the command edmConfigFromDB. Did you initialize your CMSSW runtime environment ?"
   exit(1)

runKeys = queryRR(options.firstRun,options.lastRun,options.groupName)

# maps from HLT key to prescale information. 
# indexed as: prescaleTable[hlt_key][hlt_path_name]
prescaleTable = {}

# maps from 
hlt_path_names_table = {}

# set of all HLT paths seen so far
all_hlt_path_names_seen = set()

runs = sorted(runKeys.keys())

# loop over all hlt keys found

all_hlt_keys_seen = set(runKeys.values())

print >> stderr,"found %d runs and %d HLT menus" % ( len(runKeys), len(all_hlt_keys_seen))

index = 1

for hlt_key in all_hlt_keys_seen:

   print >> stderr,"(%3d/%3d) Querying ConfDB for HLT menu %s" % (index, len(all_hlt_keys_seen) , hlt_key)
   process = getProcessObjectFromConfDB(hlt_key)

   if process == None:
       print >> stderr,"WARNING: unable to retrieve hlt_key '" + hlt_key + "'"
       continue

   prescaleTable[hlt_key] = getPrescaleTableFromProcessObject(process)

   all_path_names = set(process.paths.keys())

   # remember which hlt paths were in this menu
   hlt_path_names_table[hlt_key] = all_path_names

   # add this configuration's HLT paths to the list
   # of overall path names seen
   all_hlt_path_names_seen.update(all_path_names)

   index += 1

# end of loop over all HLT keys

# make sure the list of all HLT path names ever seen is sorted
all_hlt_path_names_seen = sorted(all_hlt_path_names_seen)

#--------------------
# filter paths if required by the user
if len(options.pathPatterns) > 0:
   import fnmatch

   tmp = []

   for path in all_hlt_path_names_seen:

       for pattern in options.pathPatterns:
           if fnmatch.fnmatch(path.lower(), pattern.lower()):

               # accept this path
               tmp.append(path)
               break

   all_hlt_path_names_seen = tmp

# sanity check

if len(all_hlt_path_names_seen) == 0:
   print >> stderr,"no HLT paths found, exiting"
   exit(1)

#--------------------
# now that we know all path names of all runs, produce the CSV
import csv

previous_hlt_key = None

fout = open(csv_output_file,"w")

csv_writer = csv.writer(fout,delimiter=";")

csv_writer.writerow(['Table of HLT prescale factors'])
csv_writer.writerow([])
csv_writer.writerow(['Explanation:'])
csv_writer.writerow(['number(s) = prescale factor(s), HLT path present in this menu'])
csv_writer.writerow(['empty = HLT path NOT present in this menu'])
csv_writer.writerow(['0 = HLT path present in this menu but prescale factor is zero'])
csv_writer.writerow(['U = could not retrieve menu for this HLT key from confDB'])
csv_writer.writerow([])
csv_writer.writerow([])

# write the header line
column_names = [ 'run','' ]
column_names.extend(all_hlt_path_names_seen)
csv_writer.writerow(column_names)

csv_writer.writerow([])

for run in runs:
   hlt_key = runKeys[run]

   if hlt_key == previous_hlt_key:
       # the hlt key is the same as for the previous run
       # just reuse the existing contents of the variable 'values'
       # (apart from the run number)
       values[0] = run
       csv_writer.writerow(values)
       continue

   # HLT key has changed

   # insert a line with the menu's name
   #
   # check whether we actually could determine the menu
   if not hlt_path_names_table.has_key(hlt_key):
       # could previously not retrieve the python
       # configuration for this key 
       #
       # put some warnings in the output table

       csv_writer.writerow([hlt_key, "COULD NOT RETRIEVE MENU FROM CONFDB"])

       values = [ run , '' ]
       values.extend(len(all_hlt_path_names_seen) * [ "U" ])

       csv_writer.writerow(values)

       previous_hlt_key = hlt_key
       continue

   # everything seems ok for this key

   csv_writer.writerow([hlt_key])

   # now put together the list of prescales
   values = [ run , '' ]

   # find out which HLT keys were present and which prescale factors
   # they had
   for hlt_path in all_hlt_path_names_seen:

       if hlt_path in hlt_path_names_table[hlt_key]:
           # this HLT path was present in this menu
           # check whether there was an entry in the prescale
           # table

           # get the prescale factors (list) or just a list with 1
           # if this path was not present in the prescale table
           # for this menu
           prescales = prescaleTable[hlt_key].get(hlt_path, [ 1 ] )

           # merge the values to one comma separated string 
           # print "prescales=",prescales
           values.append(",".join([str(x) for x in prescales]))

       else:
           # path not found for this hlt key. append
           # an empty string for this column
           values.append('')

   # end loop over hlt paths        

   csv_writer.writerow(values)

   previous_hlt_key = hlt_key

# end loop over runs

fout.close()

print >> stderr,"created CSV file",csv_output_file,". Field delimiter is semicolon."


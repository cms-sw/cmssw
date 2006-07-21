#
# Python functions used by 'hlt' script to generate a config file
# that defines triggers with prescale values.
#
# Original Authors: David Lange, Doug Wright


import os,sys,commands,re

#
#....global variables
#
list = [] #....trigger list dictionary 
hlt_file_dir = "HLTrigger/HLTfilters/utilities"  #....location of hlt config files

#
#....Add trigger to the list, check that trigger name is unique
#
def trig(name,prescale):
 if not_in_list(name):
    list.append( (name,prescale) )
 else:
    print "Error:", name,\
          "is already defined! Did not change prescale to", prescale
    
#
#....Write config file to load all of the triggers with prescale values set.
#    Will include trig.cfg if the file exists in CMSSW_SEARCH_PATH

def make_cfg_file(file):
  print "Creating file:",file,"with trigger definition."
  f = open(file,'w')
  for trig in list:
    name = trig[0]
    prescale = str( trig[1] )
    name_prescale = "HLT%sPrescale" % name
    name_sequence = "HLT%sSequence" % name
    name_path     = hlt_file_dir + "/" + name_sequence + ".cfi"
    sequence = name_prescale
    
    f.write("\n")

    #....Setup prescale module
    f.write("module %s= HLTPrescaler { \n" % name_prescale)
    f.write("   uint32 prescaleFactor = %s\n" % prescale)
    f.write("   bool makeFilterObject = true \n }\n")

    #...Add include statement if name.cfg file exists
    if findInc(name_path):
        f.write("include \"%s\"\n" % name_path)
        sequence += ", " + name_sequence
    else:
        print "Skipping missing include " +name_path

    #....Add trigger path
    f.write("path %s ={%s}\n"  %  (name,sequence) )
        
  f.write("\n")
  f.close()

#
#....Return True if a name.cfg file exists somewhere in CMSSW_SEARCH_PATH
#
def findInc(name):
    enVar = os.environ.get('CMSSW_SEARCH_PATH')
    if enVar:
        dirList=enVar.split(':')
        for dir in dirList:
            if os.path.isfile(dir+'/'+name):
                return True
    else:
        print 'CMSSW_SEARCH_PATH is not defined.'
    return False

#
#....Return True if the named trigger is not in the list
#
def not_in_list(name):
    for trig in list:
        if trig[0] == name : return False
    return True

      

#
# Python functions used by 'hlt' script to generate a config file
# that defines triggers with prescale values.
#
# Original Authors: David Lange, Doug Wright


import os,sys,commands,re

#
#....global variable
#
hlt = {} #....trigger list dictionary (i.e., a hash table)
hlt_file_dir = "HLTrigger/HLTfilters/utilities"  #....location of hlt config files

#
#....Add trigger to the list, check that trigger name is unique
#
def trig(name,prescale):
 if name not in hlt:
    hlt[name] = prescale
 else:
    print "Error:", name,\
          "is already defined! Did not change prescale to", prescale
    
#
#....Write config file to load all of the triggers with prescale values set.
#    Will include trig.cfg if the file exists in CMSSW_SEARCH_PATH

def make_cfg_file(file):
  print "Creating file:",file,"with trigger definition."
  f = open(file,'w')
  for name in hlt.keys():
    prescale=str(hlt[name])
    name_prescale = "HLT" + name + "Prescale"
    name_sequence = "HLT" + name + "Sequence"
    name_path     = hlt_file_dir + "/" + name_sequence + ".cfi"
    sequence = name_prescale
    
    f.write("\n")
    #....Add include statement if name.cfg file exists
    if findInc(name_path):
        f.write("include \"" + name_path + "\"\n")
        sequence += ", " + name_sequence
    else:
        print "Skipping missing include " +name_path
    
    f.write("module   " + name_prescale +"= HLTPrescaler { \n   uint32 prescaleFactor = " + prescale + "\n   bool makeFilterObject = true \n }\n")
    f.write("path     " + name + " =")
    f.write("{" + sequence + "}\n")
        
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

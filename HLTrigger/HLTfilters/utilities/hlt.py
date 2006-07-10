#
# python script to setup hlt trigger table
#
# Doug Wright

import os,sys,commands,re

hlt = {} # trigger

#
#....add trigger to the list, check that trigger name is unique
#
def trig(name,prescale):
 if name not in hlt:
    hlt[name] = prescale
 else:
    print "Error:", name,\
          "is already defined! Did not change prescale to", prescale

# look for trigger path include file 
def findInc(name):
    env= os.environ.get('CMSSW_SEARCH_PATH')
    dirList=env.split(':')

    if not dirList: return 0

    for dir in dirList:
        if not os.path.isdir(dir): continue
        fname=dir+'/'+name
        print fname
        if os.path.isfile(fname):
            return 1

    return 0
    
#
#....use trigger list to create the .cfg file to load all of the triggers
#    with prescale values
#
def make_cfg_file(file):
  print "Creating file:",file,"with trigger definition."
  f = open(file,'w')
  for name in hlt.keys():
    prescale=str(hlt[name])
    name_prescale = "HLT" + name + "Prescale"
    name_sequence = "HLT" + name + "Sequence"
    hlt_sequence_path = "HLTrigger/HLTfilters/data/" + name_sequence + ".cfi"
    f.write("\n")
    foundValidIncFile=findInc(hlt_sequence_path)
    if ( foundValidIncFile == 1 ):
        f.write("include \"" + hlt_sequence_path + "\"\n")
    else:
        print "Skipping missing include " + hlt_sequence_path + "\n"
    
    f.write("module   " + name_prescale +"= Prescaler { int32 prescaleFactor = " + prescale + " }\n")
    f.write("path     " + name + " =")
    f.write("{" + name_prescale)
    if ( foundValidIncFile == 1 ):
        f.write(", Hlt" + name_sequence + "}\n")
    else:
        f.write("}\n")
        
  f.write("\n")
  f.close()


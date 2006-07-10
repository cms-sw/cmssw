#
# python script to setup hlt trigger table
#
# Doug Wright

hlt = {} # trigger

#
#....add trigger to the list, check that trigger name is unique
#
def trig(name,prescale,l1bits):
 if name not in hlt:
    hlt[name] = prescale
 else:
    print "Error:", name,\
          "is already defined! Did not change prescale to", prescale

#
#....use trigger list to create the .cfg file to load all of the triggers
#    with prescale values
#
def make_cfg_file(file):
  print "Creating file:",file,"with trigger definition."
  f = open(file,'w')
  for name in hlt.keys():
    prescale=str(hlt[name])
    name_prescale = name + "Prescale"
    name_sequence = name + "Sequence"
    hlt_sequence_path = "HLTCore/HLTfilters/data/"
    f.write("\n")
    f.write("include \"" + hlt_sequence_path + name +".cfi\"\n")
    f.write("module   " + name_prescale +"= Prescaler { int32 prescaleFactor = " + prescale + " }\n")
    f.write("path     " + name + " =")
    f.write(", Hlt" + name_prescale)
    f.write(", Hlt" + name_sequence + "}\n")
  f.write("\n")
  f.close()


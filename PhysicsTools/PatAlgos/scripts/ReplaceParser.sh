#!/bin/bash

# pipe the config files given on the command line
cat $* | \
# remove all comments
sed 's/#/\nREMOVE/' | \
sed 's/\/\//\nREMOVE/' | \
grep -v REMOVE | \
# remove lines only containing whitespace or empty lines
grep -v '^[ ]*$' | \
# parse all module definitions out of the input module files
awk '/(module)/ {                               # match the string "module"
  open = -1;                                    # initialize nr brackets to -1 until the opening bracket is read
  while (open != 0) {                           # loop until module opening & closing bracket is found
    print $0;                                   # print current line
    for (i=1; i<=NF; i++) {                     # loop over fields looking for brackets
      if ($i~"{") {                             # if opening bracket found...
        if (open == -1) open = 0;               # ... and none found yet -> initialize to 0 and...
        open++;                                 # increment nr brackets
      }
      if ($i~"}") { open--; }                   # if closing bracket found -> decrement nr brackets
    }
    if (open != 0) getline;                     # if module not finished yet -> read next line
  }
}' | \
# process now the module definitions to produce replace statements
awk '/(module)/ {                               # match the string "module"
  modname = $2;                                 # store module name before changing the fields
  open = -1;                                    # initialize nr brackets to -1 until the opening bracket is read
  while (open != 0) {                           # loop until module opening & closing bracket is found
    for (i=1; i<=NF; i++) {                     # loop over fields looking {, } or =
      if ($i~"{") {                             # if opening bracket found...
        if (open == -1) open = 0;               # ... and none found yet -> initialize to 0 and...
        open++;                                 # increment nr brackets
      }
      if ($i~"}") { open--; }                   # if closing bracket found -> decrement nr brackets
      if ($i~"=" && open>0) {                   # if a configurable definition found
        print "replace "modname"."$(i-1)"%"     # print out the start of the replace statement (end with % as tmp hack)
        if ($(i+1)~"{") {                       # if the parameter value contains brackets
          open2 = -1;                           # initialize nr brackets to -1 until the opening bracket is read
          while (open2 != 0) {                  # loop until parameter value opening & closing bracket is found
            for (j=1; j<=NF; j++) {             # loop over all fields
              if (j<i) $j="";                   # zero out fields of parameter type and name
              if (j>=i) {                       # process parameter value fields
                if ($j~"{") {                   #  if opening bracket found...
                  if (open2 == -1) open2 = 0;   # ... and none found yet -> initialize to 0 and...
                  open2++;                      # increment nr brackets
                }
                if ($j~"}") { open2--; }        # if closing bracket found -> decrement nr brackets
              }
            }
            print $0                            # print whatever remains on this line
            if (open2 != 0) { i = 1; getline; } # if param value unfinished -> read next line + make sure all fields get processed
            if (open2 == 0) { i = NF; }         # if full param value read, make sure to stop the for-loop
          }
        } else {                                # if the parameter value doesnt contain brackets
          for (j=1; j<i; j++) $j="";            # zero out fields of parameter type and name
          print $0                              # print whatever remains on this line
        }
      }
    }
    if (open != 0) getline;                     # get next line if module opening & closing bracket not found yet
  }
}' | \
# join again deliberately-split parameter defintion lines
sed '/\%$/N;s/\%\n */ /'

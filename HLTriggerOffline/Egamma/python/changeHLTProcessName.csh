#!/usr/bin/tcsh

# Change process name in Matthias' config files

foreach file (`ls HLT_*_cfi.py`)    
# echo sed 's/'\"'$1'\"'/'\"'$2'\"'/g' ${file} > ${file}new
 sed s/'\"'$1'\"'/'\"'$2'\"'/g ${file} > ${file}new
 mv ${file}new ${file}
end

echo Process name changed from $1 to $2 in all files
 

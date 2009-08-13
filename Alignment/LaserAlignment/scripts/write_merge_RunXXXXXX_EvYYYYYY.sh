#!/bin/sh

# define needed variables
name_part1=file:/tmp/aperiean/RunXXXXXX/RunXXXXXX_EvYYYYYY/TkAlLAS_RunXXXXXX_EvYYYYYY_F
name_part2=.root 

echo "Write the merge_RunXXXXXX_EvYYYYYY.py file..."

inum_files=$1

# copy first part
    less merge_RunXXXXXX_EvYYYYYY_part1.py > merge_RunXXXXXX_EvYYYYYY.py
# loop and add input files    
    for (( i = 1; i <= $inum_files; ++i ))
	do
	echo "    '$name_part1$i$name_part2'," >> merge_RunXXXXXX_EvYYYYYY.py
	done
# copy second part 
    less merge_RunXXXXXX_EvYYYYYY_part2.py >> merge_RunXXXXXX_EvYYYYYY.py

echo "file merge_RunXXXXXX_EvYYYYYY.py is done. Next step..."

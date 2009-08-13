#!/bin/sh

### define needed variables

### copy default CRAFT_cff.py in py_config
for (( i = 1;  i <= nFiles; ++i ))
do
    ## copy first part
    less t0ProducerStandalone_part1.py > t0ProducerStandalone_RunXXXXXX_EvYYYYYY_F$i.py
    ## get the input file
    LINE_NUM=`expr \( ${i} + 1 \)`p
    sed -n $LINE_NUM t0ProducerStandalone_part2_Run.py >> t0ProducerStandalone_RunXXXXXX_EvYYYYYY_F$i.py
    ## copy third part 
    less t0ProducerStandalone_part3_RunXXXXXX_EvYYYYYY.py >> t0ProducerStandalone_RunXXXXXX_EvYYYYYY_F$i.py
    sed -i "s/_F0/_F$i/g" t0ProducerStandalone_RunXXXXXX_EvYYYYYY_F$i.py
    mv t0ProducerStandalone_RunXXXXXX_EvYYYYYY_F$i.py py_config
done

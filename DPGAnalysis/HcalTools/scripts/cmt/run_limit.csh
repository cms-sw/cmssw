#!/bin/csh

# max= Njobs+1
set max=11
echo ${max}

set i=1
while ( ${i} < ${max} )
echo ${i}

#bsub -q 1nh myjob.csh ${i}
#bsub -q 1nw myjob.csh ${i}
#bsub -q 8nm myjob.csh ${i}
bsub -q 8nh myjob.csh ${i}

@ i = ${i} + "1"
end

###  bjobs
##  bjobs -a
##  bkill <jobID>
##  bjobs -l NN??
##  bjobs

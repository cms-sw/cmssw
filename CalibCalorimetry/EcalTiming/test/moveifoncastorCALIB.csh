#!/bin/tcsh -f


foreach run ( `/bin/ls | grep "Calib_"`)
echo "Does $run exist on castor?"

if ( `rfdir /castor/cern.ch/user/c/ccecal/CRUZET4/Calibration | grep -c $run` < 1 ) then
echo "good it doesn't" 
tar -cvf ${run}.tar $run
rfcp ${run}.tar run /castor/cern.ch/user/c/ccecal/CRUZET4/Calibration/.
rm -rf ${run}.tar $run
endif 
end

#end of file

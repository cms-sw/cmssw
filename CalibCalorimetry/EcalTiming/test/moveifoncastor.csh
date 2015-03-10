#!/bin/tcsh -f


foreach run ( `/bin/ls | grep "Laser_"`)
echo "Does $run exist on castor?"
if ( `rfdir /castor/cern.ch/user/c/ccecal/CRUZET4/CosmicsAnalysis/ | grep -c $run` > 0 ) then
echo "yes it is" 
rm -rf $run
echo "now removed"
endif

end

#end of file

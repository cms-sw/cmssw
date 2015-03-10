#!/bin/tcsh -f

foreach typ ( Laser Timing )
  set typu=`echo $typ | tr "[:lower:]" "[:upper:]"`
  echo $typu
  foreach runf ( `/bin/ls plots/$typ/*/*.root`)
    echo "File location is $runf"
    echo "Does $runf:t exist on castor?"
    echo "Looking in "
    echo "/castor/cern.ch/user/c/ccecal/${typ} and /castor/cern.ch/user/c/ccecal/${typu}"
    if ( `rfdir /castor/cern.ch/user/c/ccecal/$typ | grep -c $runf:t` > 0 ||`rfdir /castor/cern.ch/user/c/ccecal/$typu | grep -c $runf:t` > 0 ) then
    echo "yes it is" 
    rm -rf $runf
    echo "now removed"
    else
    echo "it needss to be moved: doing it now."
    echo "..."
    rfcp $runf /castor/cern.ch/user/c/ccecal/$typ/$runf:t
    rm -rf $runf
    endif
    
  end
end

#end of file

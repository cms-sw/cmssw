#!/bin/tcsh 
set ohone = `(grep --count " 0\.0001" complines.out)`
set wccnt = `(wc -l complines.out | awk '{print $1}')`
if ( $ohone == $wccnt ) then
    echo All differences in position are less than 0.0001.  There are $ohone differences.
else
    set ndifs = $ohone - $wccnt 
    echo Some ($ndifs) differences are not less than 0.0001.  Please check/verify.
endif


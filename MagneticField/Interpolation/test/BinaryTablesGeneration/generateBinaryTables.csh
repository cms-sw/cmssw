#!/bin/tcsh

setenv PATH {$CMSSW_BASE}/test/${SCRAM_ARCH}/:${PATH}

foreach file ( `cat tableList.txt` )
    set number=`basename $file | tr -cd "[:digit:]"`
    prepareFieldTable $file grid.${number}.bin
    
    if ($?) echo ERROR table not processed: $file
end

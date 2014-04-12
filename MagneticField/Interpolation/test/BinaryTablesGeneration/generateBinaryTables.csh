#!/bin/tcsh

foreach file ( `cat tableList.txt` )
    set number=`basename $file | tr -cd "[:digit:]"`
    prepareFieldTable $file grid.${number}.bin
    
    if ($?) echo ERROR table not processed: $file
end

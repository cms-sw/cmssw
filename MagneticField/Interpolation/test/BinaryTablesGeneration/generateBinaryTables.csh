#!/bin/tcsh

foreach file ( `cat tableList.txt` )
    set number=`basename $file | tr -cd "[:digit:]"`
    echo prepareFieldInterpolation $file grid.${number}.bin
end

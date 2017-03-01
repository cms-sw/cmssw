#!/bin/tcsh
setenv PATH {$CMSSW_BASE}/test/${SCRAM_ARCH}/:${PATH}

foreach file ( `cat tableList.txt` )
    set number=`basename $file | tr -cd "[:digit:]"`
    set fdir=`dirname $file`
    set sector=`basename $fdir | grep -o "s.." | tr -cd "[:digit:]"`
    set dir=s${sector}
    mkdir -p $dir
#    echo $dir $sector
    prepareFieldTable $file ${dir}/grid.${number}.bin $sector   
    if ($?) echo ERROR table not processed: $file
end

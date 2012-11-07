#!/bin/tcsh

set base = RelVal_HLT

foreach gtag ( STARTUP DATA )

  echo
  echo $gtag

  foreach table ( GRun PIon 5E33v4 7E33v2 7E33v3 7E33v4 8E33v1 8E33v2 HIon )

    echo
    set name = ${table}_${gtag}
    echo $name

    foreach task ( ONLINE_HLT RelVal_HLT_Reco RelVal_HLT2 )

      echo
      echo "Compare  ${base}_${name} to  ${task}_${name}"
      echo "diff -C0 ${base}_${name}.log ${task}_${name}.log"
#           diff -C0 ${base}_${name}.log ${task}_${name}.log | grep L1T
            diff -C0 ${base}_${name}.log ${task}_${name}.log | grep "HLT-Report "
#           diff -C0 ${base}_${name}.log ${task}_${name}.log | grep "TrigReport "

    end

  end

end

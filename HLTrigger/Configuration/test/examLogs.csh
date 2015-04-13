#!/bin/tcsh

set base = RelVal_HLT

foreach gtag ( MC DATA )

  echo
  echo $gtag

  foreach table ( GRun 50nsGRun HIon PIon 25ns14e33_v1 50ns_5e33_v1 Fake )

    echo
    set name = ${table}_${gtag}
    echo $name

    foreach task ( OnLine_HLT RelVal_HLT_Reco RelVal_HLT2 )

      echo
      echo "Compare  ${base}_${name} to  ${task}_${name}"
      echo "diff -C0 ${base}_${name}.log ${task}_${name}.log"
#           diff -C0 ${base}_${name}.log ${task}_${name}.log | grep L1T
            diff -C0 ${base}_${name}.log ${task}_${name}.log | grep "HLT-Report "
#           diff -C0 ${base}_${name}.log ${task}_${name}.log | grep "TrigReport "

    end

  end

end

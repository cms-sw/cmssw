#!/bin/bash

test_exe=$CMSSW_BASE/test/$SCRAM_ARCH/xhtmlmaker_t
testoutput=test.xhtml

a=`$test_exe 2>&1`
sa=$?
if [ $sa != 0 ]; then
    echo $a
    /bin/rm -f $testoutput
    exit $sa
fi

b=`xmllint --dtdvalid http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd --noout $testoutput`
sb=$?
if [ $sb != 0 ]; then
    echo $b
    /bin/rm -f $testoutput
    exit $sb
fi

/bin/rm -f $testoutput

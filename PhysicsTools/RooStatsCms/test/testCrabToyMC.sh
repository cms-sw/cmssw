#!/bin/sh
#########################
#
# Driver script for Toy Monte Carlo submission with CRAB 
#
# author: Luca Lista, INFN
#                      
#########################

if [ -e outputToy ]; then 
  rm -rf outputToy 
fi
mkdir outputToy

i="$1"
if [ "$i" == "help" ]; then
  echo "usage: testCrabToyMC.sh <job index> [<max events>]"
  exit 0;
fi
if [ "$i" = "" ]; then
  echo "Error: missing job index"
  exit 1;
fi
echo "max events from CRAB: $MaxEvents"
n="$MaxEvents" 
if [ "$n" = "" ]; then
  n="$2"
fi
if [ "$n" = "" ]; then
  echo "Error: missing number of experiments"
  exit 2;
fi

rm -f fitResults.txt
echo "# s_true s s_err b_true b b_err" > fitResults.txt
j=`expr \( $i - 1 \) \* $n + 1`
jmax=`expr $j + $n - 1`
echo "job number: #$i with $n experiments ($j to $jmax)"
while [ $j -le $jmax ]; do
echo "running toy MC : ./testCrabToyMC -s $j"
        ./testCrabToyMC -s $j >& log.txt
        # retrieve fit values and store into a single file  
	grep 'RooRealVar::s ' log.txt| grep -v '+/-' | cut -f 3 -d\ | tr '\n' ' ' >> fitResults.txt 
	grep 'RooRealVar::s ' log.txt| grep '+/-' | cut -f 3,5 -d\ | tr '\n' ' ' >> fitResults.txt  
	grep 'RooRealVar::b ' log.txt| grep -v '+/-' | cut -f 3 -d\ | tr '\n' ' ' >> fitResults.txt 
	grep 'RooRealVar::b ' log.txt| grep '+/-' | cut -f 3,5 -d\ | tr '\n' ' ' >> fitResults.txt  
	echo >> fitResults.txt
	mv binnedChi2Fit.eps outputToy/binnedChi2Fit_$j.eps
	mv log.txt outputToy/log_$j.txt 
        j=`expr $j + 1`
done
echo "pack the results"
tar cvfz outputToy.tgz outputToy/





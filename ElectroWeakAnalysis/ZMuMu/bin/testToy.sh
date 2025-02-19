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
l="$2"
if [ "$i" == "help" ]; then
  echo "usage: testToy.sh <job index> <luminosity>"
  exit 0;
fi
if [ "$i" = "" ]; then
  echo "Error: missing job index"
  exit 1;
fi
if [ "$i" = "" ]; then
  echo "Error: missing job index"
  exit 1;
fi
if [ "$l" = "" ]; then
  echo "Error: missing luminosity"
  exit 1;
fi

rm -f fitResults.txt
echo "# par_name init_val fin_val par_err global_corr" > fitResults.txt
((j = 1)) 
((jmax=$i))

echo "job number: #$i"
echo "j value: #$j"
while [ $j -le $jmax ]; do
echo "running toy MC : zMuMuRooFit -i Analisi_45pb.root -o out.root -r 2 2 10 60  -t -s $j -l $l"
        zMuMuRooFit -i Analisi_45pb.root  -o out.root -r 2 2 10 60 -t -s $j -l $l >& log.txt
        # retrieve fit values and store into a single file  
	grep "        Yield" log.txt  >> fitResults.txt
	grep "        a0" log.txt  >> fitResults.txt	
	grep "        a1" log.txt  >> fitResults.txt	
	grep "        a2" log.txt  >> fitResults.txt	
	grep "        alpha" log.txt  >> fitResults.txt	
	grep "        b0" log.txt  >> fitResults.txt	
	grep "        b1" log.txt  >> fitResults.txt	
	grep "        b2" log.txt  >> fitResults.txt	
	grep "        beta" log.txt  >> fitResults.txt	
	grep "        eff_hlt" log.txt  >> fitResults.txt	
	grep "        eff_iso" log.txt  >> fitResults.txt
	grep "        eff_sa" log.txt  >> fitResults.txt
	grep "        eff_tk" log.txt  >> fitResults.txt
	grep "        nbkg_mumuNotIso" log.txt  >> fitResults.txt
	grep "        nbkg_mutrk" log.txt  >> fitResults.txt
	mv mass.eps outputToy/mass_$j.eps
	mv out.root outputToy/out_$j.root
	mv log.txt outputToy/log_$j.txt 
        mv fitResults.txt  outputToy/
        ((j= $j + 1))
done
echo "pack the results"
tar cvfz outputToy.tgz outputToy/


#!/bin/sh
#########################
#                       # 
# author: Pasquale Noli #
# INFN Naples           #
# script to run ToyMC   #
#                       #
#########################

if [ -e outputToy ]; then 
  rm -rf outputToy 
fi
mkdir outputToy

i="$1"
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

iterations=1000
nz=5077000 
eff_trk=0.75
eff_sa=0.75
eff_iso=0.75
eff_hlt=0.75
bkg_scale=100
max_mass=140
rm -f fitResult.txt
echo "# Yield eff_trk eff_sa eff_iso eff_hlt" > fitResult.txt
echo "$nz $eff_trk $eff_sa $eff_iso $eff_hlt" >> fitResult.txt
j=`expr \( $i - 1 \) \* $n + 1`
jmax=`expr $j + $n - 1`
echo "job number: #$i with $n experiments ($j to $jmax)"
while [ $j -le $jmax ]; do
echo "running toy MC : ./toyMonteCarlo -n 1 -s $j  -y $nz -T $eff_trk -S $eff_sa -I $eff_iso -H $eff_hlt -f $bkg_scale -M $max_mass"
     	./toyMonteCarlo -n 1 -s $j  -y $nz -T $eff_trk -S $eff_sa -I $eff_iso -H $eff_hlt -f $bkg_scale -M $max_mass
        echo "merging histograms: mergeTFileServiceHistograms -o analysis_$j.root -i zmm_1.root bkg_1.root"
	mergeTFileServiceHistograms -o analysis_$j.root -i zmm_1.root bkg_1.root
        echo "performing fit: zFitToyMc -M 140 analysis_$j.root >& log_fit_$j.txt"
	./zFitToyMc -M 140 analysis_$j.root >& log_fit_$j.txt
	mv  analysis_$j.root outputToy
	mv *eps outputToy
	mv log_fit_$j.txt outputToy 
        j=`expr $j + 1`
done
echo "pack the results"
tar cvfz outputToy.tgz outputToy/


#########################
#                       # 
# author: Pasquale Noli #
# INFN Naples           #
# script to run ToyMC   #
#                       #
#########################

#!/bin/csh
if(-e outputToy) rm -rf outputToy
mkdir outputToy
set i=1
set iterations = $1
set nz = $2 
set eff_trk = $3
set eff_sa = $4
set eff_iso = $5
set eff_hlt = $6
set bkg_scale = $7
set max_mass = $8
rm -f fitResult.txt
echo "# Yield eff_trk eff_sa eff_iso eff_hlt" > fitResult.txt
echo "$nz $eff_trk $eff_sa $eff_iso $eff_hlt" >> fitResult.txt
while ($i <= $iterations)
	echo  $i
     	toyMonteCarlo -p Analysis_10pb.root -n 1 -s $i  -y $nz -T $eff_trk -S $eff_sa -I $eff_iso -H $eff_hlt -f $bkg_scale -M $max_mass
	# -S 1 -T 1 -H 1
	mergeTFileServiceHistograms  -i zmm_1.root bkg_1.root
	mv out.root analysis_$i.root
	zChi2Fit -c -M 120 analysis_$i.root >& log_fit_$i.txt
	#mv zmm_1.root zmm_$i.root
	#zFitToyMc  zmm_1.root >& log_fit_$i.txt
	#mv  analysis_$i.root outputToy
	#mv zmm_$i.root outputToy
	mv  analysis_$i.root outputToy
	mv *eps outputToy
	mv log_fit_$i.txt outputToy 
	set i=`expr $i + 1`
end
	root -l -q create_tree_for_toyMC.C
	root -l -q pulls.C
	mv fitResult.root outputToy
	mv fitResult.txt outputToy
        mv pulls.eps outputToy
echo "Pulls are saved into pulls.eps"
    
#gv pulls.eps 

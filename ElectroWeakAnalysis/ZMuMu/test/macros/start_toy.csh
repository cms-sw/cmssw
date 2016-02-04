#########################
#                       # 
# author: Pasquale Noli #
# INFN Naples           #
# script to run ToyMC   #
#                       #
#########################

#!/bin/csh
rm fitResalt*
mkdir outputToy
set i=1
while ($i <= $1)
	echo  $i
     	toyMonteCarlo -n 1 -s $i 
	mergeTFileServiceHistograms -o analysis_$i.root -i zmm_1.root bkg_1.root
	zFitToyMc analysis_$i.root >& log_fit_$i.txt
	rm -f analysis_$i.root
	mv *ps outputToy
	mv log_fit_$i.txt outputToy
	set i=`expr $i + 1`
end
	root -l -q create_tree_for_toyMC.C
	root -l -q pulls.C
	mv pulls.eps outputToy
echo "Pulls are saved into pulls.eps"
    
gv outputToy/pulls.eps 

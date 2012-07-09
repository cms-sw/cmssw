read -p "PARALLEL PROCESSING: how many cores can you afford? " answer

echo "do the training, make sure you extracted the variables first using cmsRun VariableExtractor_LR_cfg.py (usually QCD events are used)"

#!/bin/sh
path_to_rootfiles=/user/pvmulder/NewEraOfDataAnalysis/BTagServiceWork/CMSSW_4_4_4_development/src/BTagging/RootFilesAlexis
prefix=CombinedSV
Combinations="NoVertex_B_DUSG NoVertex_B_C PseudoVertex_B_DUSG PseudoVertex_B_C RecoVertex_B_DUSG RecoVertex_B_C"

CAT="Reco Pseudo No"
for i in $CAT ; do
##	hadd $path_to_rootfiles/${prefix}${i}Vertex_DUSG.root $path_to_rootfiles/${prefix}${i}Vertex_D.root $path_to_rootfiles/${prefix}${i}Vertex_U.root $path_to_rootfiles/${prefix}${i}Vertex_S.root $path_to_rootfiles/${prefix}${i}Vertex_G.root 
	hadd $path_to_rootfiles/${prefix}${i}Vertex_B_DUSG.root $path_to_rootfiles/${prefix}${i}Vertex_DUSG.root $path_to_rootfiles/${prefix}${i}Vertex_B.root
	hadd $path_to_rootfiles/${prefix}${i}Vertex_B_C.root $path_to_rootfiles/${prefix}${i}Vertex_C.root $path_to_rootfiles/${prefix}${i}Vertex_B.root
done

g++ ../histoJetEtaPt.cpp `root-config --cflags --glibs` -o histos
./histos $path_to_rootfiles $prefix

g++ ../fitJetEtaPt.cpp `root-config --cflags --glibs` -o fitter
mkdir weights
for i in $CAT ; do
	./fitter ${prefix}${i}Vertex_B_C_histo.root
  mv out.txt weights/${prefix}${i}Vertex_BC_histo.txt
  mv out.root weights/${prefix}${i}Vertex_BC_histo_check.root

	./fitter ${prefix}${i}Vertex_B_histo.root ${prefix}${i}Vertex_C_histo.root
	mv out.txt weights/${prefix}${i}Vertex_B_C_ratio.txt
  mv out.root weights/${prefix}${i}Vertex_B_C_ratio_check.root
	
	./fitter ${prefix}${i}Vertex_B_DUSG_histo.root 
	mv out.txt weights/${prefix}${i}Vertex_BDUSG_histo.txt
  mv out.root weights/${prefix}${i}Vertex_BDUSG_ratio_check.root

	./fitter ${prefix}${i}Vertex_B_histo.root ${prefix}${i}Vertex_DUSG_histo.root
	mv out.txt weights/${prefix}${i}Vertex_B_DUSG_ratio.txt
  mv out.root weights/${prefix}${i}Vertex_B_DUSG_ratio_check.root
done

for j in $( ls ../MVATrainer_*cfg.py ) ; do cp $j . ; done
for j in $( ls MVATrainer_*cfg.py ) ; do
	sed -i 's@CombinedSV@'$prefix'@g#' $j # change the path of the input rootfiles
	sed -i 's@"./'$prefix'@"'$path_to_rootfiles'/'$prefix'@g#' $j # change the path of the input rootfiles
done

for j in $( ls ../Save_*xml ) ; do cp $j . ; done
for j in $( ls Save*xml ) ; do
	sed -i 's@CombinedSV@'$prefix'@g#' $j # change the name of the tag in the file
done

files=("MVATrainer_No_B_DUSG_cfg.py" "MVATrainer_No_B_C_cfg.py" "MVATrainer_Pseudo_B_DUSG_cfg.py" "MVATrainer_Pseudo_B_C_cfg.py" "MVATrainer_Reco_B_DUSG_cfg.py" "MVATrainer_Reco_B_C_cfg.py")
l=0
while [ $l -lt 6 ]
do
	jobsrunning=0
	while [ $jobsrunning -lt $answer ]
	do
		nohup cmsRun ${files[l]} &
		let jobsrunning=$jobsrunning+1
		let l=$l+1
	done
	wait
done


g++ ../biasForXml.cpp `root-config --cflags --glibs` -o bias
./bias $path_to_rootfiles $prefix
echo "ARE YOU SURE THAT YOU HAVE ENOUGH STATISTICS TO DETERMINE THE BIAS ACCURATELY?"

for i in $Combinations ; do
	sed -n -i '/<bias_table>/{p; :a; N; /<\/bias_table>/!ba; s/.*\n//}; p' Train_${i}.xml # remove bias table in file
	for line in $( cat ${i}.txt ) ; do 
#		echo "$line" 
		newline2=$(cat Train_${i}.xml | grep -n '</bias_table>' | grep -o '^[0-9]*')
		sed -i ${newline2}'i\'$line Train_${i}.xml
	done 
done

Vertex="NoVertex PseudoVertex RecoVertex"
Flavour="B_DUSG B_C"
for k in $Vertex ; do
	for l in $Flavour ; do
		sed -n -i '/<bias_table><!--'$l'-->/{p; :a; N; /<\/bias_table><!--'$l'-->/!ba; s/.*\n//}; p' Train_${k}.xml # remove bias table in file
		for line in $( cat ${k}_${l}.txt ) ; do 
#			echo "$line" 
			newline1=$(cat Train_${k}.xml | grep -n '</bias_table><!--'$l'-->' | grep -o '^[0-9]*')
			sed -i ${newline1}'i\'$line Train_${k}.xml
		done
	done
done

CombinationsArray=("NoVertex_B_DUSG" "NoVertex_B_C" "PseudoVertex_B_DUSG" "PseudoVertex_B_C" "RecoVertex_B_DUSG" "RecoVertex_B_C")
l=0
while [ $l -lt 6 ]
do
	jobsrunning=0
	while [[ $jobsrunning -lt $answer && $jobsrunning -lt 6 ]] 
	do
		echo Processing ${CombinationsArray[l]}
 		mkdir tmp${CombinationsArray[l]}
 		cd tmp${CombinationsArray[l]}
 		#echo Train_${CombinationsArray[l]}.xml
		#echo train_${CombinationsArray[l]}_save.root
		nohup mvaTreeTrainer ../Train_${CombinationsArray[l]}.xml tmp.mva ../train_${CombinationsArray[l]}_save.root &
		cd ..
		let jobsrunning=$jobsrunning+1
		let l=$l+1
	done
	wait
done

VertexCategory=("NoVertex" "PseudoVertex" "RecoVertex")
l=0
while [ $l -lt 3 ]
do
	jobsrunning=0
	while [[ $jobsrunning -lt $answer  && $jobsrunning -lt 3 ]] 
	do
		#echo tmp${VertexCategory[l]}_B_*/*.xml
		cp tmp${VertexCategory[l]}_B_*/*.xml .
 		nohup mvaTreeTrainer -l Train_${VertexCategory[l]}.xml ${prefix}${VertexCategory[l]}.mva train_${VertexCategory[l]}_B_DUSG_save.root train_${VertexCategory[l]}_B_C_save.root &
		let jobsrunning=$jobsrunning+1
		let l=$l+1
	done
	wait
done

#
echo "do cmsRun ../copyMVAToSQLite_cfg.py to copy the mva training output to sqlite format"
echo "run the validation from Validation/RecoB/test/ -> usually on ttbar events"

#cmsRun ../copyMVAToSQLite_cfg.py
#cmsRun reco_validationNEW_CSVMVA_categories_cfg.py

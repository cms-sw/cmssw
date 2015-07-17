#!/bin/bash



if [[ "$#" -eq 0 ]]; then
echo "ERROR: the script needs at least one argument. Relunch it with one of the following options:"
echo "source validate_MT2.sh inputFolder"
echo "./validate_MT2.sh fileA.root fileB.root labelA labelB outputFolderName"
exit;
fi;

if [[ "$#" -eq 1 ]]; then
    X=$PWD/$1; 

# $1 is VALIDATEMT2
# copy files mt2_tree.root inside
# CMGTools/TTHAnalysis/cfg/VALIDATEMT2/SNT/mt2/mt2_tree.root
# CMGTools/TTHAnalysis/cfg/VALIDATEMT2/ETHCERN/mt2/mt2_tree.root
# to dun execute: source validate_MT2.sh VALIDATEMT2


    if test \! -d $X/ETHCERN; then echo "Did not find ETHCERN in $X"; exit 1; fi
    test -L $PWD/SNT || ln -sd $PWD/SNT $X/ -v;
    ( cd ../python/plotter; 
	python mcPlots.py -f --tree mt2  -P $X susy-mT2/validation_mca_MT2.txt susy-mT2/validation_MT2.txt susy-mT2/validation_plots_MT2.txt --pdir plots/70X/validation -p ref_ttHWWdata,ttHWWdata -u -e --plotmode=norm --showRatio --maxRatioRange 0.65 1.35 --flagDifferences
    )

## susy-mT2/validation_mca_MT2.txt --> this contains style
## susy-mT2/validation_MT2.txt --> contains selections
## susy-mT2/validation_plots_MT2.txt --> contains content, plot titles and binning



fi;

if [[ "$#" -ge 5 ]]; then
eval `scramv1 runtime -sh`
workingDir="$PWD"

fileA="$1"
fileB="$2"

labelA="$3"
labelB="$4"
outputFolder="$5"
isData="$6"

if [ -d $outputFolder ]; then
    echo "output folder " $outputFolder " already exists. Exiting.."
    exit;
else
    mkdir $outputFolder;
    mkdir -p $outputFolder/$labelA/mt2;
    mkdir -p $outputFolder/$labelB/mt2;
    cp $fileA $outputFolder/$labelA/mt2/mt2_tree.root
    cp $fileB $outputFolder/$labelB/mt2/mt2_tree.root
fi 



cat <<EOF > $outputFolder/inputs.txt
ttHWWdata   : $labelB : 1./0.0315 ; FillColor=ROOT.kOrange+10 , Label="$labelB"
ref_ttHWWdata+ : $labelA : 1./1.124 ; FillColor=ROOT.kAzure+2, Label="$labelA"
EOF

cd ../python/plotter/

if [[ "$isData" -eq "-data" ]]; then
    python mcPlots.py -f --tree mt2  -P $workingDir/$outputFolder  $workingDir/$outputFolder/inputs.txt susy-mT2/validation_MT2.txt susy-mT2/validation_plots_MT2.data.txt --pdir $workingDir/$outputFolder/plots -p ref_ttHWWdata,ttHWWdata -u -e --plotmode=norm --showRatio --maxRatioRange 0.65 1.35 --flagDifferences
else
    python mcPlots.py -f --tree mt2  -P $workingDir/$outputFolder  $workingDir/$outputFolder/inputs.txt susy-mT2/validation_MT2.txt susy-mT2/validation_plots_MT2.txt --pdir $workingDir/$outputFolder/plots -p ref_ttHWWdata,ttHWWdata -u -e --plotmode=norm --showRatio --maxRatioRange 0.65 1.35 --flagDifferences
fi;

cd $OLDPWD

cp $outputFolder/$labelA/mt2/mt2_tree.root $outputFolder/plots/$labelA.root
cp $outputFolder/$labelB/mt2/mt2_tree.root $outputFolder/plots/$labelB.root

fi;
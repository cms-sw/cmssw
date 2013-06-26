#/bin/bash

function USAGE ()
{
  echo ""
  echo "USAGE: "
  echo $0 "[options] run_label"
  echo ""
  echo "ATTENTION: run from your local alignment release's src/ directory where a number of certain soft-linked scripts is expected"
  echo "Options:"
  echo "-n N          number of iteration. Default is 5."
  echo "-i DIR        directory to copy the input from. Default is the current dir"
  echo "              The results are expected to be under DIR/run_label_01/ .. DIR/run_label_0N/"
  echo "-0 g0.xml     location of the initial geometry xml file (only chambers relative to ideal geometry are needed)."
  echo "-z            if provided, no tarball would be created in the end"
  echo "-w nnnnnn     The corrections mask. Defailt is 110011."
}


function COPY_CODE
{
  if [ -e $1 ]; then cp $1 $2/; return 0; fi
  for t in `echo -e "./\n$CMSSW_BASE/src/\n$CMSSW_RELEASE_BASE/src/"`
  do
    if [ -e $t/Alignment/MuonAlignmentAlgorithms/scripts/$1 ]; then cp $t/Alignment/MuonAlignmentAlgorithms/scripts/$1 $2/; return 0; fi
    if [ -e $t/Alignment/MuonAlignmentAlgorithms/python/$1 ]; then cp $t/Alignment/MuonAlignmentAlgorithms/python/$1 $2/; return 0; fi
    if [ -e $t/Alignment/MuonAlignment/python/$1 ]; then cp $t/Alignment/MuonAlignment/python/$1 $2/; return 0; fi
  done
  echo "$1 was not found in any known location"
  return 1
}


function FIND_SVG_TEMPLATES
{
  svg_templ_dir="$PWD"
  for t in `echo -e "$PWD/\n./Alignment/MuonAlignment/data/\n$CMSSW_BASE/src/Alignment/MuonAlignment/data/\n$CMSSW_RELEASE_BASE/src/Alignment/MuonAlignment/data/"`
  do
    svg_templ_dir=$t
    if [ -e "${t}disk1_template.svg" ]; then return 0; fi
  done
  echo "Could not find location of SVG templates!"
  return 1
}


# defaults
iter=5
copy_from='.'
xml0='initialMuonAlignment_March25_chambers.xml'
tgz=1
dwhich='110011'

# options
while getopts "n:i:0:zw:?" Option
do
    case $Option in
        n    ) iter=$OPTARG;;
        i    ) copy_from=$OPTARG;;
        0    ) xml0=$OPTARG;;
        z    ) tgz=0;;
        w    ) dwhich=$OPTARG;;
        ?    ) USAGE
               exit 0;;
        *    ) echo ""
               echo "Unimplemented option chosen."
               USAGE   # DEFAULT
               exit 0
    esac
done
shift $(($OPTIND - 1))

if [ "$#" != "1" ]
then
  USAGE
  exit 0
fi

svg_templ_dir="${PWD}/Alignment/MuonAlignment/data/"
FIND_SVG_TEMPLATES

echo $# $iter $copy_from $xml0 $svg_templ_dir

#exit 0

d="$1"
if [ -e $d ]; then rm -r $d; fi
mkdir $d

#./Alignment/MuonAlignmentAlgorithms/scripts/convertSQLiteXML.py $2 "$1/$1_00.xml"  --noLayers
cp ${xml0} $d/$1_00.xml

COPY_CODE diffTwoXMLs.py $d/
COPY_CODE reportVsReport.py $d/
COPY_CODE plotscripts.py $d/
COPY_CODE geometryXMLparser.py $d/
COPY_CODE mutypes.py $d/
COPY_CODE signConventions.py $d/
COPY_CODE geometryDiffVisualization.py $d/
COPY_CODE geometryDiffVisualizer.py $d/
COPY_CODE svgfig.py $d/
COPY_CODE auto_gallery.php $d/

#iter=3
#svg_templ_dir="${PWD}/Alignment/MuonAlignment/data/"

#copy_from='.'
#copy_from='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONALIGN/SWAlignment/Reference-Target/aysen/CMSSW_4_2_3_patch3/src'
#copy_from='/afs/cern.ch/cms/CAF/CMSALCA/ALCA_MUONALIGN/SWAlignment/Reference-Target/aysen/CMSSW_4_1_6/src'

for n in `seq 1 $iter`
do
  nn=`printf "%02d" $n`
  cp "${copy_from}/$1_${nn}/$1_${nn}.xml" $d/
  cp "${copy_from}/$1_${nn}/$1_${nn}_report.py" $d/
done

#exit 0

touch $d/do

for n in `seq 0 $((${iter}-1))`
do
  n1=`printf "%02d" $n`
  n2=`printf "%02d" $(($n+1))`
  echo "doing" $n1 $n2
  
  xrep="$1_${n2}.xml $1_${n1}.xml $1_${n2}_report.py"
  label="$1_${n2}-${n1}"

  echo "./diffTwoXMLs.py ${label} DT ${xrep}" >> $d/do
  echo "./diffTwoXMLs.py ${label} CSC ${xrep}" >> $d/do
  #echo "./diffTwoXMLs.py ${label} CSCE1 ${xrep}" >> $d/do
  #echo "./diffTwoXMLs.py ${label} CSCE2 ${xrep}" >> $d/do

  echo "./diffTwoXMLs.py vsReport_${label} DT ${xrep}" >> $d/do
  echo "./diffTwoXMLs.py vsReport_${label} CSC ${xrep}" >> $d/do
  #echo "./diffTwoXMLs.py vsReport_${label} CSCE1 ${xrep}" >> $d/do
  #echo "./diffTwoXMLs.py vsReport_${label} CSCE2 ${xrep}" >> $d/do

  if [ $n -ne 0 ]; then
    label="$1_x${n1}-y${n2}"
    rep="-x 'iter ${n1} ' -y  'iter ${n2} ' $1_${n1}_report.py $1_${n2}_report.py"
    echo "./reportVsReport.py -l ${label} -w ${dwhich} -s DT ${rep}" >> $d/do
    echo "./reportVsReport.py -l ${label} -w ${dwhich} -s CSC ${rep}" >> $d/do
    #echo "./reportVsReport.py -l ${label} -s CSCE1 ${rep}" >> $d/do
    #echo "./reportVsReport.py -l ${label} -s CSCE2 ${rep}" >> $d/do
  fi
done

n2=`printf "%02d" $iter`
xrep="$1_${n2}.xml $1_00.xml $1_${n2}_report.py"
label="$1_${n2}-00"
echo "./diffTwoXMLs.py ${label} DT ${xrep}" >> $d/do
echo "./diffTwoXMLs.py ${label} CSC ${xrep}" >> $d/do
#echo "./diffTwoXMLs.py ${label} CSCE1 ${xrep}" >> $d/do
#echo "./diffTwoXMLs.py ${label} CSCE2 ${xrep}" >> $d/do

xrep="${svg_templ_dir} $1_${n2}.xml $1_00.xml"
label="$1_${n2}"
echo "./geometryDiffVisualizer.py ${label} ${xrep}" >> $d/do

echo "mkdir ME; mv *_CSC* ME/; mkdir MB; mv *_DT* MB/" >> $d/do
echo "cp auto_gallery.php ME/csc_extras.php" >> $d/do
echo "cp auto_gallery.php MB/dt_extras.php" >> $d/do

cd $d/
source do
#cat do
cd -

if [ $tgz == 1 ]; then
 tar czvf "$1.tgz" $d/
fi


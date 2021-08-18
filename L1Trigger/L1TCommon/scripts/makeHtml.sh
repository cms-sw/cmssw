#! /usr/bin/env bash
# makeHtml.sh: makes simple HTML file using *.png files in direcotory



#directory=${1-`pwd`}
#  Defaults to current working directory,
#+ if not otherwise specified.
#  Equivalent to code block below.
# ----------------------------------------------------------
# ARGS=1                 # Expect one command-line argument.
#
# if [ $# -ne "$ARGS" ]  # If not 1 arg...
# then
#   directory=`pwd`      # current working directory
# else
#   directory=$1
# fi
# ----------------------------------------------------------

compDir1=$1
compDir2=$2

#compStr1=sed -e 's/.*.\///' $compDir1
#compStr2=sed -e 's/.*.\///' $compDir2

directory=`pwd`
agreeDirectory=`pwd`/inAgrement
mkdir $agreeDirectory

## convert all the .pdf files to .png
find . -type f -name '*.pdf' -print0 |
  while IFS= read -r -d '' file
      do convert -verbose "${file}" "${file%.*}.png"
  done

# move all the plots in agreement in separate directory
mv agree* $agreeDirectory

sleep 1

tempDate=`date "+%Y-%m-%d"`;

# Make webpage for plots in agreement
outputFileAgreement=inAgrement/inAgreement_$tempDate.htm
>$outputFileAgreement
cat <<EOF >>$outputFileAgreement

<HTML>

<HEAD><TITLE> Result WebPages for $tempDate</TITLE></HEAD>

<BODY link="Red">
<FONT color="Black">
<h2><A name="EB"><FONT color="Black"> Results  Web Pages for $tempDate</FONT></A><BR></h2>
<h3><A name="EB"><FONT color="Blue">Result Agreeing Histograms: </FONT></A><BR></h3>
<h3><A name="EB"><FONT color="Blue">comparing </FONT><FONT color="Black">$compDir1 (line)</FONT> <FONT color="Blue"> and </FONT> <FONT color="Black">$compDir2 (points)</FONT></A><BR></h3>

EOF

for file in `ls $agreeDirectory | grep png` ; 
#for file in "$( find $directory -name "*.png" )"   # -type l = symbolic links
do
  echo "$file"
cat <<EOF >>$outputFileAgreement
<A HREF=$file> <img height="300" src="$file"> </A>
<h3><A name="EB"><FONT color="Black">Plot: $file </FONT></A><BR></h3>
<h3><A name="EB"><FONT color="Black">----------------------</FONT></A><BR></h3>

EOF
done | sort

# Make main webpage for plots with discrepancies
outputFile=plots_$tempDate.htm
>$outputFile

cat <<EOF >>$outputFile

<HTML>

<HEAD><TITLE> Result WebPages for $tempDate</TITLE></HEAD>

<BODY link="Red">
<FONT color="Black">
<h2><A name="EB"><FONT color="Black"> Results  Web Pages for $tempDate</FONT></A><BR></h2>
<h2><A name="EB"><FONT color="Blue">Compare </FONT><FONT color="Black">$compDir1</FONT> <FONT color="Blue"> and </FONT> <FONT color="Black">$compDir2</FONT></A><BR></h3>

<A HREF=diff_menu_a_vs_menu_b.txt> diff_menu_a_vs_menu_b.txt </A>
<h3><A name="EB"><FONT color="Black">----------------------</FONT></A><BR></h3>

<A HREF=$outputFileAgreement> Plots In Agreement </A>
<h3><A name="EB"><FONT color="Blue">----------------------</FONT></A><BR></h3>

<A> Plots In Disagrement </A>
<h3><A name="EB"><FONT color="Blue">----------------------</FONT></A><BR></h3>
<h3><A name="EB"><FONT color="Blue">Result Discrepancy Histograms: </FONT></A><BR></h3>
<h3><A name="EB"><FONT color="Blue">comparing </FONT><FONT color="Black">$compDir1 (line)</FONT> <FONT color="Blue"> and </FONT> <FONT color="Black">$compDir2 (points)</FONT></A><BR></h3>
EOF

for file in `ls $directory | grep png` ; 
#for file in "$( find $directory -name "*.png" )"   # -type l = symbolic links
do
  echo "$file"
cat <<EOF >>$outputFile
<A HREF=$file> <img height="300" src="$file"> </A>
<h3><A name="EB"><FONT color="Black">Plot: $file </FONT></A><BR></h3>
<h3><A name="EB"><FONT color="Black">----------------------</FONT></A><BR></h3>

EOF

done | sort
ln -s $outputFile index.html

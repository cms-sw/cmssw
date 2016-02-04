#! /bin/bash

typeset -i mLineCnt=0

nlines=2

IN_FILE="./mickey.file"
OUT_FILE="./output.dat"

if [ $# -lt 1 ];then
echo "Please provide at least one argument, the input file"
exit
else
    IN_FILE=$1
    if [ $# -eq 2 ]; then
	OUT_FILE=$2
    fi
fi

#mLineCnt=0
mOutLine=''
while read mLine
do
  mLineCnt=${mLineCnt}+1
if [ ${mLineCnt} -eq 1 ]; then
    mOutLine=\'${mLine}
    else
    mOutLine=${mOutLine}\'\,\'${mLine}
    if [ ${mLineCnt} -eq $nlines ]; then
	echo ${mOutLine}\' >> $OUT_FILE
	mOutLine=''
	mLineCnt=0
    fi
fi

done < $IN_FILE
if [ ${mLineCnt} -ne 0 ]; then
  echo ${mOutLine}\' >> $OUT_FILE
fi

#!/bin/bash

infile=$1
iovfile=$2
nbreaks=-1
if [[ $3 != "" ]];then
	let nbreaks=$3
fi 

iovlist=()
while IFS='' read -r line || [[ -n "$line" ]]; do
    iovlist+=($line)
done < $iovfile

let niovs=${#iovlist[@]}
let lastiov=$niovs-1
for (( i=0; i<${niovs}; i++ ));
do
let ni=$i+1

core=${infile%%.dat*}
newfile=$core".dat_"${iovlist[i]}
rm -f $newfile

str=""
let nadded=1

for f in $(cat $infile)
do

if [[ $f != "#"* ]];then
	nobegin=${f#*/000/}
	noend=${nobegin%%/00000*}
	let run=${noend/\//}
	addfile=${f%*,}
	if [ $i -eq $lastiov ];then
		if [ $run -ge ${iovlist[i]} ]; then

	appendstr=""
	if [ $nadded -eq $nbreaks ];then
		appendstr="\n"
		let nadded=0
	else
		appendstr=","
	fi
	addfile=$addfile$appendstr

			str=$str$addfile
			let nadded=nadded+1
		fi
	elif [ $run -ge ${iovlist[i]} ] && [ $run -lt ${iovlist[$ni]} ]; then

	appendstr=""
	if [ $nadded -eq $nbreaks ];then
		appendstr="\n"
		let nadded=0
	else
		appendstr=","
	fi
	addfile=$addfile$appendstr

		str=$str$addfile
		let nadded=nadded+1
	fi
fi

done

str2=${str%,}
if [[ $str2 != "" ]];then
	echo -e $str2 >> $newfile
else
	$newfile" is empty. Please skip!"
fi

done

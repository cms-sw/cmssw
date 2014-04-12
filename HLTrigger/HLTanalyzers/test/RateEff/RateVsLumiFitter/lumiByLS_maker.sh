#!/bin/sh

if [ $# != 1 ]; then
	echo Usage $0 runNumber
  exit
fi


#lumiCalc2.py -r $1 lumibyls > lumiByLS_Temp.txt

echo "double lumiByLS(int runNumber, int LS) {" > lumiByLSMap_${1}.icc 
echo "typedef std::map <int, double> mapLS;"  >> lumiByLSMap_${1}.icc 
echo "mapLS lumiByLS_${1};" >> lumiByLSMap_${1}.icc 
cat lumiByLS_Temp.txt | grep "STABLE BEAMS" | sed 's/:/ /g' | awk '{if ( $19/$17 > 0.95 ) print "lumiByLS_"$2"["$4"] = "$17/23.3 ";"}' >> lumiByLSMap_${1}.icc 
echo "if (runNumber == $1) return lumiByLS_${1}[LS];" >> lumiByLSMap_${1}.icc 
echo "return 0;" >> lumiByLSMap_${1}.icc 
echo "}" >> lumiByLSMap_${1}.icc 
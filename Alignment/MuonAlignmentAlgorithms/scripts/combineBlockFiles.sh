#!/usr/bin/env bash
# Combines several files produced by groupFilesInBlocks.py into one
# Usage:
# ./combineBlockFiles.sh output.py input1.py input2.py ...

n=-1
touch tmpasdf_1
rm tmpasdf_*
for fn in $*
do
 eval n=$(($n+1))
 if [ $n == 0 ]; then continue; fi
 sed -e "s/fileNamesBlocks/fileNamesBlocks_${n}/g" $fn > tmpasdf_${n}
done

echo "writing to" $1

echo "" > $1 
for fn in tmpasdf_*
do 
  cat $fn >> $1
  echo "" >> $1
done
rm tmpasdf_*

echo "" >> $1
echo "fileNamesBlocks = []" >> $1
for i in `seq $n`
do
  echo "fileNamesBlocks.extend(fileNamesBlocks_${i})" >> $1
done

echo "print 'number of file blocks = ', len(fileNamesBlocks)" >> $1


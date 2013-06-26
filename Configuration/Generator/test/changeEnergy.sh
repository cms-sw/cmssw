#! /bin/bash

# script to change energy in all files using comEnergy
# $1 = old energy (e.g. 10000.) $2 = new energy (e.g. 7000.), $3 string for new files (e.g. 7TeV)

# move the script into the directory where you wnat to add files with new comEnergy
# usage example: ./changeEnergy 10000. 7000. 7TeV

# F. Stoeckli

numFiles=$(grep $1 * | grep -c comEnergy)

iter=1
while [ $iter -le $numFiles ]
  do
  fileName=$(grep $1 * | grep comEnergy | sed -n $iter'p' | sed 's/:/\n/g' | sed -n 1'p')
  newFile=$(echo $fileName | sed 's/\_cf/\_'$3'\_cf/g')
  echo "From file "$fileName" creaeting new file "$newFile"."
  sed 's/comEnergy = cms.double('$1$'/comEnergy = cms.double('$2'/g' $fileName > $newFile
  echo "Diff from file "$fileName" and new file "$newFile":"
  diff $fileName $newFile
  echo "--------------------------------------------------------------------"
  iter=`expr $iter + 1`
done

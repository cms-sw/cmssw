#!/bin/bash

echo "Prepare lumi files"

sed -i 's/|//g' `find . -name \*.dat`
sed -i 's/=//g' `find . -name \*.dat`
sed -i 's/L1T_//g' `find . -name \*.dat`
sed -i 's/-//g' `find . -name \*.dat`
sed -i 's/Run//g' `find . -name \*.dat`
sed -i 's/LS//g' `find . -name \*.dat`
sed -i 's/Delivered//g' `find . -name \*.dat`
sed -i 's/Recorded//g' `find . -name \*.dat`
sed -i 's/(\/μb)//g' `find . -name \*.dat`

cat `find . -name \*.dat`>> lumis/lumis.dat

cd lumis

cp $L1RATES_DIR/toolbox/mkTree.C .

root .x mkTree.C

rm mkTree.C

rm *.dat

cd ..





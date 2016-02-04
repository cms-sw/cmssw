cmsRun validateField.cfg > ! output.txt
cat newtable.txt | cut -d' ' -f 2,3,4 > ! allpoints
grep Disc output.txt | cut -f 7 -d' ' | tr '(,)' ' ' > ! outliers
gnuplot < plot3d.gnuplot

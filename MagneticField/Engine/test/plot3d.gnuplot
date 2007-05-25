set terminal png
set xlabel "x coordinate (cm)"
set ylabel "y coordinate (cm)"
set zlabel "z coordinate (cm)"
set output "viewXY.png"
set view 0,90
splot 'allpoints','outliers'
set output "view3D.png"
set view 30,60
splot 'allpoints','outliers'
set output "viewXZ.png" 
set view 90,0
splot 'allpoints','outliers'


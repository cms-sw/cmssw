load "myStyle.h"
unset key

log10 = log(10.)
e(x) = exp(log10 * x)

set missing "0"

set style line 1 lt 7 lw 3 pt 4

set ylabel "Fake track rate"

set yrange [0:0.5]
set size xsize,ysize
set origin 0,0

######################################
set output "eps/fakeRateEta.eps"
set xlabel "{/Symbol h}"
set xrange [-3:3]
set mxtics 4
set mytics 5
plot "out/fakeRate_Eta.dat" w e ls 1
set key off

######################################
set output "eps/fakeRatePt.eps"
set xlabel "p_T [GeV/c]"

set multiplot
set xrange [0:2]

plot "out/fakeRate_Pt.dat"  w e ls 1
unset label

q = 0.7
set origin (1-q-0.05)*xsize,(1-q-0.05)*ysize
set size   q*xsize,q*ysize
set xlabel
set ylabel
set auto y
set yrange [1e-4:*]
set log y; set format y "10^{%T}"
# replot
set auto x
unset log y; set format y

set nomultiplot

######################################
set output "eps/fakeRateLogPt.eps"
set xlabel "p_T [GeV/c]"
set ylabel "Fake track rate"

set yrange [0:0.5]

set origin 0,0
set size xsize,ysize
set multiplot
set xrange [0.1:10]
set log x

plot "out/fakeRate_LogPt.dat" u (e($1)):2:3 w e ls 1
unset label

set origin (1-q-0.05)*xsize,(1-q-0.05)*ysize
set size   q*xsize,q*ysize
set xlabel
set ylabel
set auto y
set yrange [1e-3:*]
set log y; set format y "10^{%T}"
# replot
unset log y; set format y
set auto x
unset log x

set nomultiplot

set origin 0,0
set size xsize,ysize

set missing "?"
#set log cb ; set format cb "10^{%T}"
#set cbrange [1e-4:*]

set auto
set pm3d map corners2color c1 ; set size 1.5*1,1.5*1.2
set xlabel "{/Symbol h}"
set ylabel "p_T [GeV/c]"
set output "eps/fakeRateEtaPt.eps"
splot "out/fakeRate_EtaPt.dat"


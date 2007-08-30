load "myStyle.h"

set xlabel "{/Symbol h}"
set ylabel "1/N_{ev} dN_{ch}/d{/Symbol h}"

set output "../eps/ppMulti/dNdeta.eps"
set style data his
set key width -4
set yrange [0:6.5]
plot "../out/ppMulti/sia.dat" u 1:($2/1000.) t "simTracks" ls 1, \
     "../out/ppMulti/sim.dat" u 1:($2/1000.) t "simTracks,  1 rec vtx" ls 2, \
     "../out/ppMulti/rec.dat" u 1:($2/1000.) t "acc recHits, 1 rec vtx" ls 3
set auto y

set key nobox
set output "../eps/ppMulti/dNdeta_rat.eps"
set ylabel "Ratio of accepted recHits / simTracks]"
plot "../out/ppMulti/rat.dat" ls 9

set output "../eps/ppMulti/dNdeta_irat.eps"
set ylabel "Ratio of simTracks / accepted recHits]"
plot [][0.6:1.2] "../out/ppMulti/rat.dat" u 1:(1/$2) ls 9

! epstopdf ../eps/ppMulti/dNdeta.eps
! epstopdf ../eps/ppMulti/dNdeta_rat.eps
! epstopdf ../eps/ppMulti/dNdeta_irat.eps

set size 1,1.4

set key noauto nobox

w(eloss) = eloss*1e-3
f(eta) = 21 * cosh(eta) 
g(eta) = 21 * cosh(eta) - 11
#g(eta) = 21 * cosh(eta) - 5.5

set pm3d explicit map corners2color c1
set parametric

set cbrange [0:800]

set xrange [-3:3]
set yrange [0:250]

set xlabel "{/Symbol h}"
set ylabel "Deposited charge (10^{3} e^-)"

set output "../eps/ppMulti/pri_cosh.eps"
set title "Primary recHits"
splot "../out/ppMulti/pri.dat" u 1:(w($2)):3 w pm3d, u,f(u),0 w l 9

set output "../eps/ppMulti/pri.eps"
set title "Primary recHits"
splot "../out/ppMulti/pri.dat" u 1:(w($2)):3 w pm3d, u,g(u),0 w l 2 

set output "../eps/ppMulti/bac.eps"
set title "Secondary recHits"
splot "../out/ppMulti/bac.dat" u 1:(w($2)):3 w pm3d, u,g(u),0 w l 2

set output "../eps/ppMulti/loo.eps"
set title "Looper recHits"
splot "../out/ppMulti/loo.dat" u 1:(w($2)):3 w pm3d, u,g(u),0 w l 2

set output "../eps/ppMulti/all_bare.eps"
set title "All recHits"
splot "../out/ppMulti/all.dat" u 1:(w($2)):3 w pm3d

set output "../eps/ppMulti/all.eps"
set title "All recHits"
splot "../out/ppMulti/all.dat" u 1:(w($2)):3 w pm3d, u,g(u),0 w l 2

! epstopdf ../eps/ppMulti/pri_cosh.eps
! epstopdf ../eps/ppMulti/pri.eps
! epstopdf ../eps/ppMulti/bac.eps
! epstopdf ../eps/ppMulti/loo.eps
! epstopdf ../eps/ppMulti/all_bare.eps
! epstopdf ../eps/ppMulti/all.eps

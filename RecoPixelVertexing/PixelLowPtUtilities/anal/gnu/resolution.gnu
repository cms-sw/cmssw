load "myStyle.h"

log10 = log(10.)
e(x) = exp(log10 * x)

set style line 1 lt 1 lw 3 pt 6
set style line 2 lt 2 lw 3 pt 8
set style line 3 lt 3 lw 3 pt 4
set style line 4 lt 7 lw 3 pt 6

set mxtics 4
set mytics 5

set xrange [0:2]

a1=1; b1=1; c1=1
a2=1; b2=1; c2=1
a3=1; b3=1; c3=1

#################################
set output "eps/ptBias.eps"
set xlabel "p_{Tsim} [GeV/c]"
set ylabel "p_{Trec} / p_{Tsim}"
set yrange [0.90:1.05]

f1(x) = a1+b1*exp(-x/c1)
f2(x) = a2+b2*exp(-x/c2)
f3(x) = a3+b3*exp(-x/c3)
if(f) fit f1(x) "out/ptBias_pion.dat" u 1:($2/$1):($3/$1) via a1,b1,c1
if(0) fit f2(x) "out/ptBias_kaon.dat" u 1:($2/$1):($3/$1) via a2,b2,c2
if(0) fit f3(x) "out/ptBias_prot.dat" u 1:($2/$1):($3/$1) via a3,b3,c3

sw(x) = (x != 0. ? 2e-2/x**2 : 1e-10)
set parametric; set trange [0.075:2]
plot t,1 w l ls 4, \
 "out/ptBias_pion.dat" u 1:($2/$1):($3/$1) \
   t "pion" w e ls 1, t,f1(t) w l 1, \
 "out/ptBias_kaon.dat" u 1:($2/$1):($3/$1) \
   t "kaon" w e ls 2, \
 "out/ptBias_prot.dat" u 1:($2/$1):($3/$1) \
   t "prot" w e ls 3
unset parametric

set output "eps/ptBiar.eps"
set xlabel "p_{Trec} [GeV/c]"
set ylabel "p_{Tsim} / p_{Trec}"
set yrange [0.95:1.1]

b1=-b1; b2=-b2; b3=-b3
if(f) fit f1(x) "out/ptBiar_pion.dat" u 1:($2/$1):($3/$1) via a1,b1,c1
if(0) fit f2(x) "out/ptBiar_kaon.dat" u 1:($2/$1):($3/$1) via a2,b2,c2
if(0) fit f3(x) "out/ptBiar_prot.dat" u 1:($2/$1):($3/$1) via a3,b3,c3
   
set parametric; set trange [0.075:2]
plot t,1 w l ls 4, \
 "out/ptBiar_pion.dat" u 1:($2/$1):($3/$1) \
   t "pion" w e ls 1, t,f1(t) w l 1, \
 "out/ptBiar_kaon.dat" u 1:($2/$1):($3/$1) \
   t "kaon" w e ls 2, \
 "out/ptBiar_prot.dat" u 1:($2/$1):($3/$1) \
   t "prot" w e ls 3
unset parametric

set auto y

#################################
set output "eps/ptReso.eps"
set xlabel "p_{Tsim} [GeV/c]"
set ylabel "Relative resolution of p_{Trec}"
set yrange [0:0.2]

f1(x) = abs(a1)+abs(b1)*exp(-log(x)/c1)+abs(d1)*x
f2(x) = a2+b2*exp(-x/c2)+d2*x
f3(x) = a3+b3*exp(-x/c3)+d3*x

a1 = 0.05; b1 = 0.1; c1 = 0.2; d1 = 0.01
a2 = 0.02; b2 = 0.1; c2 = 0.1; d2 = 0.01
a3 = 0.02; b3 = 0.1; c3 = 0.1; d3 = 0.01

set missing "0"
d1 = 0.
if(f) fit f1(x) "out/ptReso_pion.dat" u 1:($2/$1):(sqrt($3/$1)) via a1,b1,c1#,d1
if(0) fit f2(x) "out/ptReso_kaon.dat" u 1:($2/$1):($3/$1) via a2,b2,c2,d2
if(0) fit f3(x) "out/ptReso_prot.dat" u 1:($2/$1):($3/$1) via a3,b3,c3,d3

sw(x) = (x != 0. ? 1e-1/x**2 : 1e-10)
set parametric; set trange [0.075:2]
plot \
 "out/ptReso_pion.dat" u 1:($2/$1):($3/$1) \
   t "pion" w e ls 1, t,f1(t) w l ls 1, \
 "out/ptReso_kaon.dat" u 1:($2/$1):($3/$1) \
   t "kaon" w e ls 2, \
 "out/ptReso_prot.dat" u 1:($2/$1):($3/$1) \
   t "prot" w e ls 3
unset parametric

set output "eps/ptLogReso.eps"
set xlabel "p_{Tsim} [GeV/c]"
set ylabel "Relative resolution of p_{Trec}"
set yrange [0:0.2]

set parametric; set trange [0.1:10]
set auto x
set xrange [0.1:10]
set log x
plot \
 "out/ptLogReso_pion.dat" u (e($1)):($2/e($1)):($3/e($1)) \
   t "pion" w e ls 1, t,f1(t) w l ls 1, \
 "out/ptLogReso_kaon.dat" u (e($1)):($2/e($1)):($3/e($1)) \
   t "kaon" w e ls 2, \
 "out/ptLogReso_prot.dat" u (e($1)):($2/e($1)):($3/e($1)) \
   t "prot" w e ls 3
unset log x
unset parametric
set xrange [0:2]

set output "eps/ptResr.eps"
set xlabel "p_{Trec} [GeV/c]"
set ylabel "Relative resolution of p_{Tsim}"

set missing "0"
if(f) fit f1(x) "out/ptResr_pion.dat" u 1:($2/$1):(1) via a1,b1,c1#,d1
if(0) fit f2(x) "out/ptResr_kaon.dat" u 1:($2/$1):($3/$1) via a2,b2,c2,d2
if(0) fit f3(x) "out/ptResr_prot.dat" u 1:($2/$1):($3/$1) via a3,b3,c3,d3

sw(x) = (x != 0. ? 1e-1/x**2 : 1e-10)
set parametric; set trange [0.075:2]
plot \
 "out/ptResr_pion.dat" u 1:($2/$1):($3/$1) \
   t "pion" w e ls 1, t,f1(t) w l ls 1, \
 "out/ptResr_kaon.dat" u 1:($2/$1):($3/$1) \
   t "kaon" w e ls 2, \
 "out/ptResr_prot.dat" u 1:($2/$1):($3/$1) \
   t "prot" w e ls 3
unset parametric

! rm fit.log

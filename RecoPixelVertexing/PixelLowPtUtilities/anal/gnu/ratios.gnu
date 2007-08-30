load "myStyle.h"

set missing "0"

set style line 1 lt 1 lw 3 pt 6
set style line 2 lt 2 lw 3 pt 8
set style line 3 lt 3 lw 3 pt 4
set style line 4 lt 7 lw 3 pt 6

set mxtics 4
set mytics 5

log10 = log(10.)
e(x) = exp(log10 * x)

set key bottom right

a = 1.
f(x) = a

if(f) fit [-1:1] f(x) "out/geomAccep_Eta_pion.dat" via a
apion = a
if(f) fit [-1:1] f(x) "out/geomAccep_Eta_kaon.dat" via a
akaon = a
if(f) fit [-1:1] f(x) "out/geomAccep_Eta_prot.dat" via a
aprot = a

if(f) fit [0.5:1] f(x) "out/geomAccep_Pt_pion.dat" via a
bpion = a
if(f) fit [0.5:1] f(x) "out/geomAccep_Pt_kaon.dat" via a
bkaon = a
if(f) fit [0.5:1] f(x) "out/geomAccep_Pt_prot.dat" via a
bprot = a

if(f) fit [-1:1] f(x) "out/algoEffic_Eta_pion.dat" via a
cpion = a
if(f) fit [-1:1] f(x) "out/algoEffic_Eta_kaon.dat" via a
ckaon = a
if(f) fit [-1:1] f(x) "out/algoEffic_Eta_prot.dat" via a
cprot = a

if(f) fit [0.5:1] f(x) "out/algoEffic_Pt_pion.dat" via a
dpion = a
if(f) fit [0.5:1] f(x) "out/algoEffic_Pt_kaon.dat" via a
dkaon = a
if(f) fit [0.5:1] f(x) "out/algoEffic_Pt_prot.dat" via a
dprot = a

print " Acc"
print "  eta ",apion,akaon,aprot
print "   pt ",bpion,bkaon,bprot

print " Eff"
print "  eta ",cpion,ckaon,cprot
print "   pt ",dpion,dkaon,dprot

! rm fit.log

####################################
set ylabel "Geometrical acceptance"
set yrange [0:1]

set output "eps/geomAccepEta.eps"
set xlabel "{/Symbol h}"
set key off
plot [-3:3] \
 "out/geomAccep_Eta_pion.dat" t "pion" w errorlines ls 1, \
 "out/geomAccep_Eta_kaon.dat" t "kaon" w errorlines ls 2, \
 "out/geomAccep_Eta_prot.dat" t "prot" w errorlines ls 3
set key on

set output "eps/geomAccepPt.eps"
set xlabel "p_T [GeV/c]"
plot [0:2] \
 "out/geomAccep_Pt_pion.dat" t "pion" w errorlines ls 1, \
 "out/geomAccep_Pt_kaon.dat" t "kaon" w errorlines ls 2, \
 "out/geomAccep_Pt_prot.dat" t "prot" w errorlines ls 3

set output "eps/geomAccepLogPt.eps"
set xlabel "p_T [GeV/c]"
set log x
plot [0.1:10] \
 "out/geomAccep_LogPt_pion.dat" u (e($1)):2:3 t "pion" w errorlines ls 1, \
 "out/geomAccep_LogPt_kaon.dat" u (e($1)):2:3 t "kaon" w errorlines ls 2, \
 "out/geomAccep_LogPt_prot.dat" u (e($1)):2:3 t "prot" w errorlines ls 3
unset log x

set auto y

set missing "?"

set pm3d map corners2color c1; set size 1.5*1,1.5*1.2
set xlabel "{/Symbol h}"
set ylabel "p_T [GeV/c]"
set output "eps/geomAccepEtaPt_pion.eps"
splot "out/geomAccep_EtaPt_pion.dat" t "pion"
set output "eps/geomAccepEtaPt_kaon.eps"
splot "out/geomAccep_EtaPt_kaon.dat" t "kaon"
set output "eps/geomAccepEtaPt_prot.eps"
splot "out/geomAccep_EtaPt_prot.dat" t "prot"
unset pm3d; set view ,,1,1; set size 1,1.2

set missing "0"

####################################
set ylabel "Algorithmic efficiency"
set yrange [0:1]

set xlabel "{/Symbol h}"
set key off

set output "eps/algoEfficEta.eps"
plot [-3:3] \
 "out/algoEffic_Eta_pion.dat" t "pion" w e ls 1, \
 "out/algoEffic_Eta_kaon.dat" t "kaon" w e ls 2, \
 "out/algoEffic_Eta_prot.dat" t "prot" w e ls 3

set key on
unset label

set xlabel "p_T [GeV/c]"
set output "eps/algoEfficPt.eps"
plot [0:2] \
 "out/algoEffic_Pt_pion.dat" t "pion" w e ls 1, \
 "out/algoEffic_Pt_kaon.dat" t "kaon" w e ls 2, \
 "out/algoEffic_Pt_prot.dat" t "prot" w e ls 3

set xlabel "p_T [GeV/c]"
set output "eps/algoEfficLogPt.eps"
set log x
plot [0.1:10] \
 "out/algoEffic_LogPt_pion.dat" u (e($1)):2:3 t "pion" w e ls 1, \
 "out/algoEffic_LogPt_kaon.dat" u (e($1)):2:3 t "kaon" w e ls 2, \
 "out/algoEffic_LogPt_prot.dat" u (e($1)):2:3 t "prot" w e ls 3
unset log

unset label

set auto y

set missing "?"

set pm3d map corners2color c1; set size 1.5*1,1.5*1.2
set xlabel "{/Symbol h}"
set ylabel "p_T [GeV/c]"
set output "eps/algoEfficEtaPt_pion.eps"
splot "out/algoEffic_EtaPt_pion.dat" t "pion"
set output "eps/algoEfficEtaPt_kaon.eps"
splot "out/algoEffic_EtaPt_kaon.dat" t "kaon"
set output "eps/algoEfficEtaPt_prot.eps"
splot "out/algoEffic_EtaPt_prot.dat" t "prot"

unset pm3d; set view ,,1,1; set size 1,1.2

set missing "0"

####################################
set ylabel "Multiple counting"
set yrange [0:0.5]

set key top right

set output "eps/multCountEta.eps"
set xlabel "{/Symbol h}"
set key off
plot [-3:3] \
 "out/multCount_Eta_pion.dat" w e ls 1

set output "eps/multCountPt.eps"
set xlabel "p_T [GeV/c]"
plot [0:2] \
 "out/multCount_Pt_pion.dat" w e ls 1
unset label

set output "eps/multCountLogPt.eps"
set xlabel "p_T [GeV/c]"
set log x
plot [0.1:10] \
 "out/multCount_LogPt_pion.dat" u (e($1)):2:3 w e ls 1
unset log x
unset label

set key on

set auto y
set cbrange [0:0.5]

set missing "?"

set pm3d map corners2color c1; set size 1.5*1,1.5*1.2
set xlabel "{/Symbol h}"
set ylabel "p_T [GeV/c]"
set output "eps/multCountEtaPt_pion.eps"
splot "out/multCount_EtaPt_pion.dat" t "pion"
set output "eps/multCountEtaPt_kaon.eps"
splot "out/multCount_EtaPt_kaon.dat" t "kaon"
set output "eps/multCountEtaPt_prot.eps"
splot "out/multCount_EtaPt_prot.dat" t "prot"
unset pm3d; set view ,,1,1; set size 1,1.2

set missing "?"

! rm fit.log

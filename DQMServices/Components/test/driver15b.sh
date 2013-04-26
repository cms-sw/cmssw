#!/bin/bash

eval `scramv1 r -sh`

function doReport {
    igprof-analyse -g -d -v -p -r $1   -s IgProf.$2.gz   | sqlite3 igreport_$2_$1.sql3
    igprof-analyse -g -d -v -p -r $1   -s IgProfRECO.$2.gz   | sqlite3 igreportRECO_$2_$1.sql3
}

function doCumulative {
    igprof-analyse -g -d -v -p -r $1   -s IgProf$2.gz   | sqlite3 igreport_$2_$1.sql3
    igprof-analyse -g -d -v -p -r $1   -s IgProfRECO$2.gz   | sqlite3 igreportRECO_$2_$1.sql3
}

doReport MEM_LIVE 1
doReport MEM_LIVE 26
doReport MEM_LIVE 51
doReport MEM_LIVE 76
doReport MEM_TOTAL 1
doReport MEM_TOTAL 26
doReport MEM_TOTAL 51
doReport MEM_TOTAL 76
doCumulative MEM_LIVE Cumulative_100
doCumulative MEM_TOT Cumulative_100


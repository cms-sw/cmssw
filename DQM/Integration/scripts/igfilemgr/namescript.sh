#! /bin/bash
ls Hi*.ig |  awk '{i++;print("mv -f " $0 " MWGR_W1_052009_" i ".ig")}' > renameig.sh 
source renameig.sh

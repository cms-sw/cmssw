#!/bin/bash
if [ $# -ne 1 ]
then
    echo "Command required. Nothing done."
else
	runs=( 	
			315512 # DONE
			315713 # DONE
			315840 # DONE
			316114 # DONE
			316199 # DONE
			316240 # DONE
			316505 # DONE
			316666 # DONE
			316758 # DONE
			# 316985
			317182 # DONE
			317320 # DONE
			317435 # DONE
			317527 # DONE
			317641 # DONE
			317696 # DONE
			# TS1
			319337 # DONE
			319450 # DONE
			319579 # DONE
			319756 # DONE
			319991 # DONE
			320038 # DONE
			320804 # DONE
			320917 # DONE
			321051 # DONE
			321149 # DONE
			321233 # DONE
			321396 # DONE
			321457 # DONE
			321755 # DONE
			321831 # DONE
			321909 # DONE
			321988 # DONE
			322106 # DONE
			322252 # DONE
			322356 # DONE
			322431 # DONE
			322625 # DONE
			# # TS2
			323487 # DONE
			323702 # DONE
			323790 # DONE
			323940 # DONE
			324077 # DONE
			324293 # DONE
			324747 # DONE
			324791 # DONE
			324841 # DONE
			)

	export CMSSW_BASE=`readlink -f ../../..`
	for run in ${runs[@]}
	do
		echo "$1 $run"
		eval $1 $run
	done
fi
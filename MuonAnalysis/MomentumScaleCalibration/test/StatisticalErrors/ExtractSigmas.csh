#!/bin/csh

if( $# != 2 ) then
    echo "Error: first and last line requested"
    exit
endif

if( $1 >= $2 ) then
    echo "Error: second parameter must be > than the first"
    exit
endif

set ii=$1
set numpar=0

if( -f Values.txt ) rm Values.txt
if( -f Sigmas.txt ) rm Sigmas.txt
touch Sigmas.txt

while( $ii <= $2 )
    ./TakeParameterFromBatch.sh $ii
    if( ! -f Values.txt ) then
	echo "Error: file Values.txt not found for line "$ii"."
	exit
    endif
    root -l -b -q MakePlot.C >&! OutputFit_param_${numpar}.txt
    grep sigma_final OutputFit_param_${numpar}.txt | awk '{print $2}' >> Sigmas.txt
    if( -f plot_param_x.gif ) mv plot_param_x.gif plot_param_${numpar}.gif
    rm Values.txt
    @ ii++
    @ numpar++
end

exit


#!/bin/bash
i=0
while read line
do
    name=$line
		let i=i+1
		if [ $(($i%1)) -eq 0 ]
			then
#			echo "'$name'" >> ALCARECOTkAlCosmicsCTF0T.dat_PromptReco 
#                        echo "'$name'" >> ALCARECOTkAlCosmicsCTF0T.dat_StreamExpress
                        echo "'$name'" >> ALCARECOTkAlMinBias.dat_2016B_Prompt_v1
		else
#			echo "'$name'" | tr "\n" "," >> ALCARECOTkAlCosmicsCTF0T.dat_PromptReco 
#                        echo "'$name'" | tr "\n" "," >> ALCARECOTkAlCosmicsCTF0T.dat_StreamExpress
                        echo "'$name'" | tr "\n" "," >> ALCARECOTkAlMinBias.dat_2016B_Prompt_v1
		fi
		#if [ $i -eq 51 ]
		#	then break
		#fi
done < $1

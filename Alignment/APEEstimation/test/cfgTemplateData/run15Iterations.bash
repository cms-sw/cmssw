#!/bin/bash

COUNTER="0"

# if data samples are larger switch to longer sleep times
while [ $COUNTER -lt 15 ]; do
	bash ../createStep1.bash $COUNTER
	bash ../startStep1.bash
	sleep 15m 
	bash ../createStep2.bash $COUNTER
	bash ../startStep2.bash
	let COUNTER=COUNTER+1
done

bash ../createStep1.bash 15 True
bash ../startStep1.bash
sleep 15m
bash ../createStep2.bash 15
bash ../startStep2.bash
	

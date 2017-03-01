#!/bin/bash

COUNTER="0"

while [ $COUNTER -lt 10 ]; do
	bash ../createStep1.bash $COUNTER
	bash ../startStep1.bash
	sleep 25m
	bash ../createStep2.bash $COUNTER
	bash ../startStep2.bash
	let COUNTER=COUNTER+1
done

bash ../createStep1.bash 10 True
bash ../startStep1.bash
sleep 25m
bash ../createStep2.bash 10
bash ../startStep2.bash
	

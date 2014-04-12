#!/bin/bash

export workdir=$1

    if [ ! -d "$workdir/rootfiles" ]; then
	mkdir $workdir/rootfiles;
    fi

    if [ ! -d "$workdir/plots" ]; then
	mkdir $workdir/plots;
	mkdir $workdir/plots/TIB;
	mkdir $workdir/plots/TOB;
	mkdir $workdir/plots/TID;
	mkdir $workdir/plots/TEC;

	for i in {1..4}; do
	    mkdir $workdir/plots/TIB/Layer$i;
	    mkdir $workdir/plots/TIB/Layer$i/Trends;
	done

	for i in {1..6}; do
	    mkdir $workdir/plots/TOB/Layer$i;
	    mkdir $workdir/plots/TOB/Layer$i/Trends;
	done

	for i in {1..2}; do
	    mkdir $workdir/plots/TID/Side$i;
	    for j in {1..3}; do
		mkdir $workdir/plots/TID/Side$i/Disk$j;
		mkdir $workdir/plots/TID/Side$i/Disk$j/Trends;
	    done
	done

	for i in {1..2}; do
	    mkdir $workdir/plots/TEC/Side$i;
	    for j in {1..9}; do
		mkdir $workdir/plots/TEC/Side$i/Disk$j;
		mkdir $workdir/plots/TEC/Side$i/Disk$j/Trends;
	    done
	done

	mkdir $workdir/plots/Trends
	mkdir $workdir/plots/TIB/Trends;
	mkdir $workdir/plots/TOB/Trends;
	mkdir $workdir/plots/TID/Side1/Trends;
	mkdir $workdir/plots/TID/Side2/Trends;
	mkdir $workdir/plots/TEC/Side1/Trends;
	mkdir $workdir/plots/TEC/Side2/Trends;
	mkdir $workdir/plots/Summary

    fi

for i in {1..4}; do
    for Plot in `ls *.png | grep TIBLayer$i`; do
	mv $Plot $workdir/plots/TIB/Layer$i/Trends;
    done
done

for i in {1..6}; do
    for Plot in `ls *.png | grep TOBLayer$i`; do
	mv $Plot $workdir/plots/TOB/Layer$i/Trends;
    done
done

for i in {1..3}; do
    for Plot in `ls *.png | grep TID-Disk$i`; do
	mv $Plot $workdir/plots/TID/Side1/Disk$i/Trends;
    done
    for Plot in `ls *.png | grep TID+Disk$i`; do
	mv $Plot $workdir/plots/TID/Side2/Disk$i/Trends;
    done
done

for i in {1..9}; do
    for Plot in `ls *.png | grep TEC-Disk$i`; do
	mv $Plot $workdir/plots/TEC/Side1/Disk$i/Trends;
    done
    for Plot in `ls *.png | grep TEC+Disk$i`; do
	mv $Plot $workdir/plots/TEC/Side2/Disk$i/Trends;
    done
done

for Plot in `ls *.png | grep TIB`; do
    mv $Plot $workdir/plots/TIB/Trends;
done

for Plot in `ls *.png | grep TOB`; do
    mv $Plot $workdir/plots/TOB/Trends;
done

for Plot in `ls *.png | grep TID-`; do
    mv $Plot $workdir/plots/TID/Side1/Trends;
done

for Plot in `ls *.png | grep TID+`; do
    mv $Plot $workdir/plots/TID/Side2/Trends;
done

for Plot in `ls *.png | grep TEC-`; do
    mv $Plot $workdir/plots/TEC/Side1/Trends;
done

for Plot in `ls *.png | grep TEC+`; do
    mv $Plot $workdir/plots/TEC/Side2/Trends;
done

for Plot in `ls *.png | grep Tracker`; do
    mv $Plot $workdir/plots/Trends;
done

mv TrackerSummary.root $workdir/rootfiles;
rm -f TrackerPlots.root;


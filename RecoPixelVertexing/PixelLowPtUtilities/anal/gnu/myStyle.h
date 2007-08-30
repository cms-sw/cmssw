# Set terminal
set term post eps enh color dashed "Helvetica" 25

set output "test.eps"
test

# Set size
xsize = 1
ysize = 1.2
set size xsize,ysize

# Set various
set pointsize 1.5
set ticscale 2 1
set ylabel 1,0

# Set lines
set style line 1 lt 1 lw 5
set style line 2 lt 2 lw 5
set style line 3 lt 3 lw 5
set style line 4 lt 4 lw 5
set style line 5 lt 5 lw 5
set style line 6 lt 6 lw 5
set style line 7 lt 7 lw 5
set style line 8 lt 8 lw 5
set style line 9 lt 9 lw 5

# Make errorbars small
set bar small

# Set key
set key top right Left reverse samplen 2 noauto box

# Set palette "rainbow"
#set palette defined ( 0 "blue", 3 "green", 6 "yellow", 10 "red" )
#set palette gray negative

# Fit
f = 0

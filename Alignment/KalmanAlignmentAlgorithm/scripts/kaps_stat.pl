#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_stat.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Display local kaps database
#
#
#  Usage: kaps_stat.pl
#
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;

system "kaps_update.pl >| /dev/null";
read_db();
print_memdb();




sub set_sdir() {
    $called = "$0";
    # de-reference symbolic links up to depth 3
    if (-l $called) { 
	$called = readlink $called;
	if (-l $called) { 
	    $called = readlink $called;
	    if (-l $called) { $called = readlink $called;}
	}
    }
    # find the path
    if ($called =~ m/(\/.+\/)/) {
	$thePath = $1;
    }
    else {
	$libName = "";
	exit;
    }
    # check whether the library exists
    $libName = $thePath . "kapslib/Kapslib.pm";

    unless (-r "$libName") {
	$libName = "";
    }
    $theLibName = $libName;
    # print "theLibName is $theLibName\n";
}

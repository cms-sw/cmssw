#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     28-Nov-2007
#     A. Parenti, DESY Hamburg    16-Apr-2008
#     $Revision: 1.2 $
#     $Date: 2008/04/17 16:38:47 $
#
#  Display local mps database
#
#
#  Usage: mps_stat.pl
#
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

system "mps_update.pl >| /dev/null";
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
    $libName = $thePath . "mpslib/Mpslib.pm";

    unless (-r "$libName") {
	$libName = "";
    }
    $theLibName = $libName;
    # print "theLibName is $theLibName\n";
}

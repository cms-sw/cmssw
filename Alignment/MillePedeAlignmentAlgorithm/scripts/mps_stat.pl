#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     28-Nov-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.14 $
#     $Date: 2008/03/25 16:15:57 $
#
#  Display local mps database
#
#
#  Usage: mps_stat.pl
#
#

use lib './mpslib';
use Mpslib;

$sdir = get_sdir();
if ($sdir eq ".") {
  print "Error from mps_stat.pl: configuration file not found or unreadable\n";
  exit;
}
system "$sdir/mps_update.pl >| /dev/null";
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

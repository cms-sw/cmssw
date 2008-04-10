#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     28-Nov-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.1 $
#     $Date: 2008/04/10 16:10:12 $
#
#  Setup internal paths correctly for mps scripts in a local directory.
#  Optionally, create this local directory first & copy mps scripts there. The 
#  source directory is the one from which mps_install.pl is called.
#  
#
#  Usage:
#
#  mps_install.pl <directory>

$targetDir = "undefined";
# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $targetDir = $arg;
    }
  }
}

# first find out from which directory mps_install.pl has been called
$called = "$0";

# de-reference symbolic links up to depth 3
if (-l $called) { 
    $called = readlink $called;
    if (-l $called) { 
	$called = readlink $called;
	if (-l $called) { $called = readlink $called;}
    }
}

# now extract the path
$nn = ($called =~ m/(.+)\/mps_install\.pl/);
$thePath = $1;

if ($nn != 1) {
    print "mps_install.pl: cannot extract the path\n";
    exit 1;
}
print "Path of source directory is $thePath/\n";

$workPath = $thePath;


# if a directory parameter has been given, first copy files to this new directory
if ($targetDir ne "undefined") {
    print "Path of destination directory is $targetDir/\n";
    # check if already exists
    if (-e $targetDir) {
	print "Error: file or directory $targetDir already existing\n";
	exit 2;
    }

    # check if path is absolute
    unless ($targetDir =~ m/^\//) {
	print "Error: must specify absolute path for destination directory\n";
	exit 3;
    }

    # create directory
    system "mkdir $targetDir";

    unless ((-d $targetDir) && (-w $targetDir)) {
	print "Error: creation of directory $targetDir failed\n";
	exit 4;
    }
    system "mkdir $targetDir/mpslib";

    # now copy the files
    @NAMES = ("mps_auto.pl", "mps_fire.pl", "mps_script.pl", "mps_split.pl", "mps_update.pl", 
	      "mps_check.pl", "mps_install.pl", "mps_merge.pl", "mps_save.pl", "mps_setup.pl",
	      "mps_fetch.pl", "mps_kill.pl", "mps_retry.pl", "mps_scriptm.pl", "mps_splice.pl",
	      "mps_stat.pl",
	      "mps_runMille_template.sh", "mps_runPede_rfcp_template.sh",
	      "mps_template.cfg");
    $libName = "mpslib/Mpslib.pm";

    while (@NAMES) {
	$theName = shift @NAMES;
	system "cp $thePath/$theName $targetDir/$theName";
    }
    print "cp $thePath/$libName $targetDir/$libName\n";
    system "cp $thePath/$libName $targetDir/$libName";
    $workPath = $targetDir;
}

# now adjust path names. This only needs to be done for scripts using the library.

print "mps_install will update path directives in directory $workPath in 5s ";
system "sleep 5";

# $| = 1;
#$reply = <STDIN>;
# chomp $reply;
# print "result $reply\n";
#if ($result ne 'y') {
#    exit;
#}
#else {
#    print "OK\n";
#}
# unless ($result eq "y") { exit;}

print "Here we go\n";
@TODO = ("mps_auto.pl", "mps_fire.pl", "mps_update.pl", "mps_check.pl", "mps_save.pl", 
	 "mps_setup.pl", "mps_fetch.pl", "mps_kill.pl", "mps_retry.pl", "mps_scriptm.pl", 
	 "mps_splice.pl", "mps_stat.pl");
while (@TODO) {
    $theName = shift @TODO;
    print "Open $workPath/$theName\n";
    open INFILE,"$workPath/$theName";
    undef $/;  # undefine the INPUT-RECORD_SEPARATOR
    $body = <INFILE>;  # read whole file
    # print "Body is $body\n";
    close INFILE;
    $/ = "\n"; # back to normal
    # $nn = ($body =~ m/use lib \'.+\'\;/);
    # print "nn = $nn \n $& \n";
    $nn = ($body =~ s/use lib \'.+\'\;/use lib \'$workPath\/mpslib\'\;/);
    open OUTFILE,">$workPath/$theName";
    print OUTFILE $body;
    close OUTFILE;
}

# we also need to update the path in the lib
$theName = "mpslib/Mpslib.pm";
print "Open $workPath/$theName\n";
open INFILE,"$workPath/$theName";
undef $/;  # undefine the INPUT-RECORD_SEPARATOR
$body = <INFILE>;  # read whole file
# print "Body is $body\n";
close INFILE;
$/ = "\n"; # back to normal
# $nn = ($body =~ m/use lib \'.+\'\;/);
# print "nn = $nn \n $& \n";
$nn = ($body =~ s/theSdir \= \".+\"\;/theSdir \= \"$workPath\"\;/);
print "nn = $nn \n $& \n";
open OUTFILE,">$workPath/$theName";
print OUTFILE $body;
close OUTFILE;

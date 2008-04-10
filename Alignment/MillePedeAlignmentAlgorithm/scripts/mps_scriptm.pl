#!/usr/local/bin/perl
#     R. Mankel, DESY Hamburg     06-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.14 $
#     $Date: 2008/03/25 16:15:57 $
#
#  Prepare the run script for the merge job.
#  The main action is to embed the output directory
#  into the script
#
#  Usage:
#
#  mps_scriptm.pl inScript outScript runDir cfgName njobs mssDir
#

use POSIX;

$inScript = "undefined";
$outScript = "undefined";
$runDir = "undefined";
$cfgName = "undefined";
$nJobs = "undefined";
$mssDir = "undefined";

# parse the arguments
while (@ARGV) {
  $arg = shift(ARGV);
  if ($arg =~ /\A-/) {  # check for option 
    if ($arg =~ "h") {
      $helpwanted = 1;
    }
    elsif ($arg =~ "d") {
      $localdir = 1;
    }
    elsif ($arg =~ "u") {
      $updateDb = 1;
    }
    $optionstring = "$optionstring$arg";
  }
  else {                # parameters not related to options
    $i = $i + 1;
    if ($i eq 1) {
      $inScript = $arg;
    }
    elsif ($i eq 2) {
      $outScript = $arg;
    }
    elsif ($i eq 3) {
      $runDir = $arg;
    }
    elsif ($i eq 4) {
      $cfgName = $arg;
    }
    elsif ($i eq 5) {
      $nJobs = $arg;
    }
    elsif ($i eq 6) {
      $mssDir = $arg;
    }
  }
}

if ($cfgName eq "undefined") {
  print "Insufficient information given\n";
  exit 1;
}

# open the input file
open INFILE,"$inScript";
undef $/;  # undefining the INPUT-RECORD_SEPARATOR means slurp whole file
$body = <INFILE>;  # read whole file
close INFILE;
$/ = "\n"; # back to normal

# replace RUNDIR setting
$nn = ($body =~ m/RUNDIR=(.+)$/m);
if ($nn != 1) {
  print "mps_script.pl: no (unambiguous) RUNDIR directive found in runscript\n";
  exit 1;
}
$nn = ($body =~ s/RUNDIR=(.+)$/RUNDIR=$runDir/m);

# replace MSSDIR setting
$nn = ($body =~ m/MSSDIR=(.+)$/m);
if ($nn != 1) {
  print "mps_script.pl: no (unambiguous) MSSDIR directive found in runscript\n";
}
$nn = ($body =~ s/MSSDIR=(.+)$/MSSDIR=$mssDir/m);

# replace the cfg name
$nn = ($body =~ m/cmsRun +([A-Z,a-z,0-9\-\.])/g);
# $nn = ($body =~ m/cmsRun +(.+)/g);
if ($nn <1) {
  print "Warning: mps_script matches cfg: $nn\n";
}
# $nn = ($body =~ s/cmsRun\s([A-Za-z0-9]+?\.cfg)/cmsRun $cfgName/g);
# $nn = ($body =~ s/cmsRun +(.+)/cmsRun $cfgName/g);
$nn = ($body =~ s/cmsRun +[a-zA-Z_0-9\-]+\.cfg/cmsRun $cfgName/g);

# now we have to expand lines that contain the ISN directive
@LINES = split "\n",$body;

foreach $theLine (@LINES) {
  if ($theLine =~ m/ISN/) {
    $newBlock = "";
    for ($i = 1; $i <= $nJobs; ++$i) {
      $newLine = $theLine;
      $isnRep = sprintf "%03d",$i;
      $newLine =~ s/ISN/$isnRep/;
      if ($i != 1) { $newBlock = $newBlock . "\n"; }
      $newBlock = $newBlock . $newLine;
    }
    $theLine = $newBlock;
  } 
}
$body = join "\n",@LINES;

# store the output file
open OUTFILE,">$outScript";
print OUTFILE $body;
close OUTFILE;
system "chmod a+x $outScript";



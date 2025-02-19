#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_scriptm.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Prepare the run script for the merge job.
#  The main action is to embed the output directory
#  into the script
#
#  Usage:
#
#  kaps_scriptm.pl [-c] inScript outScript runDir cfgName njobs mssDir
#

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;
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
    elsif ($arg =~ "c") {
# Check which jobs are "OK" and write just them to the cfg file
      $checkok = 1;
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

if ($checkok == 1) {
  read_db();
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
  print "kaps_script.pl: no (unambiguous) RUNDIR directive found in runscript\n";
  exit 1;
}
$nn = ($body =~ s/RUNDIR=(.+)$/RUNDIR=$runDir/m);

# replace MSSDIR setting
$nn = ($body =~ m/MSSDIR=(.+)$/m);
if ($nn != 1) {
  print "kaps_script.pl: no (unambiguous) MSSDIR directive found in runscript\n";
}
$nn = ($body =~ s/MSSDIR=(.+)$/MSSDIR=$mssDir/m);

# replace the cfg name
$nn = ($body =~ m/cmsRun +([A-Z,a-z,0-9\-\.])/g);
# $nn = ($body =~ m/cmsRun +(.+)/g);
if ($nn <1) {
  print "Warning: kaps_script matches cfg: $nn\n";
}
# $nn = ($body =~ s/cmsRun\s([A-Za-z0-9]+?\.cfg)/cmsRun $cfgName/g);
# $nn = ($body =~ s/cmsRun +(.+)/cmsRun $cfgName/g);
$nn = ($body =~ s/cmsRun +[a-zA-Z_0-9\-]+\.py/cmsRun \$RUNDIR\/$cfgName/g);

# now we have to expand lines that contain the ISN directive
@LINES = split "\n",$body;

foreach $theLine (@LINES) {
  if ($theLine =~ m/ISN/) {
    $newBlock = "";
    for ($i = 1; $i <= $nJobs; ++$i) {

      if ($checkok==1 && @JOBSTATUS[$i-1] ne "OK") {next;}

      $newLine = $theLine;
      $isnRep = sprintf "%03d",$i;
      $newLine =~ s/ISN/$isnRep/g;
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



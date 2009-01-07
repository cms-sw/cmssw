#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_merge.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  produce cfg file for merging run
#
#  Usage:
#
#  kaps_merge.pl [-c] inCfg mergeCfg mergeDir njobs

BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/kapslib");
}
use Kapslib;

$inCfg = "undefined";
$mergeCfg = "undefined";
$mergeDir = "undefined";
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
      $inCfg = $arg;
    }
    elsif ($i eq 2) {
      $mergeCfg = $arg;
    }
    elsif ($i eq 3) {
      $mergeDir = $arg;
    }
    elsif ($i eq 4) {
      $nJobs = $arg;
    }
    elsif ($i eq 5) {
      $mssDir = $arg;
    }

  }
}

if ($nJobs eq "undefined") {
  print "Insufficient information given\n";
  exit 1;
}

if ($checkok == 1) {
  read_db();
}

# open the input file
open INFILE,"$inCfg";
undef $/;  # undefining the INPUT-RECORD_SEPARATOR means slurp whole file
$body = <INFILE>;  # read whole file
close INFILE;
$/ = "\n"; # back to normal

# remove comment lines
$nn = ($body =~ s/^\s*\#.*$/COMMENTLINEREMOVED/mg);
$nn = ($body =~ s/COMMENTLINEREMOVED\n//mg);
$nn = ($body =~ s/COMMENTLINEREMOVED//mg);

# remove all occurences of dummy string 'ISN'
$nn = ($body =~ s/ISN//mg);

# create merge dir
unless (-d $mergeDir) {
    system "mkdir $mergeDir";
}

# change mode to merge
$nn = ($body =~ s/MergeResults.+?\)/MergeResults \= cms.bool\( True \)/);
if ($nn != 1) {
    print "No process.AlignmentProducer.algoConfig.MergeResults directive found.\n";
    exit(0);
}

# build list of binary files
$mergerFiles = "\n        ";
for ($i=1; $i<=$nJobs; ++$i) {
  $sep = ",\n        ";
  if ($i == $nJobs) { $sep = "" ;}

  $newName = sprintf "kaaOutput%03d.root",$i;
  print "Adding $newName to list of input files\n";
  $newLine = "\"$newName\"$sep";
  $mergerFiles = "$mergerFiles $newLine";
}

# replace list of input files
$nn = ($body =~ s/InputMergeFileNames.+?\)/InputMergeFileNames \= cms.vstring\($mergerFiles\)/);

if ($nn != 1) {
    print "No process.AlignmentProducer.algoConfig.Merger.InputMergeFileNames directive found!!!\n";
    exit(0);
}

# replace source directive... this is nasty.
$nn = ($body =~ s/maxEvents.+?\).+?\)/maxEvents = cms.untracked.PSet\( input = cms.untracked.int32\(0\) \)/s);
$nn = ($body =~ s/Source.+?skipEvents.+?fileNames.+?\).+?\)/Source\( \"EmptySource\" \)/s);
if ($nn != 1) { $nn = ($body =~ s/Source.+?fileNames.+?skipEvents.+?\).+?\)/Source\( \"EmptySource\" \)/s); }
if ($nn != 1) { $nn = ($body =~ s/Source.+?fileNames.+?\).+?\)/Source\( \"EmptySource\" \)/s); }

# replace MSSDIR setting
$nn = ($body =~ s/MSSDIR/$mssDir/gm);

# store the output file
open OUTFILE,">$mergeCfg";
print OUTFILE $body;
close OUTFILE;


#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_splice.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Take card file, blank all INFI directives and insert the INFI
#  directives from the modifier file instead.
#
#  Usage:
#
#  kaps_splice.pl inCfg files outCfg isn mssdir

$inCfg = "undefined";
$modCfg = "undefined";
$outCfg = "undefined";
$isn = "undefined";
$mssdir = "undefined";

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
      $inCfg = $arg;
    }
    elsif ($i eq 2) {
      $modCfg = $arg;
    }
    elsif ($i eq 3) {
      $outCfg = $arg;
    }
    elsif ($i eq 4) {
      $isn = $arg;
    }
    elsif ($i eq 5) {
      $mssdir = $arg;
    }
  }
}

if ($outCfg eq "undefined") {
  print "Insufficient information given\n";
  exit 1;
}


# open the input file
open INFILE,"$inCfg";
undef $/;  # undefining the INPUT-RECORD_SEPARATOR means slurp whole file
$body = <INFILE>;  # read whole file
close INFILE;
$/ = "\n"; # back to normal

# read the modifier file
open MODFILE,"$modCfg";
undef $/;  # undefining the INPUT-RECORD_SEPARATOR means slurp whole file
$mods = <MODFILE>;  # read whole file
close MODFILE;
$/ = "\n"; # back to normal
chomp $mods;

# prepare the new filenames directive
@FILENAMES = split "\n",$mods;
# GF
if ($FILENAMES[0] =~ /^CastorPool=/) { # starts with CastorPool
  @FILENAMES = @FILENAMES[1..$#FILENAMES]; # remove that line
}
# end GF
$newFileNames = "";
while (@FILENAMES) {
  $theFile = shift(@FILENAMES);
  chomp $theFile;
  $newFileNames = "$newFileNames\"$theFile\"";
  if (@FILENAMES) {
    $newFileNames = "$newFileNames,\n";
  }
}

# insert fileNames directive
$nrep = ($body =~ s/fileNames.*?untracked\.vstring\(.*?\)/fileNames = cms.untracked.vstring\($newFileNames\n\) /s);

# replace ISN for the root output file
$nrep = ($body =~ s/ISN/$isn/gm);

# replace MSSDIR for the root output file
$nrep = ($body =~ s/MSSDIR/$mssdir/gm);

# store the output file
open OUTFILE,">$outCfg";
print OUTFILE $body;
close OUTFILE;


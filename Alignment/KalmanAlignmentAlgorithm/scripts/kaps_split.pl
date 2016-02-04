#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_split.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Extract INFI directives from input file and write chunk to standard
#  output according to specified splitting scheme. A first line starting
#  with 'CastorPool=' will always be output as first line.
#
#  Usage:
#
#  kaps_split.pl infile thisChunk numberOfChunks
#
use POSIX;

$inFile = "undefined";
$thisGen = 1;
$totalGen = 0;

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
      $inFile = $arg;
    }
    elsif ($i eq 2) {
      $thisGen = $arg;
    }
    elsif ($i eq 3) {
      $totalGen = $arg;
    }
  }
}

if ($totalGen <= 0) {
  print "Insufficient information given\n";
  exit 1;
}

if ($thisGen <= 0) {
  print "kaps_split: zero thisGen --> exit\n";
  exit 1;
}
if ($thisGen > $totalGen) {
  print "kaps_split: thisGen gt totalGen, $thisGen gt $totalGen --> exit\n";
  exit 1;
}

# open the input file
open INFILE,"$inFile";
undef $/;  # undefining the INPUT-RECORD_SEPARATOR means slurp whole file
$body = <INFILE>;  # read whole file
close INFILE;
$/ = "\n"; # back to normal

# split the input into lines
@LINES = split "\n",$body;

# GF
my $poolDef = "";
if ($LINES[0] =~ /^CastorPool=/) { # starts with CastorPool
  $poolDef = $LINES[0];
#  $pool =~ s/CastorPool=//; # first appearance erased
  @LINES = @LINES[1..$#LINES]; # shorten
}
# end GF

# how many INFI cards
$ninfi = $#LINES + 1;

# Now for the arithmetics
$chunkSize = $ninfi / $totalGen;

$startId =floor( ($thisGen - 1) * $chunkSize ) + 1;
$endId = floor( $thisGen * $chunkSize);

if ($thisGen == 1) {
  $startId = 1;
}
if  ($thisGen == $totalGen) {
  $endId = $ninfi;
}

# print "chunkSize $chunkSize\n ninfi $ninfi startID $startId endId $endId\n";

if ($endId < $startId) {
  print "Invalid interval, startId=$startId, endID=$endId\n";
  print "ninfi was $#infi\n";
  exit 2;
}

# print "startId= $startId  endID= $endId\n";

print "$poolDef\n" if ($poolDef ne ""); #GF

# prepare the split output
for ($i = $startId; $i <= $endId; ++$i) {
  print "@LINES[$i-1]\n";
}

#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     06-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.2 $
#     $Date: 2008/07/30 14:10:18 $
#
#  Extract INFI directives from input file
#  and write chunk to standard output according to
#  specified splitting scheme.
#  A first line starting with 'CastorPool=' will always be output as first line.
#
#  Usage:
#
#  mps_split.pl infile thisChunk numberOfChunks
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
  print "mps_split: zero thisGen --> exit\n";
  exit 1;
}
if ($thisGen > $totalGen) {
  print "mps_split: thisGen gt totalGen, $thisGen gt $totalGen --> exit\n";
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

# how many INFI cards
$ninfi = $#LINES + 1;

my $poolDef = "";
if ($LINES[0] =~ /^CastorPool=/) { # starts with CastorPool
  $poolDef = $LINES[0];
  $ninfi = $ninfi - 1; # so we have one line less
}

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
if ($poolDef ne "") {  #first line was pool definition, so shift again
  $startId = $startId + 1;
  $endId = $endId + 1;
}

# print "chunkSize $chunkSize\n ninfi $ninfi startID $startId endId $endId\n";

if ($endId < $startId) {
  print "Invalid interval, startId=$startId, endID=$endId\n";
  print "ninfi was $#infi\n";
  exit 2;
}

# print "thisGen= $thisGen total=$totalGen startId= $startId  endID= $endId\n";

print "$poolDef\n" if ($poolDef ne ""); #GF

# prepare the split output
for ($i = $startId; $i <= $endId; ++$i) {
  print "@LINES[$i-1]\n";
}

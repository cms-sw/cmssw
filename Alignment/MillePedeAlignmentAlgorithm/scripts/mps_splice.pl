#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     03-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.9 $
#     $Date: 2011/06/15 14:24:52 $
#
#  Take card file, blank all INFI directives and insert
#  the INFI directives from the modifier file instead.
#
#  Usage:
#
##GF  mps_splice.pl inCfg modCfg outCfg isn
#  mps_splice.pl inCfg files outCfg isn

$inCfg = "undefined";
$modCfg = "undefined";
$outCfg = "undefined";
$isn = "undefined";

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
  }
}

if ($outCfg eq "undefined") {
  print "mps_splice.pl: Insufficient information given\n";
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

#FIXME: make more robust regexp
$nn = ($body =~ s/mode \= \'full\'/mode \= \'mille\'/);
if ($nn) {
  print "Replaced mode from 'full' to 'mille'.\n";
}


########## (AP) Look for the fileNames/readFiles directives

$nn = ($body =~ s/fileNames = cms.untracked.vstring\(.*?\)/fileNames = readFiles/gs); # Remove filenames from "fileNames" directive

$body =~ s/^ *readFiles.extend.*$//mg; # First remove any "readFiles.extend" directive
$nn = ($body =~ s/^ *readFiles *=.*$/readFiles = cms.untracked.vstring()\nreadFiles.extend(\'file.root'\)/m); # Look for "readFiles =" directive

if ($nn != 1) { # If not found, add the "readFiles =" directive before process.source
  $nn = ($body =~ s/^ *process.source = cms.Source/readFiles = cms.untracked.vstring()\nreadFiles.extend(\'file.root'\)\n\nprocess.source = cms.Source/m);
}

# prepare the new filenames directive
@FILENAMES = split "\n",$mods;
# GF
if ($FILENAMES[0] =~ /^CastorPool=/) { # starts with CastorPool
  @FILENAMES = @FILENAMES[1..$#FILENAMES]; # remove that line
}

# end GF

######### RC/AP

$newFileNames = "\n";
$TempFileNames = "\n";
$MergingFileNames="\n";

while(@FILENAMES>255) {
  $f=0;
  while($f<255) {
## first while: check if theSplit contains more than 255 files

    $theFile = shift(@FILENAMES);
    chomp $theFile;
    $f++;
    $TempFileNames = "$TempFileNames        \'$theFile\'";
    if ($f!=255) {
      $TempFileNames = "$TempFileNames,\n";
    }
  } #end 2nd while
  $MergingFileNames="$MergingFileNames \nreadFiles.extend\(\($TempFileNames \)\)\n";
  $TempFileNames ="\n";
} #end 1st while

while(@FILENAMES) {

## second while: check if theSplit is not empty

  $theFile = shift(@FILENAMES);
  chomp $theFile;
  $newFileNames = "$newFileNames        \'$theFile\'";
  if (@FILENAMES) {
    $newFileNames = "$newFileNames,\n";
  } ## end if
} ## end while

# Count how many files are there in $newFileNames
$nn = () = $newFileNames =~ /\.root/g;

if ($nn > 1) {
  $MergingFileNames="$MergingFileNames \nreadFiles.extend\(\($newFileNames \)\)";
} else { # One file only... use single parenthesis
  $MergingFileNames="$MergingFileNames \nreadFiles.extend\(\[$newFileNames\]\)";
}

$nrep = ($body =~ s/^ *readFiles.extend\(\'file.root\'\)/$MergingFileNames/gm);

####### RC/AP end

# replace ISN for the root output file
$nrep = ($body =~ s/ISN/$isn/gm);

$body .= "\n\nprocess.AlignmentProducer.saveDeformationsToDB = False\n";

# store the output file
open OUTFILE,">$outCfg";
print OUTFILE $body;
close OUTFILE;


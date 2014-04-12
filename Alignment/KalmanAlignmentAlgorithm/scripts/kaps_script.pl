#!/usr/bin/env perl
#
#  This script is part of the Kalman Alignment Production System (KAPS).
#  It is an adapted version of mps_script.pl, a part of the MillePede
#  Production System (MPS), developed by R. Mankel (DESY).
#
#  Prepare the run script for this job. The main action is to embed the
#  output directory into the script
#
#  Usage:
#
#  kaps_script.pl inScript outScript runDir cfgName fileSplit isn mssDir
#

use POSIX;

$inScript = "undefined";
$outScript = "undefined";
$runDir = "undefined";
$cfgName = "undefined";
$fileSplit = "undefined";
$isn = "undefined";
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
      $fileSplit = $arg;
    }
    elsif ($i eq 6) {
      $isn = $arg;
    }
    elsif ($i eq 7) {
      $mssDir = $arg;
    }
  }
}

if ($isn eq "undefined") {
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
$nn = ($body =~ s/cmsRun +[a-zA-Z_0-9\-]+\.py/cmsRun \$RUNDIR\/$cfgName/g);

# here we insert prestager commands, a la
# stager_get -M /castor/cern.ch/cms/store/mc/2007/5/9/Spring07-ZToMuMu-1532/0001/E665DFFD-99FF-DB11-8730-000E0C3F08AE.root
$prestageBlock = "# Begin of Castor Prestage Block";
if ($fileSplit ne "undefined") {
  open SPLITFILE,"$fileSplit";
  while ($text = <SPLITFILE>) {
    chomp $text;

    # Format 1: /store/...  : in this case, prepend /castor path
    if ( $text =~ m/^ *\/store/ ) {
      $text =~ s/\/store/stager_get -M \/castor\/cern.ch\/cms\/store/;
    }
    # Format 2: rfio:/castor/...  : in this case, remove rfio prefix path
    elsif ( $text =~ m/^ *rfio:/ ) {
      $text =~ s/rfio:/stager_get -M /;
#GF
    }
    # Format 3: CastorPool definition (should happen only for first entry... FIXME?)
    elsif ($text =~ /^CastorPool=/) {
      $text =~ s/CastorPool=//; # first appearance of CastorPool erased
      if ($body =~ /^\#!\/bin\/(tcsh|csh)/) { # script starts with #!/bin/tcsh or #!/bin/csh
	$text = "setenv STAGE_SVCCLASS ".$text # (t)csh way of setting variables
      } else { # other shells
	$text = "export STAGE_SVCCLASS=".$text
      }
# end GF
    }

    $prestageBlock = "$prestageBlock\n$text";
  }
  close SPLITFILE;
  $prestageBlock = "$prestageBlock\n# End of Castor Prestage Block";
  # insert prestage block at the beginning (after the first line)
  $body =~ s/.+/$&\n$prestageBlock\n/;
} 

# replace ISN for the root output file
$nrep = ($body =~ s/ISN/$isn/gm);

# store the output file
open OUTFILE,">$outScript";
print OUTFILE $body;
close OUTFILE;
system "chmod a+x $outScript";



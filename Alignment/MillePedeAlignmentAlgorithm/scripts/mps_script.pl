#!/usr/bin/env perl
#     R. Mankel, DESY Hamburg     06-Jul-2007
#     A. Parenti, DESY Hamburg    27-Mar-2008
#     $Revision: 1.7 $
#     $Date: 2011/06/15 14:24:52 $
#
#  Prepare the run script for this job.
#  The main action is to embed the output directory
#  into the script
#
#  Usage:
#
#  mps_script.pl inScript outScript runDir cfgName fileSplit isn mssDir \
#  [CastorPool]
#

use POSIX;

$inScript = "undefined";
$outScript = "undefined";
$runDir = "undefined";
$cfgName = "undefined";
$fileSplit = "undefined";
$isn = "undefined";
$mssDirLocal = "undefined"; # not to confuse with mssDir from 'mpslib'.
$castorPool = "undefined";
$cmsCafPool = 0;

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
      $mssDirLocal = $arg;
    }
    elsif ($i eq 8) {
      $castorPool = $arg;
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
  print "mps_script.pl: no (unambiguous) RUNDIR directive found in runscript\n";
  exit 1;
}
$nn = ($body =~ s/RUNDIR=(.+)$/RUNDIR=$runDir/m);

#replace CMSSW_RELEASE_AREA with evironment variable
$body =~ s/cd\s+CMSSW_RELEASE_AREA/cd $ENV{'CMSSW_BASE'}/g;

# replace MSSDIR setting
$nn = ($body =~ m/MSSDIR=(.+)$/m);
if ($nn != 1) {
  print "mps_script.pl: no (unambiguous) MSSDIR directive found in runscript\n";
}
$nn = ($body =~ s/MSSDIR=(.+)$/MSSDIR=$mssDirLocal/m);

if ($castorPool ne "undefined") {
# replace MSSDIRPOOL setting...
  $nn = ($body =~ s/MSSDIRPOOL=(.*)$/MSSDIRPOOL=$castorPool/m);
} else {
#... or empty the field.
  $nn = ($body =~ s/MSSDIRPOOL=(.*)$/MSSDIRPOOL=/m);
}

# replace the cfg name
$nn = ($body =~ m/cmsRun +([A-Z,a-z,0-9\-\.])/g);
# $nn = ($body =~ m/cmsRun +(.+)/g);
if ($nn <1) {
  print "Warning: mps_script matches cfg: $nn\n";
}
# $nn = ($body =~ s/cmsRun\s([A-Za-z0-9]+?\.cfg)/cmsRun $cfgName/g);
# $nn = ($body =~ s/cmsRun +(.+)/cmsRun $cfgName/g);
$nn = ($body =~ s/cmsRun +[a-zA-Z_0-9\-]+\.cfg/cmsRun \$RUNDIR\/$cfgName/g);

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

      if ($text =~ /cmscaf/) {
# AP 24.06.2010 - Input data resides on cmscaf pool
	$cmsCafPool = 1;
      }

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

  if ($cmsCafPool eq 1) {
# AP 24.06.2010 - Input data resides on cmscaf pool... no prestaging, just setup cmscaf!
    $prestageBlock = ". /afs/cern.ch/cms/caf/setup.sh";
  }

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



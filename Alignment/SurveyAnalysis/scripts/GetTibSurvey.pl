#!/usr/bin/env perl -s
#
# Perl script to retrieve TIB/TID survey data
#
# Data is stored in the format defined by the relevant DetId constructors.
#
# TIB:
# Layer# | Z-/Z+ | int/ext | string# | module # | data
#
# TID:
# Z-/Z+ | Disk# | Ring# | Fw/Bw | module# | dummy | data
#
# where data contains:
# - the global position vector;
# - the 3 orientation vectors w, u, v (in the global frame);
# - the radius vector.
#
# Author: F. Ronga
# Data: February 13, 2007
#

use strict;

# Do not bufferize output
$|=1;

### Files to be excluded
my %TIDexclude = ( "TID+_Ring3_Disk3_back" => 1,
                   "TID-_Ring3_Disk3_back" => 1 );

### Global variables
my $baseUrl = "http://hep.fi.infn.it/CMS/software/CMS_Geometry/Programs/";
my $nTibLayers = 4; # Number of TIB layers
my $nTidDisks  = 3; # Number of TID disks
my $nTidRings  = 3; # Number of TID rings

# Output file
my $outputFile = "Survey.dat";

### Initialise http agent
require LWP::UserAgent;
my $ua = LWP::UserAgent->new;

### 1. Retrieve TIB data
my @strSides = ( "int", "ext" );
my @zSides = ( "-", "+" );


my $TIBoutput = "TIB".$outputFile;
open( OUTPUT,">$TIBoutput" ) or die "Couldn't open $TIBoutput: $!";

print "Writing output to $TIBoutput\n";
# Loop on all TIB files
for ( my $iLayer = 1; $iLayer <= $nTibLayers; $iLayer++ ) {
  foreach my $zSide ( @zSides ) {
    foreach my $strSide ( @strSides )  {
      # Form file and directory name
      my $name = "TIB".$zSide."_Layer".$iLayer."_".$strSide;

      print "  Processing $name...";

      # Geometry data
      my $geomUrl = $baseUrl."/".$name."/datafiles/".$name."_geometry.dat";
      my $gResponse = $ua->get($geomUrl);
      die $gResponse->status_line if ( !$gResponse->is_success );
      my $geometry = $gResponse->content();

      # Survey data
      my $surveyUrl = $baseUrl."/".$name."/datafiles/".$name."_survey.dat";
      my $sResponse = $ua->get($surveyUrl);
      die $sResponse->status_line if ( !$sResponse->is_success );
      my $survey = $sResponse->content();

      my @geomData = &extractData( $geometry );
      my @surveyData = &extractData( $survey );
      my @id = ( $iLayer, ($zSide=~/-/?0:1), ($strSide=~/int/?0:1) ); # See TIBDetId

      &printOut( \@id, \@geomData, \@surveyData );

      print " done\n";
    }
  }
}

close(OUTPUT) or die "Couldn't close $TIBoutput: $!";
print $TIBoutput." all done\n";


# 2. Retrieve TID data
my @dSides = ( "back", "front" );

my $TIDoutput = "TID".$outputFile;
open ( OUTPUT,">$TIDoutput" ) or die "Couldn't open $TIDoutput: $!";
print "Writing output to $TIDoutput\n";

# Loop on all TID files
for ( my $iRing = 1; $iRing<=$nTidRings; $iRing++ ) {
  for ( my $iDisk = 1; $iDisk<=$nTidDisks; $iDisk++ ) {
    foreach my $zSide ( @zSides ) {
      foreach my $dSide ( @dSides ) {

        # Form file and directory name
        my $name = "TID".$zSide."_Ring".$iRing."_Disk".$iDisk."_".$dSide;
   
        # Check exclusion
        if ( $TIDexclude{$name} ) {
           print "  SKIPPING $name\n";
           next;
        }

        print "  Processing $name...";

        # Geometry data
        my $geomUrl = $baseUrl."/".$name."/datafiles/".$name."_geometry.dat";
        my $gResponse = $ua->get($geomUrl);
        die $gResponse->status_line if ( !$gResponse->is_success );
        my $geometry = $gResponse->content();

        # Survey data
        my $surveyUrl = $baseUrl."/".$name."/datafiles/".$name."_survey.dat";
        my $sResponse = $ua->get($surveyUrl);
        die $sResponse->status_line if ( !$sResponse->is_success );
        my $survey = $sResponse->content();

        my @geomData = &extractData( $geometry );
        my @surveyData = &extractData( $survey );
        my @id = ( ($zSide=~/-/?1:2), $iDisk, $iRing, ($dSide=~/back/?0:1) ); # See TIDDetId

        &printOut( \@id, \@geomData, \@surveyData );

        print " done\n";
      }
    }
  }
}

close(OUTPUT) or die "Couldn't close $TIDoutput: $!";
print $TIDoutput." all done\n";

#_______________________________________________________________________________
# Extract data from input text, sort it according to string and module
# Return sorted array
sub extractData {

  my $data   = shift;

  my @lines = split( "\n", $data );

  shift @lines; # Remove first line

  return sort byStringMod @lines;

}


#_______________________________________________________________________________
# Sort entries by string and module (first two fields)
sub byStringMod {

  my @afields = split( /\s+/, $a );
  my @bfields = split( /\s+/, $b );

  if ( $afields[0] != $bfields[0] ) { # String number
    return $afields[0] <=> $bfields[0];
  } else { # Module number
    return $afields[1] <=> $bfields[1];
  }

}


#_______________________________________________________________________________
# Print out the final numbers
sub printOut {

  my $id = shift;
  my $geometry = shift;
  my $survey   = shift;

  # Check length of data
  if ( @$geometry != @$survey )
    {
      print " *** Warning: missing data in this file. Skipping.";
      return;
    }

  my $count = 0;
  foreach my $gLine ( @$geometry )
    {
      foreach my $field ( @$id ) {
        printf OUTPUT "%2d ", $field;
      }

      my @gFields = split( /\s+/, $gLine );
      printf OUTPUT "%3d %2d ", shift @gFields, shift @gFields;
      print OUTPUT join( " ", @gFields);
      print OUTPUT " ";

      my @sFields = split( /\s+/, $$survey[$count] );
      shift @sFields; shift @sFields;
      print OUTPUT join( " ", @sFields);
      print OUTPUT "\n";

      ++$count;
    }

}

#!/bin/env perl
#
# A script to parse cfi files for parameters to replace
# Probably not very stable...
#


### To turn on verbosity, uncomment following line
#open(MSG,">&STDERR") or die "Couldn't open stderr: $!";

use strict;

# Extract source name from argument
my $config = $ARGV[0];
(my $source = $config) =~ s/.*(CMSSW)/$1/;


# Alignment of comments: column number
my $alignment = 60;

print "# Generated from $source\n";
print MSG "Parsing  $source\n";

open(INPUT,$config) or die "Couldn't open $config: $!";
my $levelName = ""; # Name of current block/module
my $lineNb = 0;
my $isMulti = 0;
MAIN: while( <INPUT> ) {
  print MSG ++$lineNb."\n";

  # Skip empty lines
  next if (!/\S/);

  # Skip includes
  next if ( /\s*include\s+\".*\"/ );

  # Skip commented-out lines
  next if (m@^\s*(//|#)@);

  chomp();
  my $line = $_; # Store line

  while ( $line =~ /\S/ ) {
    # 1. Process named blocks (modules, blocks, PSets)
    ($line,$levelName) = &processBlocks( $line, $levelName );

    # 2. Skip VPSets and sequences
    #    NB. This is fragile!
    if ( $line =~ /(VPSet|sequence)\s+(\S+)\s*=\s*\{\s*/ ) {
      print MSG "Found $1 with name $2: skipping\n";
      my $nBraces = 1; # Start with one, from above matching
      $line = $';
      $nBraces += &countBraces( $line );
      print MSG "$nBraces: $line\n";
      if ( $nBraces>0 ){
        while ( <INPUT> ) {
          $line = $_;
          $nBraces += &countBraces( $line );
          print MSG "$nBraces: $line\n";
          last if ( $nBraces <= 0 );
        }
      }
      $line = "";
      next;       # This assumes closing brace is last on line...
    }

    # 3. Process parameters (might be multiline)
    ($line,$isMulti) = &processParameters( $line, $levelName );
    if ( $isMulti ) { # Treat mutliline separately...
      while (<INPUT>) {
        print MSG ++$lineNb."\n";
        s/^\s+//; # Remove leading spaces
        chomp();
        $_ = &nukeComments( $_ ); # Remove comments in that case
        if ( /.*?\}/ ) {
          print $line.$&."\n"; # Dump what we have found
          $line = $';     # Store remainder in line for future use
          $isMulti = 0;
          last;
        } else {
          $line .= $_;
        }
      }
      print MSG "End of span\n";
    }

    # 4. Remove remaining comments on blocks
    $line = &nukeComments( $line );

    # 5. Climb up levels if braces are closed
    ($line,$levelName) = &closeBraces( $line, $levelName );
  }
}
close(INPUT);


#_______________________________________________________________________
# Check line for new blocks and add them to level name
sub processBlocks {

  my $line = shift;
  my $levelName = shift;

  if ( $line =~ /(module|block|[^V]PSet)\s+(\S+)\s*=\s*(\S+)?\s*\{\s*/ ) {
    print MSG "Found $1 with name $2".($3?" and label $3":"")."\n";
    $levelName .= (length($levelName)>0?'.':'').$2;
    $line = $';
  }

  return ($line,$levelName);
}


#_______________________________________________________________________
# Check for new parameters
sub processParameters{

  my $line = shift;
  my $levelName = shift;
  my $isMulti = 0;

  if ( $line =~ /([\w\d]+)\s+(\S+)\s*=\s*(.*)$/ ) {
    my $type = $1;
    my $name = $2;
    my $value = $3;
    $line = $';

    # Check for un-balanced closing brace and put back on line
    if ( $value !~ /\{/ && $value =~ /\}/ ) {
      $value = $`;
      $line = '}'.$line;
    }

    print MSG "Found $type with name $name and value $value";
    # Check if this parameter spans over several lines
    if ( $value =~ /\{/ && $value !~ /\}/ ) {
      $isMulti++;
      print MSG " spanning over multiple lines";
      $value = &nukeComments($value); # Don't keep comments in that case: too disturbing
    }
    print MSG "\n";

    # Dump out
    &dumpReplace($levelName,$name,$value);
    if ( !$isMulti ) { print "\n"; }

  }


  return $line,$isMulti;

}

#_______________________________________________________________________
# Remove trailing comments from block definitions (can't carry them)
sub nukeComments {

  my $line = shift;

  if ( $line =~ m@\s*(//|#)@ ) {
    $line = $`;
  }
  return $line;

} 

#______________________________________________________________________
# Close braces and adjust levelName correspondingly
sub closeBraces {

  my $line = shift;
  my $levelName = shift;

  if ( $line =~ /^\s*\}\s*/ ) {
    my $curLevel = $levelName;
    print MSG "Found closing brace - climbing up: '$levelName' -> ";
    $levelName =~ s/\.[^\.]*?$//;
    if ( $curLevel =~ /^$levelName$/ ) { # Treat special case...
      $levelName = "";
    }
    $line = $';
    print MSG "'$levelName'\n";
  }
  return ($line,$levelName);

}

#______________________________________________________________________
# Count number of braces
# Opening adds one, closing removes one
sub countBraces {

  my $string = shift;
  my $nBraces = 0;
  my $char = "";

  while ( length($string)>0 ) {
    $char = chop($string);
    ++$nBraces if ( $char =~ /\{/ );
    --$nBraces if ( $char =~ /\}/ );
  }

  return $nBraces;

}


#______________________________________________________________________
# Subroutine to nicely dump the replace statements
# Tries to align comments
sub dumpReplace {
  my $prefix = shift; # Level name
  my $name   = shift; # Parameter name
  my $string = shift; # Value, including possible comment

  if ( $string =~ /(#|\/\/)\s*/ ) {
    my $value = $`;
    my $comment = $';
    $value =~ s/\s+$//g; # Remove trailing spaces
    # Alignment: add necessary spaces
    while ( length($prefix.$name." = ".$value) < $alignment ) { $value .= " "; }
    $string = $value." # ".$comment;
  }
  print "replace $prefix.$name = $string";
}

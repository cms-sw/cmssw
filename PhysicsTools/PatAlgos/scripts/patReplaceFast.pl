#!/bin/env perl
#
# Replaces for fast simulation
#
# Usage: $0 <file> <replaces file>
#        Replaces in <file> from <replaces file>

use strict;

die "Usage: $0 <file> <replaces>\n" if ( @ARGV != 2 );

my $cfgFile = $ARGV[0];
my $replaceFile = $ARGV[1];

my %replaces;

# Get list of replaces
open(IN,$replaceFile) or die "Couldn't open $replaceFile: $!";
while( <IN> ) {
  if ( /replace\s+(\S+)(\s*=.*)[#|\/\/]?/ ) {
    my $key   = $1;
    my $value = $2;
    # Check for multi-line value
    while ( &countBraces($value) > 0 ) {
      my $line = <IN>;
      $value .= $line;
    }
    $replaces{$key} = $value;
  }
}
close(IN);

# Open temporary file
my $temp = $cfgFile.".tmp";
open(TEMP,">$temp") or die "Couldn't open $temp: $!";

# Scan cfgfile and replace
open(CFG,$cfgFile) or die "Couldn't open $cfgFile: $!";
while ( <CFG> ) {
  chomp();
  my $line = $_;
  my $matched = 0;
  foreach my $replace ( keys %replaces ) {
    next if ( !$replaces{$replace} );
    if ( $line =~ /$replace/ ) {
      print TEMP "replace ".$replace.$replaces{$replace}."\n";
      $replaces{$replace} = 0;
      ++$matched;
      last;
    }
  }
  if ( !$matched ) {
    print TEMP $line."\n";
  }
}
close(CFG);


# Add missing items at the end of file (things that are only replaced in FastSim)
# Print header
print TEMP <<HEADER
############################
# Fast simulation replaces #
############################
HEADER
;
foreach my $replace ( keys %replaces ) {
  next if ( !$replaces{$replace} );
  print TEMP "replace ".$replace.$replaces{$replace}."\n";
}
close(TEMP) or die "Couldn't close $temp: $!";
rename($temp,$cfgFile);


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

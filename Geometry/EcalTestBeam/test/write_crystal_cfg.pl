#! /usr/bin/perl -w

# build the cff file for all the crystals corresponding to
# their nominal maximum reference point as stored in the table                                                                                
# F. Cossutti - 23-Aug-2006 15:19


use diagnostics;
use strict;
                                                                                
my $file = "BarrelSM1CrystalCenterElectron120GeV.dat";

if (! open(INPUT,"<$file") ) {
    print STDERR "Can't open input file $file: $!\n";
    exit 1;
   }
                                                                                 
    while(<INPUT>){
        chomp(my $line = $_);

        print $line, "\n";

        my $crystal = my $eta = my $phi = ();

        ($crystal,$eta,$phi) = split(' ', $line, 3); 

        my $filename = "crystal".$crystal.".cff";

        open(OUTFILE, ">$filename");
        print OUTFILE "block common_beam_direction_parameters = {\n";
        print OUTFILE "  untracked double MinEta = ".$eta,"\n";
        print OUTFILE "  untracked double MaxEta = ".$eta,"\n";
        print OUTFILE "  untracked double MaxPhi = ".$phi,"\n";
        print OUTFILE "  untracked double MinPhi = ".$phi,"\n";
        print OUTFILE "  untracked double BeamMeanX = 0.\n";
        print OUTFILE "  untracked double BeamMeanY = 0.\n";
        print OUTFILE "  untracked double BeamPosition = 0.\n";
        print OUTFILE "}\n";
 
    }
    close(INPUT);
 
exit 0;

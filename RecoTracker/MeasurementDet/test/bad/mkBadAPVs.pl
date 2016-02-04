use strict;
use warnings;
open ALL, $ENV{'CMSSW_RELEASE_BASE'}."/src/CalibTracker/SiStripCommon/data/SiStripDetInfo.dat" or die "Can't find SiStripDetInfo.dat";

open OUT, "> BadAPVs.cff" or die "Can't write to BadAPVs.cff";
print OUT <<_EOF;
replace prod.BadComponentList = {
_EOF

my $first = 1;
my $tot = 0; my $bads = 0;
while(<ALL>) {
    my ($detid, $napv, $foo, $bar) = split(/\s+/, $_);
    my @badAPVs = ();

    # --- insert logic here ------    

    if (($napv <= 6) && ($detid =~ /^3/)) {  
        foreach my $apv (0 .. ($napv-1)) { 
           if (rand() < 0.2) { 
              push @badAPVs, $apv; 
           }
        }
    }

    # --- end logic --------------

    if (@badAPVs) {
        my $record = sprintf("{ uint32 BadModule = %d vuint32 BadApvList = {%s} }",
                                $detid, join(',', @badAPVs));
        print OUT ($first ? "\t " : "\t,"), $record, "\n";
        $first = 0;
        $bads++;
    }
    $tot++;
}

print OUT "}\n";
close OUT;
print "Summary: $bads modules with bad APVs over $tot total modules.\n";

use strict;
use warnings;
use CMSDetID;

open ALL, $ENV{'CMSSW_RELEASE_BASE'}."/src/CalibTracker/SiStripCommon/data/SiStripDetInfo.dat" or die "Can't find SiStripDetInfo.dat";

open OUT, "> BadModules.cff" or die "Can't write to BadModules.cff";
print OUT <<_EOF;
replace prod.BadModuleList = {
_EOF

my $first = 1;
my $tot = 0; my $bads = 0;
while(<ALL>) {
    my ($detid, $napv, $foo, $bar) = split(/\s+/, $_);
    my $bad = 0;
    my $DetId = CMSDetID->new($detid);

    # --- insert logic here ------    

    $bad = 1 if ($DetId->subdetName() eq 'TIB') && ($DetId->tiblayer() == 3);
    $bad = 1 if ($DetId->subdetName() eq 'TIB') && ($DetId->tiblayer() == 4);

    # --- end logic --------------

    if ($bad) {
        print OUT ($first ? "\t " : "\t,"), $detid, "\n";
        $first = 0;
        $bads++;
    }
    $tot++;
}

print OUT "}\n";
close OUT;
print "Summary: $bads modules over $tot total modules.\n";

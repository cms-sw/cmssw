use strict;
use warnings;
open ALL, $ENV{'CMSSW_RELEASE_BASE'}."/src/CalibTracker/SiStripCommon/data/SiStripDetInfo.dat" or die "Can't find SiStripDetInfo.dat";

open OUT, "> BadStrips.cff" or die "Can't write to BadStrips.cff";
print OUT <<_EOF;
replace prod.BadComponentList = {
_EOF

sub block($$) { 
    my ($first, $length) = @_;
    return ($first .. ($first + $length - 1));
}

my $first = 1;
my $tot = 0; my $bads = 0;
while(<ALL>) {
    my ($detid, $napv, $foo, $bar) = split(/\s+/, $_);
    my $strips = 128 * $napv;
    my @badStrips = ();

    # --- insert logic here ------    

    my $prob   = 0.20; #10% of the modules will have at least one bad strip
    my $pmult  = 0.30; #30% chanche of having another bad block after the first one
    #if (($napv <= 6) && ($detid =~ /^3/)) {  
        my $ptr = int(rand($strips/$prob));
        while ($ptr < $strips) {
            my $len = (rand() < 0.2 ? 1 : int(rand(8)+1));
            push @badStrips, block($ptr,$len); 
            $ptr += $len; last unless $ptr < $strips;
            $ptr += int(rand(($strips-$ptr)/$pmult)); 
        }
    #}

    # --- end logic --------------

    if (@badStrips) {
        my $record = sprintf("{ uint32 BadModule = %d vuint32 BadChannelList = {%s} }",
                                $detid, join(',', @badStrips));
        print OUT ($first ? "\t " : "\t,"), $record, "\n";
        $first = 0;
        $bads++;
    }
    $tot++;
}

print OUT "}\n";
close OUT;
print "Summary: $bads modules with bad Strips over $tot total modules.\n";

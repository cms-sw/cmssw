#!/usr/bin/env perl
use warnings;
use strict;
use threads;
use threads::shared;

my $dir = $ENV{'CMSSW_BASE'} . "/src/PhysicsTools/PatAlgos/test";
chdir $dir or die "Can't chdir to $dir.\n Check that CMSSW_BASE variable is set correcly, and that you have checked out PhysicsTools/PatAlgos.\n";

my %done   :shared = ();
my $fake   :shared = 0;
my $repdef :shared = "";

use Getopt::Long;
my $help;
GetOptions( 'h|?|help' => \$help, 
            'n|dry-run' => \$fake,
            'replace-defaults|rd=s' => \$repdef );
if ($help) {
    my $name = `basename $0`; chomp $name;
    print "Usage: perl $name [-n|--dry-run] [cfgs]\n" .
          "   -n or --dry-run:      just read the output, don't run cmsRun\n".
          "   -h or --help:         print this help\n".
          "   --replace-defaults X  turn on ReplaceDefaults for layer X (X can be 0, 1 or 01 for both)\n".
          "                         they must already be in the CFG file, commented with # or // \n".
          "   If no cfgs are specified, it will use PATLayer[01]_from*_{fast,full}.cfg\n\n";
    exit(0);
}
if ($fake) {
    print "=== NOT ACTUALLLY RUNNING cmsRun, JUST READING OUTPUT LOGS ===\n";
}

my @CFGs = @ARGV;

if (@CFGs) {
    my @allCFGs = ();
    foreach my $cfg (@CFGs) { push @allCFGs, grep(/\.cfg$/, glob($cfg)); }
    @CFGs = @allCFGs;
} else {
    @CFGs = glob("patLayer[01]_from*_*.cfg"); 
}
print "Will run " . scalar(@CFGs) . " config files: " . join(' ', @CFGs) . "\n\n";

foreach my $cfg (@CFGs) { 
    unless (-f $cfg) {  die "Config file $cfg does not exist in $dir\n"; } 
    repDef($cfg);
}

sub repDef {
    my $f = shift;
    open CFG, $f or die "Can't open cfg file '$f'\n";
    my $text = join('',<CFG>); my $original = $text;
    close CFG;
    foreach my $l (0, 1) {
        if ($repdef =~ /$l/) {
            #print STDERR "Toggling ON  ReplaceDefaults for patLayer$l in $f \n";
            $text =~ s!(?://\s*|\x23\s*)*(include\s+["']PhysicsTools/PatAlgos/test/patLayer${l}_ReplaceDefaults_\w+.cff.)!$1!g;
            $text =~ s!(?://\s*|\x23\s*)*(include\s+["']PhysicsTools/PatAlgos/data/famos/patLayer${l}_FamosSetup.cff.)!\x23$1!g;
        } else {
            #print STDERR "Toggling OFF ReplaceDefaults for patLayer$l in $f \n";  # \x23 is the '#' sign
            $text =~ s!(?://\s*|\x23\s*)*(include\s+["']PhysicsTools/PatAlgos/test/patLayer${l}_ReplaceDefaults_\w+.cff.)!\x23$1!g;
            $text =~ s!(?://\s*|\x23\s*)*(include\s+["']PhysicsTools/PatAlgos/data/famos/patLayer${l}_FamosSetup.cff.)!$1!g;
        }
    }
    if ($text ne $original) {
        open CFG, "> $f" or die "Can't update cfg file '$f'\n";
        print CFG $text;
        close CFG;
        #print "============== OLD ===================\n$original\n============== NEW ===================\n$text\n";
    }
}

sub cmsRun {
    my ($f, $o) = ($_[0], $_[1]);
    unless ($fake) {
        repDef($f);
        system("sh -c 'cmsRun --strict $f > $o 2>&1 '");
    } else {
        system("sh -c 'sleep ". int(rand(5)+2) ."'");
    }
    $done{$f} = time();
}

my %info = ();

my @txt = ("Jobs starting:");
foreach my $f (@CFGs) {
    my $o = $f; $o =~ s/\.cfg$/.log/;

    my $max = -1;
    open CFG, $f;
    foreach(<CFG>) { 
        m/untracked\s+PSet\s+maxEvents/ and $max = 0;
        if ($max == 0) { m/untracked int32 input = (\d+)/ and $max = $1; }
    }
    close CFG;

    push @txt, "   \e[32;1m$f\e[37;0m: starting (on $max events total)";
    $info{$f} = { 'out' => $o, 'start' => time(), 'max'=>$max };
    my $thr = threads->create(\&cmsRun, $f, $o);
}
print join("\n", @txt), "\n";

sub printDone {
    my $f = shift(@_);
    return "\e[32;1m$f\e[37;0m: \e[;;1mdone\e[m events " . $info{$f}->{'last'} . "/" . $info{$f}->{'max'} .
          ", total time " . ($done{$f} - $info{$f}->{'start'}) . "s, " .
          $info{$f}->{'lines'} . " output lines, " .
          ($info{$f}->{'excep'} ? "\e[1;31m" . $info{$f}->{'excep'} . " exceptions\e[0m" : "\e[32mno exceptions\e[0m" );
}

sub printRunning {
    my $f = shift(@_);
    my $lines = 0; my $last = 0;
    my ($excep, $exbody) = (0,"");
    open LOG, $info{$f}->{'out'};
    while (<LOG>) { 
        $lines++; 
        m/Begin processing the (\d+)\S* record\./ and $last = $1;
        if (m/---- (.*?) BEGIN/) {
            my $exname = $1;
            $excep++; 
            if ($excep == 1) { $exbody .= "\t" . $_; }
            while ($_ = <LOG>) { 
                $lines++; 
                if ($excep == 1) { $exbody .= "\t" . $_; }
                last if (m/---- $exname END/);
            }
        }
    };
    close LOG;

    my $secs = time() - $info{$f}->{'start'};
    $info{$f}->{'time'}  = $secs;
    $info{$f}->{'last'}  = $last;
    $info{$f}->{'lines'} = $lines;
    $info{$f}->{'excep'} = $excep;
    $info{$f}->{'exbody'}= $exbody;
    return "\e[32;1m$f\e[37;0m: event $last/" . $info{$f}->{'max'} ." (time ${secs}s, ${lines} output lines, " . 
        ($excep ? "\e[1;31m$excep exceptions\e[0m" : "\e[32mno exceptions yet\e[0m" ) . ")...";
}

while (scalar(keys(%done)) < scalar(@CFGs)) {
    sleep 1;

    foreach my $f (@txt) { print "\e[F\e[M"; };  @txt = ();

    my @run = (); my @done = ();
    foreach my $f (@CFGs) {
        if (defined($done{$f}) and defined($info{$f}->{'last'})) {
           push @done, "   " . printDone($f); 
        } else {
           push @run,  "   " . printRunning($f);
        }
    }
    push @txt, ("Jobs running:", @run);
    if (@done) { push @txt, ("Jobs done:", @done) };
    push @txt, "";
    print join("\n", @txt), "\n";
}

foreach my $f (@txt) { print "\e[F\e[M"; }

print "All jobs done.\n";
  
foreach my $f (@CFGs) {
    print printDone ($f), "\n";
    if ($info{$f}->{'excep'}) { print "\e[1;31m" . $info{$f}->{'exbody'} . "\e[0m"; }

    open LOG, $info{$f}->{'out'}; my @log = <LOG>; close LOG;
    foreach my $l (grep(/^Input tag was|Summary info:/, @log)) { print "  $l"; }
    foreach my $l (grep(/TrigReport Events total =/, @log)) { print "  \e[1m$l\e[0m"; }

    print "\n";
}

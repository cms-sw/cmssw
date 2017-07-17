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
my ($help,$base,$extra,$all,$shy,$obj);
my @summary :shared;
GetOptions( 'h|?|help' => \$help,
            'n|dry-run' => \$fake,
            'b|base' => \$base,
            'o|obj' => \$obj,
            'a|all' => \$all,
            'e|extra' => \$extra,
            's|summary=s' => \@summary,
            'q|quiet' => \$shy);
@summary = split(/,/, join(',',@summary));
if ($help) {
    my $name = `basename $0`; chomp $name;
    print "Usage: perl $name [-n|--dry-run] [cfg.pys]\n" .
          "   -n or --dry-run: just read the output, don't run cmsRun\n".
          "   -h or --help:    print this help\n".
          "   -b or --base:    add base standard PAT config files to the jobs to run\n".
          "   -o or --obj:     add PAT config files for single physics objecs to the jobs to run\n".
          "   -e or --extra:   add the extra standard PAT config files to the jobs to run (that is, those not in base)\n".
          "   -a or --all:     add all standard PAT config files to the jobs to run\n".
          "   -s or --summary: print summary table of objects (argument can be 'aod', 'allLayer1', 'selectedLayer1', ...)\n".
          "   -q or --quiet:   print summary tables only if there are warnings/errors in that table.\n";
    exit(0);
}
if ($fake) {
    print "=== NOT ACTUALLLY RUNNING cmsRun, JUST READING OUTPUT LOGS ===\n";
}

my @CFGs = map({$_ =~ /\.py$|\*$/ ? $_ : "*$_*"}   @ARGV);


my @anyCFGs    = glob("pat*[._]cfg.py");
my @baseCFGs   = grep($_ =~ /standard|fastSim|data|PF2PAT/, @anyCFGs);
my @objCFGs   = grep($_ =~ /only/, @anyCFGs);
my @extraCFGs  = grep($_ =~ /add|metUncertainties|userData/, @anyCFGs);
if ($base ) { push @CFGs, @baseCFGs;  }
if ($obj )  { push @CFGs, @objCFGs;  }
if ($all  ) { push @CFGs, @anyCFGs;   }
if ($extra) { push @CFGs, @extraCFGs; }
if (@CFGs) {
    #pass through a hash, to remove duplicates
    my %allCFGs = ();
    foreach my $cfg (@CFGs) {
        foreach my $cfgfile (grep(/cfg\.py$/, glob($cfg))) { $allCFGs{$cfgfile} = 1; }
    }
    @CFGs = sort(keys(%allCFGs));
} else {
    print STDERR "Please specify a cfg, or use --one, --base, --all, --extra to select the standard ones\n"; exit(0);
}

print "Will run " . scalar(@CFGs) . " config files: " . join(' ', @CFGs) . "\n\n";

foreach my $cfg (@CFGs) {
    unless (-f $cfg) {  die "Config file $cfg does not exist in $dir\n"; }
}

sub cmsRun {
    my ($f, $o) = ($_[0], $_[1]);
    unless ($fake) {
        system("sh -c 'cmsRun $f > $o 2>&1 '");
    } else {
        #system("sh -c 'sleep ". int(rand(5)+2) ."'");
        system("sh -c 'sleep 1'");
    }
    $done{$f} = time();
}

my %info = ();

my @txt = ("Jobs starting:");
foreach my $f (@CFGs) {
    my $o = $f; $o =~ s/[\._]cfg\.py$/.log/;

    my $max = -1;
    open CFG, $f;
    foreach(<CFG>) {
        m/maxEvents\s*=\s*cms\.untracked\.PSet/ and $max = 0;
        if ($max == 0) { m/input\s*=\s*cms\.untracked\.int32\(\s*(\d+)\s*\)/ and $max = $1; }
        if ($max == -1) {
          m/process\.maxEvents\.input\s*=\s*/ and $max = 0;
          if ($max == 0) { m/process\.maxEvents\.input\s*=\s*(\d+)/ and $max = $1; }
        }
    }
    close CFG;

    push @txt, "   \e[32;1m$f\e[37;0m: starting (on $max events total)";
    $info{$f} = { 'out' => $o, 'start' => time(), 'max'=>$max };
    my $thr = threads->create(\&cmsRun, $f, $o);
}
print join("\n", @txt), "\n";

sub printDone {
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

    $info{$f}->{'last'}  = $last;
    $info{$f}->{'lines'} = $lines;
    $info{$f}->{'excep'} = $excep;
    $info{$f}->{'exbody'}= $exbody;
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

sub redIf($$) {
    return ($_[1] ? "\e[1;31m E> " . $_[0] . "\e[0m" : "    ".$_[0]);
}
foreach my $f (@CFGs) {
    print printDone ($f), "\n";
    if ($info{$f}->{'excep'}) { print "\e[1;31m" . $info{$f}->{'exbody'} . "\e[0m"; }

    open LOG, $info{$f}->{'out'}; my @log = <LOG>; close LOG;
    my $log = join('',@log);
    foreach my $table (@summary) {
        if ($log =~ m/(^Summary Table\s+$table.*\n(^    .*\n)*)/m) {
            my $info = $1;
            $info =~ s/^    (.*present\s+\d+\s+\(\s*(\d+\.?\d*)\s*%\).*)$/redIf($1,$2 ne '100')/meg;
            $info =~ s/^    (.*total\s+0\s.*)$/\e[1;31m E> $1\e[0m/mg;
            if (!$shy or ($info =~ /\e\[1;31m E>/)) {
                print "  ".$info;
            }
        }
    }
    foreach my $l (grep(/TrigReport Events total =/, @log)) { print "  \e[1m$l\e[0m"; }

    print "\n";
}

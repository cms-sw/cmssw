#!/usr/bin/env perl
use Env;
($#ARGV >= 2) || die "Usage::reset_seal_cache.pl libdir moddir";
$libdir = @ARGV[0];
$moddir = @ARGV[1];
$envfile = @ARGV[2];
my $sourceme = `scramv1 runtime -sh`;
@tok1 = split(/\n/,$sourceme);
while($#tok1 != -1)
{
    my $env =  @tok1[$#tok1];
    $env =~ s/export (.+)/$1/;
    $env =~ s/SCRAMRT_([^";]+)/$ENV{$1}/;
    $env =~ s/;//;
    $env =~ s/"//g;
    $env =~ /(.*)=(.*)/;
#    print "setting $1 to $2\n";
    $ENV{$1}=$2;
    pop(@tok1);
}

$ENV{'LD_LIBRARY_PATH'}=$libdir;
$ENV{'SEAL_PLUGINS'}=$moddir;
#$ENV{LOG}='stderr';
print "removing $moddir/.cache\n";
$resp = `rm $moddir/.cache`;
print "$resp\n";
$resp = `SealPluginRefresh`;
print $resp;


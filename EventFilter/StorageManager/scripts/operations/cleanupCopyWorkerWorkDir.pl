#!/usr/bin/env perl

use strict;
use File::Path;

my $basedir = '/store/copyworker/workdir/';
my %h;

sub wanted {
    # Data.00129762.1825.A.storageManager.01.0000.dat.log
    my ($dev,$ino,$mode,$nlink,$uid,$gid);

    -f "$basedir/$_" && (int(-C _) > 7)
    && /^[^.]+\.([0-9]+)\..*\.[0-9]+\.[0-9]+\.(dat|ind)\.log$/
    && push @{ $h{$1} }, $_;
}

# Traverse desired filesystems
opendir( DIR, $basedir ) or die "Can't open $basedir: $!";
wanted $_ for readdir DIR;
closedir DIR;

for my $run ( sort keys %h ) {
    print "For run number $run we have $#{$h{$run}} files\n";
    my $dir = sprintf( "$basedir/%09d", $run );
    $dir =~ s!([0-9]{3})!$1/!g;
    if(! -d $dir){
	print " mkpath( $dir, 1, 0755 ) \n ";
	mkpath( $dir, 1, 0755 );
    }else{
	print "dir $dir already exists! \n";
    }
    rename "$basedir/$_", "$dir/$_" for @{ $h{$run} };

}

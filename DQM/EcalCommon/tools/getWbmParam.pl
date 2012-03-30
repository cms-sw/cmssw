#!/usr/bin/env perl

my $targetLine = "";

my $paramName = $ARGV[0];
my $pattern = $ARGV[1];
my $fileName = $ARGV[2];

if(! -e "$fileName"){
    exit;
}

open HTMLFILE, $fileName;

while(<HTMLFILE>){
    chomp;
    $line = lc($_);
    my $pat;
    $pat = 'cms.lvl0.' . $paramName . '[^0-9a-z_]';
    if($line =~ m:$pat:){
	$targetLine = $line;
    }
    elsif($targetLine ne ""){
	$targetLine .= $line;
    }
    $pat = 'cms.lvl0.' . $paramName . '\s*<\/td>\s*<td>\s*(' . $pattern . ')\s*</td>';
    if($targetLine =~ m:$pat:){
	$targetLine = "";
	print $1;
    }
}

close HTMLFILE;

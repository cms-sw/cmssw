#!/usr/bin/env perl

use strict;

open INPUT, "<crab.tmpl";
open OFILE, ">crab.cfg";
while (<INPUT>) {

    #my $num = sprintf "%08d", $ARGV[0];
    chomp($_);

    if ($_ =~ s/XXXXXX/$ARGV[0]/) {
	print OFILE "$_\n";
    } else {
	print OFILE "$_\n";
    }

}

close(INPUT); 
close(OFILE); 

system("mkdir $ARGV[0]; mv crab.cfg $ARGV[0]; cp esAnalysis.py $ARGV[0]");
system(". /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.sh; . /afs/cern.ch/cms/sw/cmsset_default.sh; eval `scramv1 runtime -sh`; . /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh; cd $ARGV[0]; crab -create crab.cfg; crab -submit");
#system(". /afs/cern.ch/cms/sw/cmsset_default.sh; . /afs/cern.ch/cms/sw/cmsset_default.sh; cmsenv; . /afs/cern.ch/cms/ccs/wm/scripts/Crab/crab.sh; cd $ARGV[0]; crab -create crab.cfg; crab -submit");

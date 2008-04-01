#!/usr/bin/env perl
# $Id: dummy.pl,v 1.1 2007/02/05 15:48:55 klute Exp $

use strict;

open LOG, ">> /tmp/dummyScript.log" or die "open: $!\n";
print LOG scalar localtime(time),' ',join(', ',@ARGV),"\n";
close LOG;

exit 0;

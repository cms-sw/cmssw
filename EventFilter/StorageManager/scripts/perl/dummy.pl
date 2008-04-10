#!/usr/bin/env perl
# Created by Markus Klute on 2007 Jan 29.
# $Id:$
################################################################################
#
# Dummy script
#
################################################################################

use strict;

open LOG, ">> /tmp/dummyScript.log" or die "open: $!\n";
print LOG scalar localtime(time),' ',join(', ',@ARGV),"\n";
close LOG;

exit 0;

#!/usr/bin/perl -w
# Created by Tier0 team on 2007 Jan 29.
# $Id:$
################################################################################
#
# First dummy version
#
################################################################################

use strict;

print "Thank you for calling me with the arguments: ",join(', ',@ARGV),"\n";
open LOG, ">> /tmp/tier0.log" or die "open: $!\n";
print LOG scalar localtime(time),' ',join(', ',@ARGV),"\n";
close LOG;

exit 0;

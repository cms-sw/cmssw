#!/usr/bin/env perl

use warnings;
use strict;

print "\n*** CRYSTAL INTEGRITY ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_crystal_integrity.pl @ARGV");

print "\n*** TT INTEGRITY ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_tt_integrity.pl @ARGV");

print "\n*** MEM CH INTEGRITY ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_mem_ch_integrity.pl @ARGV");

print "\n*** MEM TT INTEGRITY ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_mem_tt_integrity.pl @ARGV");

print "\n*** PEDESTAL ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_pedestal.pl @ARGV");

print "\n*** PN PEDESTAL ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_pn_pedestal.pl @ARGV");

print "\n*** PEDESTAL ONLINE ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_pedestal_online.pl @ARGV");

print "\n*** TEST PULSE ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_test_pulse.pl @ARGV");

print "\n*** PN MGPA ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_pn_mgpa.pl @ARGV");

print "\n*** LASER BLUE ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_laser_blue.pl @ARGV");

print "\n*** PN BLUE ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_pn_blue.pl @ARGV");

print "\n*** LASER RED ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_laser_red.pl @ARGV");

print "\n*** PN RED ***\n\n";

system("/var/www/html/badchannels/scripts/eb-list_bad_pn_red.pl @ARGV");

print "\n";

exit;


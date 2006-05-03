#!/usr/bin/perl

use warnings;
use strict;

my $orcon_pass = "****";
my $orcoff_pass = "****";

my $conn_orcon = "sqlplus -S cms_cond_ecal/${orcon_pass}\@orcon";
my $conn_orcoff = "sqlplus -S cms_cond_ecal/${orcoff_pass}\@cms_orcoff";

my $cmd_orcon = "cat cmds.sql | ${conn_orcon} >> orcon_poll.txt";
my $cmd_orcoff = "cat cmds.sql | ${conn_orcoff} >> orcoff_poll.txt";

while (1) {
    `$cmd_orcon`;
    `$cmd_orcoff`;
    sleep(60);
}

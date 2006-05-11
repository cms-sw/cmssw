#!/usr/bin/perl

use warnings;
use strict;

my $orcon_user = ""
my $orcon_pass = "__CHANGE_ME__";
my $orcoff_user = "";
my $orcoff_pass = "__CHANGE_ME__";

my $conn_orcon = "sqlplus -S ${orcon_user}/${orcon_pass}\@orcon";
my $conn_orcoff = "sqlplus -S ${orcoff_user}/${orcoff_pass}\@cms_orcoff";

my $cmd_orcon = "cat cmds.sql | ${conn_orcon} >> orcon_poll.txt";
my $cmd_orcoff = "cat cmds.sql | ${conn_orcoff} >> orcoff_poll.txt";

while (1) {
    `$cmd_orcon`;
    `$cmd_orcoff`;
    sleep(60);
}

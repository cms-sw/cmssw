#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use TB04::Importer;
use ConnectionFile;

print "Loading Importer (connect to DB)...";
my $importer = new TB04::Importer;
print "Done.\n";

# an IoV for the duration of the 2004 test beam
my $testbeam_IoV = { since => "2004-10-21 18:04:00", till => "2004-11-23 10:44:00" };

# go to work
fill_Xtal_alpha();
fill_Xtal_T_coeff();
fill_APD_T_coeff();

# fill the Xtal alpha table with dummy values of 1.56 
sub fill_Xtal_alpha {
  my $cond_name = "Xtal_alpha";
  my $view_name = "EB_crystal_number";
  my $dummy = 1.56;

  print "Loading view $view_name...";
  $importer->load_view($view_name);
  print "Done.\n";

  print "Filling $cond_name with dummy...";
  my $SM = 10;
  foreach my $xtal (1..1700) {
    $importer->insert($cond_name, $view_name, [$SM, $xtal], $testbeam_IoV, $dummy);
  }
  print "Done.\n";
}

sub fill_Xtal_T_coeff {
  my $cond_name = "Xtal_T_coeff";
  my $view_name = "EB_crystal_number";
  my $dummy = -0.02;

  print "Loading view $view_name...";
  $importer->load_view($view_name);
  print "Done.\n";

  print "Filling $cond_name with dummy...";
  my $SM = 10;
  foreach my $xtal (1..1700) {
    $importer->insert($cond_name, $view_name, [$SM, $xtal], $testbeam_IoV, $dummy);
  }
  print "Done.\n";
}

sub fill_APD_T_coeff {
  my $cond_name = "APD_T_coeff";
  my $view_name = "EB_crystal_number";
  my $dummy = -0.02;

  print "Loading view $view_name...";
  $importer->load_view($view_name);
  print "Done.\n";

  print "Filling $cond_name with dummy...";
  my $SM = 10;
  foreach my $xtal (1..1700) {
    $importer->insert($cond_name, $view_name, [$SM, $xtal], $testbeam_IoV, $dummy);
  }
  print "Done.\n";
}


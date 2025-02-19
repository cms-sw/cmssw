#!/usr/bin/env perl

use lib "./lib";

use warnings;
use strict;
$|++;

use ConnectionFile;
use Getopt::Long;

my $all = 0;
my $printlist = 0;
GetOptions('all' => \$all,
	   'list' => \$printlist);

my $list = {
	    'HV' => \&define_HV,
	    'LV' => \&define_LV,
	    'ESS' => \&define_ESS,
	    'PTM' => \&define_PTM,
	    'LASER' => \&define_Laser,
	    'MGPA' => \&define_MGPA,
	    'CONSTRUCTION' => \&define_Construction,
	    'XTALCALIB' => \&define_Xtal_Calibrations,
	    'MONCALIB' => \&define_Monitoring_Calibrations
	   };

if ($printlist) {
  print "Valid conditions groups:\n";
  foreach (keys %{$list}) {
    print "\t$_\n";
  }
  exit;
}

if ($all) {
  @ARGV = keys %{$list};
}

die "No groups to define, exiting.\n" unless @ARGV;

print "Connecting to DB...";
my $condDB = ConnectionFile::connect();
print "Done.\n";

print "Defining conditions tables...\n";
my $count = 0;
foreach (@ARGV) {
  print "Defining conditions for group $_\n";
  my $sub = $list->{uc $_} || warn "$_ not defined\n";
  &{$sub}();
  $count++;
}
print "Done.  $count groups defined\n";

###
###   HV
###
sub define_HV {
  print "\tHV_vMon\n";
  $condDB->new_cond_type(-name=>"HV_vMon",
			 -description=>"The high voltage, monitored value",
			 -units=>"volts",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tHV_v0\n";
  $condDB->new_cond_type(-name=>"HV_v0",
			 -description=>"The high voltage, set value",
			 -units=>"volts",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tHV_iMon\n";
  $condDB->new_cond_type(-name=>"HV_iMon",
			 -description=>"The high voltage current, monitored value",
			 -units=>"amperes",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tHV_vMon\n";
  $condDB->new_cond_type(-name=>"HV_i0",
			 -description=>"The high voltage current, set value",
			 -units=>"amperes",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tHV_status\n";
  $condDB->new_cond_type(-name=>"HV_status",
			 -description=>"The high voltage status string",
			 -units=>undef,
			 -datatype=>"string",
			 -hasError=>0);

  print "\tHV_T_board\n";
  $condDB->new_cond_type(-name=>"HV_T_board",
			 -description=>"The high voltage board temperature",
			 -units=>"degrees C",
			 -datatype=>"float",
			 -hasError=>0);
}

###
###   Low Voltage
###

sub define_LV {
  print "\tLV_vMon\n";
  $condDB->new_cond_type(-name=>"LV_vMon",
			 -description=>"The low voltage, monitored value",
			 -units=>"volts",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tLV_outReg\n";
  $condDB->new_cond_type(-name=>"LV_outReg",
			 -description=>"The low voltage ???",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tLV_iMon\n";
  $condDB->new_cond_type(-name=>"LV_iMon",
			 -description=>"The low voltage current, monitored value",
			 -units=>"amperes",
			 -datatype=>"float",
			 -hasError=>0);
}

###
###   ESS
###
sub define_ESS {
  print "\tESS_temp\n";
  $condDB->new_cond_type(-name=>"ESS_temp",
			 -description=>"ESS temperature",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tESS_WLD\n";
  $condDB->new_cond_type(-name=>"ESS_WLD",
			 -description=>"ESS WLD ???",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);
}

###
###   PTM
###

sub define_PTM {
  print "\tPTM_H\n";
  $condDB->new_cond_type(-name=>"PTM_H",
			 -description=>"Humidity sensors on the modules",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tPTM_H_amb\n";
  $condDB->new_cond_type(-name=>"PTM_H_amb",
			 -description=>"Ambient humidity",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tPTM_T_amb\n";
  $condDB->new_cond_type(-name=>"PTM_T_amb",
			 -description=>"Ambient temperature",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tPTM_T_grid\n";
  $condDB->new_cond_type(-name=>"PTM_T_grid",
			 -description=>"Module grid temperature",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tPTM_T_screen\n";
  $condDB->new_cond_type(-name=>"PTM_T_screen",
			 -description=>"Module screen temperature",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tPTM_T_water_in\n";
  $condDB->new_cond_type(-name=>"PTM_T_water_in",
			 -description=>"Super-module water in temperature",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tPTM_T_water_out\n";
  $condDB->new_cond_type(-name=>"PTM_T_water_out",
			 -description=>"Super-module water out temperature",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasError=>0);
}


###
###   Laser
###

sub define_Laser {
  foreach my $color (qw/red green blue/) {
    print "\tLaser_PN_ratio_$color\n";
    $condDB->new_cond_type(-name=>"Laser_PN_ratio_$color",
			   -description=>"The PN ratio to describe the stability of the $color laser",
			   -units=>"???",
			   -datatype=>"float",
			   -hasError=>1);
  }

# foreach my $color (qw/red green blue/) {
#   print "\tLaser_pulse_max_laser_ref_$color\n";
#   $condDB->new_cond_type(-name=>"Laser_pulse_max_laser_ref_$color",
# 			 -description=>"Laser pulse max laser/ref, $color laser",
# 			 -units=>"???",
# 			 -datatype=>"float",
# 			 -hasError=>1);


#   print "\tLaser_pulse_max_laser_pin_$color\n";
#   $condDB->new_cond_type(-name=>"Laser_pulse_max_laser_pin_$color",
# 			 -description=>"Laser pulse max laser/pin, $color laser",
# 			 -units=>"???",
# 			 -datatype=>"float",
# 			 -hasError=>1);
# }

  print "\tLaser_pulse_fit_params\n";
  $condDB->new_cond_type(-name=>"Laser_pulse_fit_params",
			 -description=>"Laser pulse fit parameters 0-3",
			 -units=>"???",
			 -datatype=>"float",
			 -datasize=>4,
			 -hasError=>1);

  foreach my $color (qw/red green blue/) {
    print "\tLaser_corr_$color\n";
    $condDB->new_cond_type(-name=>"Laser_corr_$color",
			   -description=>"Laser monitoring calculated correction,  $color laser",
			   -units=>"???",
			   -datatype=>"float",
			   -hasError=>1);
  }

  foreach my $color (qw/red green blue/) {
    print "\tLaser_width_$color\n";
    $condDB->new_cond_type(-name=>"Laser_width_$color",
			   -description=>"The width of the laser pulse measured with MATACQ",
			   -units=>"???",
			   -datatype=>"float",
			   -hasError=>0);
  }

  print "\tLaser_test_pulse\n";
  $condDB->new_cond_type(-name=>"Laser_test_pulse",
			 -description=>"The test pulse amplitude",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);
}

###
###   MGPA
###
sub define_MGPA {
  print "\tMGPA_gain_intercal_g12g6\n";
  $condDB->new_cond_type(-name=>"MGPA_gain_intercal_g12g6",
			 -description=>"Values of the relative gain ".
			 "calibration of the MGPA, gain 12 / ".
			 "gain 6",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tMGPA_gain_intercal_g6g1\n";
  $condDB->new_cond_type(-name=>"MGPA_gain_intercal_g6g1",
			 -description=>"Values of the relative gain ".
			 "calibration of the MGPA, gain 6 / ".
			 "gain 1",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);
}

###
###   DCU
###
# For the test beam 2004 data only
sub define_DCU {
  print "\tDCU_capsule_temp\n";
  $condDB->new_cond_type(-name=>"DCU_capsule_temp",
			 -description=>"From the capsule thermister",
			 -units=>"deg C",
			 -datatype=>"float",
			 -hasErrors=>"0");
}

###
###   Coefficients from the Construction DB
###
sub define_Construction {
  print "\tAPD_T_coeff\n";
  $condDB->new_cond_type(-name=>"APD_T_coeff",
			 -description=>"1/M dM/dT",
			 -units=>"1/deg C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tAPD_V_coeff\n";
  $condDB->new_cond_type(-name=>"APD_V_coeff",
			 -description=>"(1/M)(dM/dV) for the capsule attached to the crystal",
			 -units=>"1/V",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tXtal_T_coeff\n";
  $condDB->new_cond_type(-name=>"Xtal_T_coeff",
			 -description=>"Temperature coefficient of the crystal",
			 -units=>"value/degrees C",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tXtal_alpha\n";
  $condDB->new_cond_type(-name=>"Xtal_alpha",
			 -description=>"The alpha coefficients due to irradiation of the crystal",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);
}

###
###   Crystal Calibration sets
###
sub define_Xtal_Calibrations {
  print "\tXtal_intercal_lab\n";
  $condDB->new_cond_type(-name=>"Xtal_intercal_lab",
			 -description=>"Calibration constant from lab measurements",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);

  print "\tXtal_intercal_milano\n";
  $condDB->new_cond_type(-name=>"Xtal_intercal_milano",
			 -description=>"Calibration constant from lab measurements, Milano group",
			 -units=>"???",
			 -datatype=>"float",
			 -hasError=>0);
}


###
###   Other Calibrations, corrections
###
sub define_Monitoring_Calibrations {
  print "\tPN_lin_corr_g16\n";
  $condDB->new_cond_type(-name=>"PN_lin_corr_g16",
			 -description=>"Linearity corrections for MGPA gain 16 PN diodes responses",
			 -units=>"???",
			 -datatype=>"float",
			 -datasize=>3,
			 -hasError=>0);

  print "\tPN_lin_corr_g1\n";
  $condDB->new_cond_type(-name=>"PN_lin_corr_g1",
			 -description=>"Linearity corrections for MGPA gain 1 PN diodes responses",
			 -units=>"???",
			 -datatype=>"float",
			 -datasize=>3,
			 -hasError=>0);

  foreach my $gain (qw/1 6 12/) {
    print "\tAPD_lin_corr_g${gain}\n";
    $condDB->new_cond_type(-name=>"APD_lin_corr_g${gain}",
			   -description=>"Linearity corrections for APD diode response with MGPA gain $gain",
			   -units=>"???",
			   -datatype=>"float",
			   -datasize=>3,
			   -hasError=>0);
  }
}

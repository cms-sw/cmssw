#!/usr/bin/env perl


#18012008 Various tests of EE numberings
use warnings;
use strict;
$|++;

use POSIX;

package CondDB::channelView;

sub new {
  my $proto = shift;
  my $class = ref($proto) || $proto;
  my $this = {};

  $this->{condDB} = shift;
  die "Usage:  channelView->new( \$condDB )\n" unless $this->{condDB};

  $this->{defs} = {
		   'DUMMY' => \&define_dummy,
		   'ECAL' => \&define_ECAL,
		   'EB' => \&define_EB,
		   'EE' => \&define_EE,
		   'EB_crystal_number' => \&define_EB_crystal_number,
		   'EB_elec_crystal_number' => \&define_EB_elec_crystal_number,
		   'EB_fe_crystal_number' => \&define_EB_fe_crystal_number,
		   'ECAL_crystal_number_fedccuxt' => \&define_ECAL_crystal_number_fedccuxt,
		   'ECAL_readout_strip' => \&define_ECAL_readout_strip,
		   'ECAL_trigger_tower' => \&define_ECAL_trigger_tower,
		   'EB_crystal_index' => \&define_EB_crystal_index,
		   'EB_trigger_tower' => \&define_EB_trigger_tower,
		   'EB_readout_tower' => \&define_EB_readout_tower,
		   'EB_supermodule' => \&define_EB_supermodule,
		   'EB_module' => \&define_EB_module,
		   'EB_HV_channel' => \&define_EB_HV_channel,
		   'EB_HV_board' => \&define_EB_HV_board,
		   'EB_LV_channel' => \&define_EB_LV_channel,
		   'EB_ESS_temp' => \&define_EB_ESS_temp,
		   'EB_PTM_H_amb' => \&define_EB_PTM_H_amb,
		   'EB_PTM_T_amb' => \&define_EB_PTM_T_amb,
		   'EB_token_ring' => \&define_EB_token_ring,
		   'EB_LM_channel' => \&define_EB_LM_channel,
		   'EB_LM_side' => \&define_EB_LM_side,
		   'EB_LM_PN' => \&define_EB_LM_PN,
		   'EB_T_capsule' => \&define_EB_T_capsule,
		   'EB_VFE' => \&define_EB_VFE,
		   'EB_LVRB_DCU' => \&define_EB_LVRB_DCU,
		   'EB_LVRB_T_sensor' => \&define_EB_LVRB_T_sensor,
		   'EB_mem_TT' => \&define_EB_mem_TT,
		   'EB_mem_channel' => \&define_EB_mem_channel,
		   'EB_trigger_crystal_number' => \&define_EB_trigger_crystal_number,
		   'ECAL_crystal_number_fedccuxt_to_EB_crystal_number' => \&define_ECAL_crystal_number_fedccuxt_to_EB_crystal_number,
		   'ECAL_trigger_tower_to_EB_trigger_tower' => \&define_ECAL_trigger_tower_to_EB_trigger_tower,
		   'EB_readout_tower_to_EB_trigger_tower' => \&define_EB_readout_tower_to_EB_trigger_tower,
		   'EB_crystal_number_to_EB_trigger_tower' => \&define_EB_crystal_number_to_EB_trigger_tower,
		   'EB_crystal_number_to_EB_LV_channel'
		   => \&define_EB_crystal_number_to_EB_LV_channel,
		   'EB_crystal_number_to_EB_module'
		   => \&define_EB_crystal_number_to_EB_module,
		   'EB_crystal_number_to_EB_HV_channel'
		   => \&define_EB_crystal_number_to_EB_HV_channel,
		   'EB_crystal_number_to_EB_LV_channel'
		   => \&define_EB_crystal_number_to_EB_LV_channel,
		   'EB_crystal_number_to_EB_LM_channel'
		   => \&define_EB_crystal_number_to_EB_LM_channel,
		   'EB_crystal_number_to_EB_LM_PN'
		   => \&define_EB_crystal_number_to_EB_LM_PN,
		   'EB_crystal_number_to_EB_LM_side'
		   => \&define_EB_crystal_number_to_EB_LM_side,
		   'EB_LM_side_to_EB_crystal_number'
		   => \&define_EB_LM_side_to_EB_crystal_number,
		   'EB_crystal_number_to_EB_T_capsule'
		   => \&define_EB_crystal_number_to_EB_T_capsule,
		   'EB_T_capsule_to_EB_crystal_number'
		   => \&define_EB_T_capsule_to_EB_crystal_number,
		   'EB_crystal_number_to_EB_VFE'
		   => \&define_EB_crystal_number_to_EB_VFE,
		   'EB_crystal_number_to_EB_elec_crystal_number',
		   => \&define_EB_crystal_number_to_EB_elec_crystal_number,
		   'EB_crystal_number_to_EB_fe_crystal_number'
		   => \&define_EB_crystal_number_to_EB_fe_crystal_number,
		   'EB_elec_crystal_number_to_EB_crystal_number',
		   => \&define_EB_elec_crystal_number_to_EB_crystal_number,
		   'EB_constr_crystal_number_to_EB_crystal_number'
		   => \&define_EB_constr_crystal_number_to_EB_crystal_number,
		   'EB_constr_supermodule_to_EB_supermodule'
		   => \&define_EB_constr_supermodule_to_EB_supermodule,
		   'EB_crystal_number_to_ECAL_LMR'
		   => \&define_EB_crystal_number_to_ECAL_LMR,
		   'EB_fe_crystal_number_to_EB_crystal_number',
		   => \&define_EB_fe_crystal_number_to_EB_crystal_number,

		#endcap
		   'EE_side' => \&define_EE_side,
		   'EE_D' => \&define_EE_D,
		   'EE_sector' => \&define_EE_sector,
		   'EE_DCC' =>\&define_EE_DCC, 	
		   'EE_crystal_number' => \&define_EE_crystal_number,
		   'ECAL_TCC'=>\&define_ECAL_TCC,
		   'EE_crystal_hashed' => \&define_EE_crystal_hashed,
		   'EE_trigger_tower'=>\&define_EE_trigger_tower,
		   'EE_readout_tower'=>\&define_EE_readout_tower,
		   'EE_readout_strip'=>\&define_EE_readout_strip,
		   'EE_trigger_strip'=>\&define_EE_trigger_strip,
		   'EE_crystal_readout_strip'=>\&define_EE_crystal_readout_strip,
		   'EE_crystal_trigger_strip'=>\&define_EE_crystal_trigger_strip,
		   'EE_elec_crystal_number'=>\&define_EE_elec_crystal_number,
		   'EE_readout_tower_xyz'=>\&define_EE_readout_tower_xyz,
		   'EE_readout_tower_zseccu'=>\&define_EE_readout_tower_zseccu,
		   'EE_HV_channel'=>\&define_EE_HV_channel,
		   'EE_HVA_channel'=>\&define_EE_HVA_channel,
		   'EE_HVD_channel'=>\&define_EE_HVD_channel,
		   'EE_LV_channel'=>\&define_EE_LV_channel,
		   'EE_readout_strip_zseccu'=>\&define_EE_readout_strip_zseccu,
		   'EE_LM_LMM'=>\&define_EE_LM_LMM,
		   'EE_LM_PN'=>\&define_EE_LM_PN,
		   'EE_PTM_T_amb'=>\&define_EE_PTM_T_amb,
		   'EE_PTM_H_amb'=>\&define_EE_PTM_H_amb,
		   'EE_LM_PN'=>\&define_EE_LM_PN,
		   'ECAL_LMR'=>\&define_ECAL_LMR,
		   'ECAL_DCC'=>\&define_ECAL_DCC,
		   'ECAL_DCC_input'=>\&define_ECAL_DCC_input,
		   'ECAL_SRP'=>\&define_ECAL_SRP,

		#endcap mappings
                   'ECAL_crystal_number_fedccuxt_to_EE_crystal_number' => \&define_ECAL_crystal_number_fedccuxt_to_EE_crystal_number,
		   'EE_readout_tower_to_EE_crystal_number'=>\&define_EE_readout_tower_to_EE_crystal_number,
		   'EE_trigger_tower_to_EE_crystal_number'=>\&define_EE_trigger_tower_to_EE_crystal_number,
		   'EE_crystal_number_to_EE_trigger_tower'=>\&define_EE_crystal_number_to_EE_trigger_tower,
		   'EE_crystal_number_to_EE_readout_tower'=>\&define_EE_crystal_number_to_EE_readout_tower,
		   'EE_crystal_number_to_EE_readout_strip'=>\&define_EE_crystal_number_to_EE_readout_strip,
		   'EE_crystal_readout_strip_to_EE_crystal_number'=>\&define_EE_crystal_readout_strip_to_EE_crystal_number,
		   'EE_crystal_trigger_strip_to_EE_crystal_number'=>\&define_EE_crystal_trigger_strip_to_EE_crystal_number,
		   'EE_crystal_number_to_EE_LV_channel'		   => \&define_EE_crystal_number_to_EE_LV_channel,
		   'EE_crystal_number_to_EE_HV_channel'		   => \&define_EE_crystal_number_to_EE_HV_channel,
		   'EE_readout_tower_xyz_to_EE_readout_tower'=>\&define_EE_readout_tower_xyz_to_EE_readout_tower,
		   'EE_readout_tower_xyz_to_EE_HV_channel'=>\&define_EE_readout_tower_xyz_to_EE_HV_channel,
		   'EE_readout_tower_xyz_to_EE_HVA_channel'=>\&define_EE_readout_tower_xyz_to_EE_HVA_channel,
		   'EE_readout_tower_xyz_to_EE_HVD_channel'=>\&define_EE_readout_tower_xyz_to_EE_HVD_channel,
		   'EE_readout_tower_xyz_to_EE_LV_channel'=>\&define_EE_readout_tower_xyz_to_EE_LV_channel,
		   'EE_crystal_number_to_EE_sector'		   => \&define_EE_crystal_number_to_EE_sector,
		   'EE_sector_to_EE_crystal_number'		   => \&define_EE_sector_to_EE_crystal_number,
		   'EE_readout_tower_zseccu_to_EE_readout_tower'=>\&define_EE_readout_tower_zseccu_to_EE_readout_tower,
		   'EE_readout_strip_zseccu_to_EE_readout_strip'=>\&define_EE_readout_strip_zseccu_to_EE_readout_strip ,
		   'EE_crystal_number_to_EE_LM_LMM'	   => \&define_EE_crystal_number_to_EE_LM_LMM,
		   'EE_crystal_number_to_EE_LM_PN'		   => \&define_EE_crystal_number_to_EE_LM_PN,
		   'EE_crystal_number_to_ECAL_LMR'	   => \&define_EE_crystal_number_to_ECAL_LMR,
                   'EE_trigger_tower_to_EE_offline_towerid' => \&define_EE_trigger_tower_to_EE_offline_towerid,
                   'EE_trigger_tower_to_EE_offline_towerid_more' => \&define_EE_trigger_tower_to_EE_offline_towerid_more,
		   'EE_trigger_strip_to_EE_offline_stripid' => \&define_EE_trigger_strip_to_EE_offline_stripid,
		   'ECAL_readout_strip_to_EE_readout_strip' => \&define_ECAL_readout_strip_to_EE_readout_strip,
		   'ECAL_readout_strip_to_EE_offline_stripid' => \&define_ECAL_readout_strip_to_EE_offline_stripid,
		   'ECAL_readout_strip_to_EB_VFE' => \&define_ECAL_readout_strip_to_EB_VFE

	 };

  bless($this, $class);
  return $this;
}


sub define {
  my $this = shift;
  my $def = shift;

  if (exists $this->{defs}->{$def}) {
    $this->import_def( &{$this->{defs}->{$def}} );
  } else {
    warn "No such definition:  $def\n";
  }
}

sub define_all {
  my $this = shift;

  foreach my $def (keys %{$this->{defs}}) {
    $this->import_def( &{ $this->{defs}->{$def} } );
  }
}

sub import_def {
  my $this = shift;
  my $def = shift;
  unless ($def->{logic_ids} && $def->{channel_ids}) {
    die "ERROR:  import_def() needs logic_ids and channel_ids!\n";
  }

  my $condDB = $this->{condDB};


  my $logic_ids = $def->{logic_ids};
  my $channel_ids = $def->{channel_ids};
  my $count = scalar @{$logic_ids};

  print "\t$def->{name}:  $count channels...";
  $condDB->begin_work();

  $def->{maps_to} = $def->{name} unless defined $def->{maps_to};

  if ($def->{description} || $def->{idnames}) {
    $condDB->new_channelView_type(-name => $def->{name},
				  -description => $def->{description},
				  -idnames => $def->{idnames},
				  -maps_to => $def->{maps_to});
  }

  for my $i (0..$count-1) {
  #  print "\t\t$i inserting ".$$logic_ids[$i]."\n";
    $condDB->insert_channel(-name => $def->{name},
			    -maps_to =>$def->{maps_to},
			    -channel_ids => $$channel_ids[$i],
			    -logic_id => $$logic_ids[$i]
			   );
  }
  $condDB->commit();
  print "Done.\n";
}

sub define_dummy {
  my $name = "DUMMY";
  my $idnames = [];
  my $description = "A dummy logic_id for testing purposes";
  
  my @logic_ids = (-1);
  my @channel_ids = ([]);

  return {name => $name, idnames => $idnames,
	  description => $description, logic_ids => \@logic_ids,
	 channel_ids => \@channel_ids};
}

sub define_EE_side {

	my $name = "EE_side";
	my $idnames = ["side"];
	my $description = "Endcap side (wrt z axis)";

	my @channel_ids;
	my @logic_ids;

	#side = 2: EE+ - side = 0: EE- in logicId assignment.
	
	#FIXME hardwired?
	push @logic_ids, 2000000012;
	push @logic_ids, 2000000010;

	push @channel_ids, [1];
	push @channel_ids, [-1];

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_EE_D {

	my $name = "EE_D";
	my $idnames = ["D"];
	my $description = "Endcap Dee";

	my @channel_ids;
	my @logic_ids;

	foreach my $D (1..4) {
		my $logic_id = sprintf "20000001%02d", $D;
		push @logic_ids, $logic_id;
		push @channel_ids, [$D];
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}


sub define_EE_sector {

	my $name = "EE_sector";
	my $idnames = ["side", "sector"];
	my $description = "Data Sectors by number in Endcap";

	my @channel_ids;
	my @logic_ids;

	foreach my $side (1,-1) {
		my $sideIndex = $side + 1;
		foreach my $sector (1..9) {
			my $logic_id = sprintf "2000001%01d%02d", $sideIndex, $sector;
			push @logic_ids, $logic_id;
			push @channel_ids, [$side, $sector];
		}
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_EE_DCC {

	my $name = "EE_DCC";
	my $idnames = ["DCC"];
	my $description = "DCC sectors by number in Endcap";

	my @channel_ids;
	my @logic_ids;

	#EE-
	foreach my $DCC (601..609) {
		my  $logic_id = sprintf "2000001%03d", $DCC;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC];
	}
	#EE+
	foreach my $DCC (646..654) {
		my $logic_id = sprintf "2000001%03d", $DCC;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC];
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_EE_crystal_number {

	my $name = "EE_crystal_number";
	my $idnames = ["side", "ix", "iy"];
	my $description = "Crystals in Ecal Endcap by number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $ix;
	my $iy;
	my $side;

	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;

		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;
		push @channel_ids, [$side, $ix, $iy];
	}

	close(FILE);

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_ECAL_TCC {

	my $name = "ECAL_TCC";
	my $idnames = ["TCC"];
	my $description = "TCC sectors by number in ECAL";

	my @channel_ids;
	my @logic_ids;

	#whole ECAL
	foreach my $TCC (1..108) {
		my  $logic_id = sprintf "1%03d", $TCC;
		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_ECAL_DCC {

	my $name = "ECAL_DCC";
	my $idnames = ["DCC"];
	my $description = "DCC number in ECAL";

	my @channel_ids;
	my @logic_ids;

	#whole ECAL
	foreach my $DCC (601..654) {
		my  $logic_id = sprintf "2%03d", $DCC;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_ECAL_DCC_input {

	my $name = "ECAL_DCC_input";
	my $idnames = ["DCC", "channel"];
	my $description = "DCC input channel MEM 1-2 SRP 3 TCC 4-7";

	my @channel_ids;
	my @logic_ids;

	#whole ECAL
	foreach my $DCC (601..654) {
	    foreach my $chan (1..7) {
		my  $logic_id = sprintf "%03d%01d", $DCC,$chan;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC,$chan];
	    }
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_ECAL_SRP {

	my $name = "ECAL_SRP";
	my $idnames = ["SRP"];
	my $description = "SRP number in ECAL";

	my @channel_ids;
	my @logic_ids;

	#whole ECAL
	foreach my $SRP (1..54) {
		my  $logic_id = sprintf "2%03d", $SRP;
		push @logic_ids, $logic_id;
		push @channel_ids, [$SRP];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}


sub define_EE_crystal_hashed {

	my $name = "EE_crystal_hashed";
	my $idnames = ["hi"];
	my $description = "Crystals in Ecal Endcap by hashed index";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $hi;
	
	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$hi = $channels[3];

		#2 02X XXX XXX: avoid possible (but not happening anyways) overlapping with crystal_number when 1st free digit is 0
		my $logic_id = sprintf "202%07d", $hi;
		push @logic_ids, $logic_id;
		push @channel_ids, [$hi];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_readout_tower {

	my $name = "EE_readout_tower";
	my $idnames = ["DCC","readout_tower"];
	my $description = "Readout Towers in the ECAL Endcap";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =DCC-600 ROTower;
	        my $id = sprintf "%03d %02d", $channels[4],$channels[5];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$DCC = $channels[0] + 600;
		$readout_tower = $channels[1];

		my $logic_id = sprintf "21100%03d%02d", $DCC, $readout_tower;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC, $readout_tower];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_trigger_tower {

	my $name = "EE_trigger_tower";
	my $idnames = ["TCC","trigger_tower"];
	my $description = "Trigger Towers in the ECAL Endcap";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =DCC.TTower;
	        my $id = sprintf "%03d %02d", $channels[8],$channels[9];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$TCC = $channels[0];
		$trigger_tower = $channels[1];

		my $logic_id = sprintf "21000%03d%02d", $TCC, $trigger_tower;
		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC, $trigger_tower];
	}

# additional towers for Pascal inner sectors 

	my @TT_new1 = (21,23,25,27);
	foreach my $TCC_m (1..18) {
	    foreach my $TT_m (@TT_new1) {
		my  $logic_id = sprintf "21000%03d%02d", $TCC_m, $TT_m;
		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC_m, $TT_m];
	    }
	}

	my @TT_new2 = (22,24,26,28);
	foreach my $TCC_m (91..108) {
	    foreach my $TT_m (@TT_new2) {
		my  $logic_id = sprintf "21000%03d%02d", $TCC_m, $TT_m;
		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC_m, $TT_m];
	    }
	}


	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_readout_strip {

	my $name = "EE_readout_strip";
	my $idnames = ["DCC","readout_tower","readout_strip"];
	my $description = "Readout Strips in the ECAL Endcap";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $readout_strip;
	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =DCC TTower readout_strip;
	        my $id = sprintf "%03d %02d %01d", $channels[4],$channels[5],$channels[6];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$DCC = $channels[0];
		$readout_tower = $channels[1];
		$readout_strip = $channels[2];
		my $logic_id = sprintf "2121%03d%02d%01d", $DCC, $readout_tower, $readout_strip;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC, $readout_tower,$readout_strip];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_readout_strip_zseccu {

	my $name = "EE_readout_strip_zseccu";
	my $idnames = ["z_x_sector","ccu_id","readout_strip"];
	my $description = "Readout Strips in the EE by sector and ccu";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $readout_tower;
	my $readout_strip;
	my @ids;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	
		my $DCC=$channels[4];
		my $sector=0;
		if($DCC<10) {
		    if($DCC<=3){
			$sector=$DCC+6;
		    } else {
			$sector=$DCC-3;
		    } 
		}else {
		    if($DCC<=48){
			$sector=$DCC-39;
		    } else {
			$sector=$DCC-48;
		    }
		} 
		my $ccu_id=$channels[13];

		my $sideIndex=$channels[2]+1;
		

	        #id =DCC TTower readout_strip;
	        my $id = sprintf "%01d %01d %02d %01d",$sideIndex, $sector, $ccu_id, $channels[6];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		my $sideIndex = $channels[0];
		my $sector= $channels[1];
		my $ccu_id= $channels[2];
		my $strip= $channels[3];
		my $side= $sideIndex-1;
		my $logic_id = sprintf "213%01d%02d%02d%02d", $sideIndex, $sector, $ccu_id, $strip;
		push @logic_ids, $logic_id;
		$sector=$sector*$side;
		push @channel_ids, [$sector, $ccu_id, $strip];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_trigger_strip {

	my $name = "EE_trigger_strip";
	my $idnames = ["TCC","trigger_tower","trigger_strip"];
	my $description = "Trigger Strips in the ECAL Endcap";
	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my $trigger_strip;
	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =TCC TriggerTower trigger_strip;
	        my $id = sprintf "%03d %02d %01d", $channels[8],$channels[9],$channels[10];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$TCC = $channels[0];
		$trigger_tower = $channels[1];
		$trigger_strip = $channels[2];
		my $logic_id = sprintf "2140%03d%02d%01d", $TCC, $trigger_tower, $trigger_strip;
		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC, $trigger_tower,$trigger_strip];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}


sub define_EE_crystal_trigger_strip {

	my $name = "EE_crystal_trigger_strip";
	#crystal_trigger_strip_id is a combination of other ids
	my $idnames = ["TCC","trigger_tower","crystal_trigger_strip_id"];
	my $description = "Numbering of crystal in Trigger Strip in the ECAL Endcap";
	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my $trigger_strip;
	my $crystal_trigger_strip;
	my $crystal_trigger_strip_id;

	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        $TCC = $channels[8];
		$trigger_tower= $channels[9];
		$trigger_strip= $channels[10];
		$crystal_trigger_strip=$channels[11];

		#crystal_trigger_strip_id=(strip-1)*5+channel_in_strip
		$crystal_trigger_strip_id=($trigger_strip-1)*5+$crystal_trigger_strip;

		my $logic_id = sprintf "203%03d%02d%02d", $TCC, $trigger_tower, $crystal_trigger_strip_id;
		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC, $trigger_tower,$crystal_trigger_strip_id];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_crystal_readout_strip {

	my $name = "EE_crystal_readout_strip";
	#crystal_readout_strip_id is a combination of other ids
	my $idnames = ["DCC","readout_tower","crystal_readout_strip_id"];
	my $description = "Numbering of crystal in Readout Strip in the ECAL Endcap";
	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $readout_strip;
	my $crystal_readout_strip;
	my $crystal_readout_strip_id;

	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        $DCC = $channels[4] + 600;
		$readout_tower= $channels[5];
		$readout_strip= $channels[6];
		$crystal_readout_strip=$channels[7];

		#crystal_readout_strip_id=(strip-1)*5+channel_in_strip
		$crystal_readout_strip_id=($readout_strip-1)*5+$crystal_readout_strip;

		my $logic_id = sprintf "204%03d%02d%02d", $DCC, $readout_tower, $crystal_readout_strip_id;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC, $readout_tower,$crystal_readout_strip_id];
	}
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};
}

sub define_EE_elec_crystal_number {

	my $name = "EE_elec_crystal_number";
	my $idnames = ["ele_number"];
	my $description = "Crystals in Ecal Endcap by electronic number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $readout_strip;
	my $crystal_readout_strip;
	my $ele_number;
	
	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$DCC = $channels[4];
		$readout_tower = $channels[5];
		$readout_strip = $channels[6];
		$crystal_readout_strip = $channels[7];
		
		#ele_number = 10 000 * DCC_id (<--FED-600) + readout_tower * 100 + (strip-1) * 5 + ch_in_sttip
		$ele_number = $DCC*10000 + $readout_tower*100 + ($readout_strip-1)*5 + $crystal_readout_strip;

		my $logic_id = sprintf "205%07d", $ele_number;
		push @logic_ids, $logic_id;
		push @channel_ids, [$ele_number];
	}

	close(FILE);

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}


sub define_EE_HV_channel {

	my $name = "EE_HV_channel";
	my $idnames = ["dee", "channel"];
	my $description = "HV channel by Dee and chan in the Endcaps";

	my @channel_ids;
	my @logic_ids;


	foreach my $D (1..4) {
	    foreach my $chan (1..13) {
		my $logic_id = sprintf "20600%02d0%02d", $D, $chan;
			push @logic_ids, $logic_id;
			push @channel_ids, [$D, $chan];
		}
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}
sub define_EE_HVA_channel {

	my $name = "EE_HVA_channel";
	my $idnames = ["dee", "channel"];
	my $description = "HV anode by Dee and chan in the Endcaps";

	my @channel_ids;
	my @logic_ids;


	foreach my $D (1..4) {
	    foreach my $chan (1..2) {
		my $logic_id = sprintf "20604%02d0%02d", $D, $chan;
			push @logic_ids, $logic_id;
			push @channel_ids, [$D, $chan];
		}
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_EE_HVD_channel {

	my $name = "EE_HVD_channel";
	my $idnames = ["dee", "channel"];
	my $description = "HV dynode by Dee and chan in the Endcaps";

	my @channel_ids;
	my @logic_ids;


	foreach my $D (1..4) {
	    foreach my $chan (1..2) {
		my $logic_id = sprintf "20605%02d0%02d", $D, $chan;
			push @logic_ids, $logic_id;
			push @channel_ids, [$D, $chan];
		}
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_EE_LMM {

	my $name = "EE_LMM";
	my $idnames = ["dee", "lmm"];
	my $description = "Endcap Laser Monitoring Modules";

	my @channel_ids;
	my @logic_ids;


	foreach my $D (1..4) {
	    foreach my $chan (1..19) {
		my $logic_id = sprintf "20601%02d0%02d", $D, $chan;
			push @logic_ids, $logic_id;
			push @channel_ids, [$D, $chan];
		}
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_EE_LM_PN {

	my $name = "EE_LM_PN";
	my $idnames = ["dee", "PN"];
	my $description = "Endcap PN";

	my @channel_ids;
	my @logic_ids;


	foreach my $D (1..4) {
	    foreach my $chan (0..9) {
		foreach my $side (1..2) {
		    my $channel=$side*100+$chan;
		    my $logic_id = sprintf "20602%02d%03d", $D,$channel;
			push @logic_ids, $logic_id;
			push @channel_ids, [$D, $channel];
		}
	    }
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}

sub define_ECAL_LMR {

	my $name = "ECAL_LMR";
	my $idnames = ["channel"];
	my $description = "EcalLaserMonitoringRegion EB-EB+EE+EE-";

	my @channel_ids;
	my @logic_ids;


	    foreach my $chan (1..92) {
		my $logic_id="";
		if($chan<=72){
		    $logic_id = sprintf "10603000%02d", $chan;
		} else {
		    $logic_id = sprintf "20603000%02d", $chan;
		}		
		push @logic_ids, $logic_id;
		push @channel_ids, [$chan];
	    }
	    
	
	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}
sub define_EE_PTM_H_amb {
  my $name = "EE_PTM_H_amb";
  my $idnames = ["Dee", "channel"];
  my $description = "ECAL Endcap Ambient humidity sensors";

  my @logic_ids;
  my @channel_ids;
  foreach my $Dee (1..4) {
    foreach my $channel (1..4) {
      my $logic_id = sprintf "211020%01d0%02d", $Dee, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $Dee, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EE_PTM_T_amb {
  my $name = "EE_PTM_T_amb";
  my $idnames = ["Dee", "channel"];
  my $description = "ECAL Endcap Ambient temperature sensors";

  my @logic_ids;
  my @channel_ids;
  foreach my $Dee (1..4) {
    foreach my $channel (1..24) {
      my $logic_id = sprintf "211010%01d0%02d", $Dee, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $Dee, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}


sub define_EE_LV_channel {

	my $name = "EE_LV_channel";
	my $idnames = ["dee", "channel"];
	my $description = "LV channel by Dee and chan in the Endcaps";

	my @channel_ids;
	my @logic_ids;


	foreach my $D (1..4) {
	    foreach my $chan (10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88) {
		my $logic_id = sprintf "20700%02d0%02d", $D, $chan;
			push @logic_ids, $logic_id;
			push @channel_ids, [$D, $chan];
		}
	}

	return {name => $name, idnames => $idnames,
		description => $description, logic_ids => \@logic_ids,
		channel_ids => \@channel_ids};

}


sub define_EE_readout_tower_xyz {

    my $name = "EE_readout_tower_xyz";
    my $idnames = ["iz","ix","iy"];
    my $description = "Readout Towers in the Endcaps by zxy";

    my @channel_ids;
    my @logic_ids;

        #opening file
    open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
        #reading it into an array
    my @lines = <FILE>;
        #getting the first line out
    shift @lines;

        #temp variables
    my $ix;
    my $iy;
    my $iz;

    my $DCC;
    my $readout_tower;

    my @ids;
    my @ixs;
    my @iys;
    my @izs;

    my $count=-1;
    foreach my $line (@lines) {

	my @channels = split (/ /, $line);

	$ix= $channels[0];
	$iy= $channels[1];
	$iz= $channels[2];


                #id =DCC-600 ROTower;
	$DCC=$channels[4]+600;
	my $id = sprintf "20800%03d%02d", $DCC,$channels[5];
	my $ifound=-1;
	for my $i (0..$count) {
	    if($id== $ids[$i]) {
		$ifound=$i;
		if( $ix<= $ixs[$i] && $iy<= $iys[$i] ) {
		    $ixs[$i]=$ix;
		    $iys[$i]=$iy;
		}

		last;
	    }
	}
    
	if($ifound==-1) {
	    $count++;
	    push @ids, $id;
	    push @ixs, $ix;
	    push @iys, $iy;
	    push @izs, $iz;
	}

    }

    for my $i (0..$count) {
	push @logic_ids, $ids[$i];
	push @channel_ids, [$izs[$i], $ixs[$i], $iys[$i]];
    }

    return {name => $name, idnames => $idnames,
                description => $description, logic_ids => \@logic_ids,
	    channel_ids => \@channel_ids};
}

sub define_EE_readout_tower_zseccu {

    my $name = "EE_readout_tower_zseccu";
    my $idnames = ["iz","sector","ccuid"];
    my $description = "Readout Towers in the EE by z sector ccuid";

    my @channel_ids;
    my @logic_ids;

        #opening file
    open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
        #reading it into an array
    my @lines = <FILE>;
        #getting the first line out
    shift @lines;

        #temp variables
    my $ix;
    my $iy;
    my $iz;


    my @ids;
    my @ixs;
    my @iys;
    my @izs;

    my $count=-1;
    foreach my $line (@lines) {

	my @channels = split (/ /, $line);

	$ix= $channels[0];
	$iy= $channels[1];
	$iz= $channels[2];

	my $sideIndex = $channels[2] + 1;

        
	my $DCC=$channels[4];
	my $sector=0;
	if($DCC<10) {
	    if($DCC<=3){
		$sector=$DCC+6;
	    } else {
		$sector=$DCC-3;
	    } 
	}else {
	    if($DCC<=48){
		$sector=$DCC-39;
	    } else {
		$sector=$DCC-48;
	    }
	} 
	my $ccu_id=$channels[13];


	my $id = sprintf "209%02d%03d%02d", $sideIndex, $sector,$ccu_id;
	my $ifound=-1;
	for my $i (0..$count) {
	    if($id== $ids[$i]) {
		$ifound=$i;
		last;
	    }
	}
    
	if($ifound==-1) {
	    $count++;
	    push @ids, $id;
	    push @ixs, $sector;
	    push @iys, $ccu_id;
	    push @izs, $iz;
	}

    }

    for my $i (0..$count) {
	push @logic_ids, $ids[$i];
	push @channel_ids, [$izs[$i], $ixs[$i], $iys[$i]];
    }

    return {name => $name, idnames => $idnames,
                description => $description, logic_ids => \@logic_ids,
	    channel_ids => \@channel_ids};
}



sub define_ECAL {
  my $name = "ECAL";
  my $idnames = [];
  my $description = "The entire ECAL subdetector";
  
  my @logic_ids = (1);
  my @channel_ids = ([]);

  return {name => $name, idnames => $idnames,
	  description => $description, logic_ids => \@logic_ids,
	 channel_ids => \@channel_ids};
}

sub define_EB {
  my $name = "EB";
  my $idnames = [];
  my $description = "The entire ECAL Barrel";  
  my @logic_ids = (1000000000);
  my @channel_ids = ([]);

  return {name => $name, idnames => $idnames,
	  description => $description, logic_ids => \@logic_ids,
	 channel_ids => \@channel_ids};
}

sub define_EE {
  my $name = "EE";
  my $idnames = [];
  my $description = "The entire ECAL Endcap";  
  my @logic_ids = (2000000000);
  my @channel_ids = ([]);

  return {name => $name, idnames => $idnames,
	  description => $description, logic_ids => \@logic_ids,
	 channel_ids => \@channel_ids};
}

sub define_EB_crystal_number {
  my $name = "EB_crystal_number";
  my $idnames = ["SM", "crystal_number"];
  my $description = "Crystals in ECAL barrel super-modules by number";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $xtal (1..1700) {
      my $logic_id = sprintf "1011%02d%04d", $SM, $xtal;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $xtal ];
    }
  }

  return {name => $name, idnames => $idnames,
	  description => $description, logic_ids => \@logic_ids, 
	  channel_ids => \@channel_ids};
}



sub define_EB_crystal_index {
  my $name = "EB_crystal_index";
  my $idnames = ["SM", "i", "j"];
  my $description = "Crystals in ECAL barrel super-modules by i,j index";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $i (0..84) {
      foreach my $j (0..19) {
	my $logic_id = sprintf "1012%02d%02d%02d", $SM, $i, $j;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $SM, $i, $j ];
      }
    }
  }

  return {name => $name, idnames => $idnames,
	  description => $description, logic_ids => \@logic_ids,
	 channel_ids => \@channel_ids};
}

sub define_EB_elec_crystal_number {
  my $name = "EB_elec_crystal_number";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel crystals, electronics numbering scheme";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $xtal (0..1699) {
      my $logic_id = sprintf "1013%02d%04d", $SM, $xtal;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $xtal ];
    }
  }

  return {name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}


sub define_EB_fe_crystal_number {
  my $name = "EB_fe_crystal_number";
  my $idnames = ["SM", "TT", "channel"];
  my $description = "ECAL Barrel crystals, front-end configuration numbering scheme";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $TT (1..68) {
      foreach my $xtal (0..24) {
	my $logic_id = sprintf "1014%02d%02d%02d", $SM, $TT, $xtal;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $SM, $TT, $xtal ];
      }
    }
  }

  return {name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}

sub define_ECAL_crystal_number_fedccuxt {
  my $name = "ECAL_crystal_number_fedccuxt";
  my $idnames = ["FED", "CCU", "channel"];
  my $description = "ECAL Barrel crystals by fed ccu crystal(0-24)";

  my @logic_ids;
  my @channel_ids;
  foreach my $FED (610..645) {
    foreach my $TT (1..68) {
      foreach my $xtal (0..24) {
	my $logic_id = sprintf "120%03d%02d%02d", $FED, $TT, $xtal;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $FED, $TT, $xtal ];
      }
    }
  }

# endcap part
  #opening file
  open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
  #reading it into an array
  my @lines = <FILE>;
  #getting the first line out	
  shift @lines;
  
  #temp variables
  my $DCC;
  my $readout_tower;
  my $readout_strip;
  my $crystal_readout_strip;
  my $crystal_readout_strip_id;
  
  foreach my $line (@lines) {

      my @channels = split (/ /, $line);
      
      $DCC = $channels[4] + 600;
      $readout_tower= $channels[5];
      $readout_strip= $channels[6];
      $crystal_readout_strip=$channels[7];
      
      
      $crystal_readout_strip_id=($readout_strip-1)*5+$crystal_readout_strip-1;
      
      my $logic_id = sprintf "120%03d%02d%02d", $DCC, $readout_tower, $crystal_readout_strip_id;
      push @logic_ids, $logic_id;
      push @channel_ids, [$DCC, $readout_tower,$crystal_readout_strip_id];
  }


  return {name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}


sub define_EB_trigger_tower {
  my $name = "EB_trigger_tower";
  my $idnames = ["SM", "trigger_tower"];
  my $description = "Trigger towers in the ECAL barrel super-modules";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $tt (1..68) {
      my $logic_id = sprintf "1021%02d00%02d", $SM, $tt;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $tt ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_ECAL_trigger_tower {
  my $name = "ECAL_trigger_tower";
  my $idnames = ["TCC", "trigger_tower"];
  my $description = "Trigger towers in the ECAL barrel by tcc and tt";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (37..72) {
    foreach my $tt (1..68) {
      my $logic_id = sprintf "122%03d00%02d", $SM, $tt;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $tt ];
    }
  }



  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}
sub define_EB_readout_tower {
  my $name = "EB_readout_tower";
  my $idnames = ["FED", "tower"];
  my $description = "Readout towers in the ECAL barrel by fed and tower";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (610..645) {
    foreach my $tt (1..68) {
      my $logic_id = sprintf "123%03d00%02d", $SM, $tt;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $tt ];
    }
  }



  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_module {
  my $name = "EB_module";
  my $idnames = ["SM", "M"];
  my $description = "ECAL barrel modules";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $M (1..4) {
      my $logic_id = sprintf "1031%02d00%02d", $SM, $M;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $M ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_supermodule {
  my $name = "EB_supermodule";
  my $idnames = ["SM"];
  my $description = "ECAL barrel Super-Modules";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    my $logic_id = sprintf "1041%02d00%02d", $SM, $SM;
    push @logic_ids, $logic_id;
    push @channel_ids, [ $SM ];
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_HV_channel {
  my $name = "EB_HV_channel";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel High Voltage Channel";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..34) {
      my $logic_id = sprintf "1051%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }
  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_HV_board {
  my $name = "EB_HV_board";
  my $idnames = ["SM", "board#"];
  my $description = "ECAL Barrel High Voltage Board Number";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $board (0,2,4,6) {
      my $logic_id = sprintf "1061%02d00%02d", $SM, $board;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $board ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_LV_channel {
  my $name = "EB_LV_channel";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel Low Voltage Channel";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..17) {
      my $logic_id = sprintf "1071%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_ESS_temp {
  my $name = "EB_ESS_temp";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel ESS temperature channel";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (0..7) {
      my $logic_id = sprintf "1081%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_PTM_H_amb {
  my $name = "EB_PTM_H_amb";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel Ambient humidity sensors";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..4) {
      my $logic_id = sprintf "1091%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_PTM_T_amb {
  my $name = "EB_PTM_T_amb";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel Ambient temperature sensors";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..10) {
      my $logic_id = sprintf "1101%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_token_ring {
  my $name = "EB_token_ring";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel Token Ring";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..8) {
      my $logic_id = sprintf "1111%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}



sub define_EB_LM_side {
  my $name = "EB_LM_side";
  my $idnames = ["SM", "side"];
  my $description = "ECAL Barrel Laser Regions";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (0..1) {
      my $logic_id = sprintf "1132%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_LM_channel {
  my $name = "EB_LM_channel";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel Laser Monitoring";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..9) {
      my $logic_id = sprintf "1121%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_LM_PN {
  my $name = "EB_LM_PN";
  my $idnames = ["SM", "channel"];
  my $description = "ECAL Barrel Laser Monitoring PN numbers";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (0..9) {
      my $logic_id = sprintf "1131%02d00%02d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_T_capsule {
  my $name = "EB_T_capsule";
  my $idnames = ["SM", "channel"];
  my $description = "Ecal Barrel Capsule temperature sensors";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $channel (1..170) {
      my $logic_id = sprintf "1141%02d0%03d", $SM, $channel;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $channel ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}

sub define_EB_VFE {
  my $name = "EB_VFE";
  my $idnames = ["SM", "TT", "VFE#"];
  my $description = "Ecal Barrel Very Front End cards";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $TT (1..68) {
      foreach my $VFE (1..5) {
	my $logic_id = sprintf "1151%02d%02d%02d", $SM, $TT, $VFE;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $SM, $TT, $VFE ];
      }
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}

sub define_ECAL_readout_strip {
  my $name = "ECAL_readout_strip";
  my $idnames = ["FED", "CCU", "strip"];
  my $description = "Ecal readout strip by fed ccu strip";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (610..645) {
    foreach my $TT (1..68) {
      foreach my $VFE (1..5) {
	my $logic_id = sprintf "121%03d%02d%02d", $SM, $TT, $VFE;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $SM, $TT, $VFE ];
      }
    }
  }


# endcap strips 

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $readout_strip;
	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =DCC TTower readout_strip;
		my $FED=$channels[4]+600;
	        my $id = sprintf "%03d %02d %02d", $FED,$channels[5],$channels[6];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$DCC = $channels[0];
		$readout_tower = $channels[1];
		$readout_strip = $channels[2];
		my $logic_id = sprintf "121%03d%02d%02d", $DCC, $readout_tower, $readout_strip;
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC, $readout_tower,$readout_strip];
	}

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}

sub define_EB_LVRB_DCU {
  my $name = "EB_LVRB_DCU";
  my $idnames = ["SM", "TT", "LVRB_DCU#"];
  my $description = "Ecal Barrel DCUs on the LVRB";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $TT (1..68) {
      foreach my $DCU (1..3) {
	my $logic_id = sprintf "1161%02d%02d%02d", $SM, $TT, $DCU;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $SM, $TT, $DCU ];
      }
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}

sub define_EB_LVRB_T_sensor {
  my $name = "EB_LVRB_T_sensor";
  my $idnames = ["SM", "TT", "T_sens#"];
  my $description = "Ecal Barrel thermisters on the LVRB";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $TT (1..68) {
      foreach my $sens (1..3) {
	my $logic_id = sprintf "1171%02d%02d%02d", $SM, $TT, $sens;
	push @logic_ids, $logic_id;
	push @channel_ids, [ $SM, $TT, $sens ];
      }
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}

sub define_EB_mem_TT {
  my $name = "EB_mem_TT";
  my $idnames = ["SM", "TT"];
  my $description = "Supermodule mem box pseudo trigger tower";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $TT (69..70) {
      my $logic_id = sprintf "1181%02d00%02d", $SM, $TT;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $TT ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}

sub define_EB_mem_channel {
  my $name = "EB_mem_channel";
  my $idnames = ["SM", "channel"];
  my $description = "Supermodule mem box pseudo channel";

  my @logic_ids;
  my @channel_ids;
  foreach my $SM (0..36) {
    foreach my $ch (1..50) {
      my $logic_id = sprintf "1191%02d00%02d", $SM, $ch;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $ch ];
    }
  }

  return {
	  name => $name, idnames => $idnames,
	  description => $description,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 }
}



###
###   Cross-Channel mappings
###

sub define_EE_crystal_number_to_EE_trigger_tower {

	my $name = "EE_crystal_number";
	my $maps_to = "EE_trigger_tower";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my $ix;
	my $iy;
	my $side;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];

		$TCC = $channels[8];
		$trigger_tower = $channels[9];

		#xtal number ids: side, ix, iy
		push @channel_ids, [$side, $ix, $iy];

		#trigger tower logic id: 20000XXXYY XXX=TCC, YY=trigger tower
		$logic_id = sprintf "21000%03d%02d", $TCC, $trigger_tower;
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}



sub define_EE_trigger_tower_to_EE_crystal_number {

	my $name = "EE_trigger_tower";	
	my $maps_to = "EE_crystal_number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my $ix;
	my $iy;
	my $sideIndex;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$sideIndex = $channels[2] + 1;

		$TCC = $channels[8];
		$trigger_tower = $channels[9];

		#trigger tower ids: TCC, trigger tower
		push @channel_ids, [$TCC, $trigger_tower];

		#xtal number logic id: 201XYYYZZZ XX=TCC, YY=trigger tower
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}


sub define_EE_trigger_tower_to_EE_offline_towerid {

	my $name = "EE_trigger_tower";	
	my $maps_to = "EE_offline_towerid";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $offline_id;
	my $trigger_tower;
	my $ix;
	my $iy;
	my $sideIndex;
	my $logic_id;
	my @ids;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$sideIndex = $channels[2] + 1;

		$TCC = $channels[8];
		$trigger_tower = $channels[9];
		$offline_id = $channels[18];

	        my $id = sprintf "%03d %02d %9d", $channels[8],$channels[9], $channels[18];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$TCC           = $channels[0];
		$trigger_tower = $channels[1];
		$offline_id    = $channels[2];

		my $logic_id = sprintf "%9d", $offline_id;
		push @logic_ids, $logic_id;

		#trigger tower ids: TCC, trigger tower
		push @channel_ids, [$TCC, $trigger_tower];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}
sub define_EE_trigger_tower_to_EE_offline_towerid_more {

	my $name = "EE_trigger_tower";	
	my $maps_to = "EE_offline_towerid";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "additionaltowers.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $offline_id;
	my $trigger_tower;
	my $ix;
	my $iy;
	my $sideIndex;
	my $logic_id;
	my @ids;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);


		$TCC           = $channels[0];
		$trigger_tower = $channels[1];
		$offline_id    = $channels[2];

		my $logic_id = sprintf "%9d", $offline_id;
		push @logic_ids, $logic_id;

		#trigger tower ids: TCC, trigger tower
		push @channel_ids, [$TCC, $trigger_tower];


	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)
	
	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_trigger_strip_to_EE_offline_stripid {

	my $name = "EE_trigger_strip";	
	my $maps_to = "EE_offline_stripid";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my $trigger_strip;
	my @ids;
	my $offline_id ;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =TCC TriggerTower trigger_strip;
	        my $id = sprintf "%03d %02d %01d %06d", $channels[8],$channels[9],$channels[10], $channels[19];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$TCC = $channels[0];
		$trigger_tower = $channels[1];
		$trigger_strip = $channels[2];
		$offline_id= $channels[3];

		my $logic_id = sprintf "%6d", $offline_id;

		push @logic_ids, $logic_id;
		push @channel_ids, [$TCC, $trigger_tower,$trigger_strip];
	}


	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}


sub define_EE_crystal_number_to_EE_sector {

	my $name = "EE_crystal_number";
	my $maps_to = "EE_sector";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables

	my $ix;
	my $iy;
	my $side;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;
		my $DCC = $channels[4];
		my $sector=0;
		if($DCC<10) {
		    if($DCC<=3){
			$sector=$DCC+6;
		    } else {
			$sector=$DCC-3;
		    } 
		}else {
		    if($DCC<=48){
			$sector=$DCC-39;
		    } else {
			$sector=$DCC-48;
		    }
		} 
		my $logic_id = sprintf "2000001%01d%02d", $sideIndex, $sector;

		#xtal number ids: side, ix, iy
		push @channel_ids, [$side, $ix, $iy];
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}




sub define_EE_sector_to_EE_crystal_number {

	my $name = "EE_sector";	
	my $maps_to = "EE_crystal_number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables

	my $ix;
	my $iy;
	my $sideIndex;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		my $side=$channels[2];
		$sideIndex = $channels[2] + 1;

		my $DCC = $channels[4];
		my $sector=0;
		if($DCC<10) {
		    if($DCC<=3){
			$sector=$DCC+6;
		    } else {
			$sector=$DCC-3;
		    } 
		}else {
		    if($DCC<=48){
			$sector=$DCC-39;
		    } else {
			$sector=$DCC-48;
		    }
		} 

		#trigger tower ids: TCC, trigger tower
		push @channel_ids, [$side, $sector];

		#xtal number logic id: 201XYYYZZZ XX=TCC, YY=trigger tower
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_crystal_number_to_EE_readout_tower {

	my $name = "EE_crystal_number";
	my $maps_to = "EE_readout_tower";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];

		$DCC = $channels[4] + 600;
		$readout_tower = $channels[5];

		#readout tower logic id: 2000XXXYY XXX=DCC, YY=readout tower
		$logic_id = sprintf "21100%03d%02d", $DCC, $readout_tower;
		push @logic_ids, $logic_id;

		#xtal number ids: ix, iy, iz
		push @channel_ids, [$side, $ix, $iy];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_crystal_number_to_EE_readout_strip {

	my $name = "EE_crystal_number";
	my $maps_to = "EE_readout_strip";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $readout_strip;
	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];

		$DCC = $channels[4] ;
		$readout_tower = $channels[5];
		$readout_strip = $channels[6];
		my $logic_id = sprintf "2121%03d%02d%01d", $DCC, $readout_tower, $readout_strip;

		push @logic_ids, $logic_id;

		#xtal number ids: ix, iy, iz
		push @channel_ids, [$side, $ix, $iy];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_readout_tower_to_EE_crystal_number {


	my $name = "EE_readout_tower";
	my $maps_to = "EE_crystal_number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;
	my $ix;
	my $iy;
	my $side;
	my $sideIndex;

	my $logic_id;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$sideIndex = $channels[2] + 1;

		$DCC = $channels[4] + 600;
		$readout_tower = $channels[5];

		#trigger tower ids: DCC, trigger tower
		push @channel_ids, [$DCC, $readout_tower];

		#xtal number logic id: 201XYYYZZZ X=sideIndex, YYY=ix YYY=iy
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}


sub define_EE_readout_tower_xyz_to_EE_readout_tower {

	my $name = "EE_readout_tower_xyz";
    my $maps_to = "EE_readout_tower";

    my @channel_ids;
    my @logic_ids;

        #opening file
    open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
        #reading it into an array
    my @lines = <FILE>;
        #getting the first line out
    shift @lines;

        #temp variables
    my $ix;
    my $iy;
    my $iz;

    my $DCC;
    my $readout_tower;

    my @ids;
    my @ixs;
    my @iys;
    my @izs;

    my $count=-1;
    foreach my $line (@lines) {

	my @channels = split (/ /, $line);

	$ix= $channels[0];
	$iy= $channels[1];
	$iz= $channels[2];


                #id =DCC-600 ROTower;
	$DCC=$channels[4]+600;
	my $id = sprintf "21100%03d%02d", $DCC,$channels[5];
	my $ifound=-1;
	for my $i (0..$count) {
	    if($id== $ids[$i]) {
		$ifound=$i;
		if( $ix<= $ixs[$i] && $iy<= $iys[$i] ) {
		    $ixs[$i]=$ix;
		    $iys[$i]=$iy;
		}
		last;
	    }
	}
    
	if($ifound==-1) {
	    $count++;
	    push @ids, $id;
	    push @ixs, $ix;
	    push @iys, $iy;
	    push @izs, $iz;
	}

    }

    for my $i (0..$count) {
	push @logic_ids, $ids[$i];
	push @channel_ids, [$izs[$i], $ixs[$i], $iys[$i]];
    }

    return {		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}

sub define_EE_readout_tower_zseccu_to_EE_readout_tower {

	my $name = "EE_readout_tower_zseccu";
    my $maps_to = "EE_readout_tower";

    my @channel_ids;
    my @logic_ids;

        #opening file
    open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
        #reading it into an array
    my @lines = <FILE>;
        #getting the first line out
    shift @lines;

        #temp variables
    my $ix;
    my $iy;
    my $iz;

    my $DCC;
    my $readout_tower;

    my @ids;
    my @ixs;
    my @iys;
    my @izs;

    my $count=-1;
    foreach my $line (@lines) {

	my @channels = split (/ /, $line);

	$ix= $channels[0];
	$iy= $channels[1];
	$iz= $channels[2];


	my $sideIndex = $channels[2] + 1;

        
	$DCC=$channels[4];
	my $sector=0;
	if($DCC<10) {
	    if($DCC<=3){
		$sector=$DCC+6;
	    } else {
		$sector=$DCC-3;
	    } 
	}else {
	    if($DCC<=48){
		$sector=$DCC-39;
	    } else {
		$sector=$DCC-48;
	    }
	} 
	my $ccu_id=$channels[13];



                #id =DCC-600 ROTower;
	$DCC=$channels[4]+600;
	my $id = sprintf "21100%03d%02d", $DCC,$channels[5];
	my $ifound=-1;
	for my $i (0..$count) {
	    if($id== $ids[$i]) {
		$ifound=$i;
		last;
	    }
	}
    
	if($ifound==-1) {
	    $count++;
	    $sideIndex = $iz + 1;

	    push @ids, $id;
	    push @ixs, $sector;
	    push @iys, $ccu_id;
	    push @izs, $iz;
	}

    }

    for my $i (0..$count) {
	push @logic_ids, $ids[$i];
	push @channel_ids, [$izs[$i], $ixs[$i], $iys[$i]];
    }

    return {		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}

sub define_EE_readout_tower_xyz_to_EE_HV_channel {

    my $name = "EE_readout_tower_xyz";
    my $maps_to = "EE_HV_channel";

    my @channel_ids;
    my @logic_ids;

        #opening file
    open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
        #reading it into an array
    my @lines = <FILE>;
        #getting the first line out
    shift @lines;

        #temp variables
    my $ix;
    my $iy;
    my $iz;

    my $DCC;
    my $readout_tower;

    my @ids;
    my @ixs;
    my @iys;
    my @izs;
    my @logi;

    my $count=-1;
    foreach my $line (@lines) {

	my @channels = split (/ /, $line);

	$ix= $channels[0];
	$iy= $channels[1];
	$iz= $channels[2];
	my $Dee=$channels[14];
	my $hv=$channels[15];

                #id =DCC-600 ROTower;
	$DCC=$channels[4]+600;
	my $id = sprintf "20000%03d%02d", $DCC,$channels[5];
	my $logic_id = sprintf "20600%02d0%02d", $Dee, $hv;
	my $ifound=-1;
	for my $i (0..$count) {
	    if($id== $ids[$i]) {
		$ifound=$i;
		if( $ix<= $ixs[$i] && $iy<= $iys[$i] ) {
		    $ixs[$i]=$ix;
		    $iys[$i]=$iy;
		}
		last;
	    }
	}
    
	if($ifound==-1) {
	    $count++;
	    push @ids, $id;
	    push @ixs, $ix;
	    push @iys, $iy;
	    push @izs, $iz;
	    push @logi, $logic_id;
	}

    }

    for my $i (0..$count) {
	push @logic_ids, $logi[$i];
	push @channel_ids, [$izs[$i], $ixs[$i], $iys[$i]];
    }

    return {		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}

sub define_EE_readout_tower_xyz_to_EE_LV_channel {

	my $name = "EE_readout_tower_xyz";
    my $maps_to = "EE_LV_channel";

    my @channel_ids;
    my @logic_ids;

        #opening file
    open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
        #reading it into an array
    my @lines = <FILE>;
        #getting the first line out
    shift @lines;

        #temp variables
    my $ix;
    my $iy;
    my $iz;

    my $DCC;
    my $readout_tower;

    my @ids;
    my @ixs;
    my @iys;
    my @izs;
    my @logi;

    my $count=-1;
    foreach my $line (@lines) {

	my @channels = split (/ /, $line);

	$ix= $channels[0];
	$iy= $channels[1];
	$iz= $channels[2];
	my $Dee=$channels[14];
	my $lv=$channels[16];

                #id =DCC-600 ROTower;
	$DCC=$channels[4]+600;
	my $id = sprintf "20000%03d%02d", $DCC,$channels[5];
	my $logic_id = sprintf "20700%02d0%02d", $Dee, $lv;
	my $ifound=-1;
	for my $i (0..$count) {
	    if($id== $ids[$i]) {
		$ifound=$i;
		if( $ix<= $ixs[$i] && $iy<= $iys[$i] ) {
		    $ixs[$i]=$ix;
		    $iys[$i]=$iy;
		}
		last;
	    }
	}
    
	if($ifound==-1) {
	    $count++;
	    push @ids, $id;
	    push @ixs, $ix;
	    push @iys, $iy;
	    push @izs, $iz;
	    push @logi, $logic_id;
	}

    }

    for my $i (0..$count) {
	push @logic_ids, $logi[$i];
	push @channel_ids, [$izs[$i], $ixs[$i], $iys[$i]];
    }

    return {		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids};
}




sub define_EE_crystal_readout_strip_to_EE_crystal_number {

	my $name = "EE_crystal_readout_strip";
	my $maps_to = "EE_crystal_number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;	
	my $readout_strip;
	my $crystal_readout_strip;
	my $crystal_readout_strip_id;
	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;

		#EE_crystal_number logicId
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

	        $DCC = $channels[4] + 600;
		$readout_tower= $channels[5];
		$readout_strip= $channels[6];
		$crystal_readout_strip=$channels[7];

		#crystal_readout_strip_id=(strip-1)*5+channel_in_strip
		$crystal_readout_strip_id=($readout_strip-1)*5+$crystal_readout_strip;

		#crystal_readout_strip ids: DCC, RO Tower, crystal_readout_strip_id
		push @channel_ids, [$DCC, $readout_tower,$crystal_readout_strip_id];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_ECAL_crystal_number_fedccuxt_to_EE_crystal_number {

	my $name = "ECAL_crystal_number_fedccuxt";
	my $maps_to = "EE_crystal_number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $DCC;
	my $readout_tower;	
	my $readout_strip;
	my $crystal_readout_strip;
	my $crystal_readout_strip_id;
	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;

		#EE_crystal_number logicId
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

	        $DCC = $channels[4] + 600;
		$readout_tower= $channels[5];
		$readout_strip= $channels[6];
		$crystal_readout_strip=$channels[7];

		#crystal_readout_strip_id=(strip-1)*5+channel_in_strip
		$crystal_readout_strip_id=($readout_strip-1)*5+$crystal_readout_strip-1;

		#crystal_readout_strip ids: DCC, RO Tower, crystal_readout_strip_id
		push @channel_ids, [$DCC, $readout_tower,$crystal_readout_strip_id];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_crystal_trigger_strip_to_EE_crystal_number {

	my $name = "EE_crystal_trigger_strip";
	my $maps_to = "EE_crystal_number";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables
	my $TCC;
	my $trigger_tower;
	my $trigger_strip;
	my $crystal_trigger_strip;
	my $crystal_trigger_strip_id;

	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;

		#EE_crystal_number logicId
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;


	        $TCC = $channels[8];
		$trigger_tower= $channels[9];
		$trigger_strip= $channels[10];
		$crystal_trigger_strip=$channels[11];

		#crystal_trigger_strip_id=(strip-1)*5+channel_in_strip
		$crystal_trigger_strip_id=($trigger_strip-1)*5+$crystal_trigger_strip;

		push @channel_ids, [$TCC, $trigger_tower,$crystal_trigger_strip_id];
	}


	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}




sub define_EE_crystal_number_to_EE_HV_channel {


	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;



	my $name = "EE_crystal_number";
	my $maps_to = "EE_HV_channel";


	#temp variables
	my $Dee;
	my $hv;	
	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		$Dee = $channels[14];
		$hv = $channels[15];
		my $sideIndex = $side + 1;

		#EE_crystal_number logicId
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

		push @channel_ids, [$Dee, $hv];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_crystal_number_to_EE_LV_channel {


	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;



	my $name = "EE_crystal_number";
	my $maps_to = "EE_LV_channel";


	#temp variables
	my $Dee;
	my $lv;	
	my $ix;
	my $iy;
	my $side;

	my $logic_id;

	foreach my $line (@lines) {

		my @channels = split (/ /, $line);
		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		$Dee = $channels[14];
		$lv = $channels[16];
		my $sideIndex = $side + 1;

		#EE_crystal_number logicId
		my $logic_id = sprintf "201%01d%03d%03d", $sideIndex, $ix, $iy;
		push @logic_ids, $logic_id;

		push @channel_ids, [$Dee, $lv];

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}



sub define_EE_readout_strip_zseccu_to_EE_readout_strip {


	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	my @ids;

	my $name = "EE_readout_strip_zseccu";
	my $maps_to = "EE_readout_strip";

	my $logic_id;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	
		my $DCC=$channels[4];
		my $sector=0;
		if($DCC<10) {
		    if($DCC<=3){
			$sector=$DCC+6;
		    } else {
			$sector=$DCC-3;
		    } 
		}else {
		    if($DCC<=48){
			$sector=$DCC-39;
		    } else {
			$sector=$DCC-48;
		    }
		} 
		my $ccu_id=$channels[13];

		my $sideIndex=$channels[2]+1;
		


	        my $id = sprintf "%01d %01d %02d %01d",$sideIndex, $sector, $ccu_id, $channels[6];
		
		my $logic_id = sprintf "2121%03d%02d%01d", $channels[4],$channels[5],$channels[6];

	        push @ids, $id;
		push @logic_ids, $logic_id;


	}



	my @logic_ids_uni;



	my $i=-1;
	for my $id (@logic_ids) {
	    $i++;
	    my $ifound=0;

	    for my $id_uni (@logic_ids_uni) {
		
		if($id == $id_uni ) {
		    $ifound=1;

		    last;
		}
	    }
	    if($ifound==0){

                push @logic_ids_uni, $logic_ids[$i];

		my @channels = split (/ /, $ids[$i]);
		my $sideIndex = $channels[0];
		my $sector= $channels[1];
		my $ccu_id= $channels[2];
		my $strip= $channels[3];
		my $side= $sideIndex-1;
		$sector=$sector*$side;
		push @channel_ids, [$sector, $ccu_id, $strip];
		
	    }
	}
    


	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids_uni, channel_ids => \@channel_ids
	};

}

sub define_EE_crystal_number_to_ECAL_LMR {

	my $name = "EE_crystal_number";
	my $maps_to = "ECAL_LMR";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables

	my $ix;
	my $iy;
	my $side;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;
		my $DCC = $channels[4];
		my $sector=0;
		if($DCC<10) {
		    if($DCC<=3){
			$sector=$DCC+6;
		    } else {
			$sector=$DCC-3;
		    } 
		}else {
		    if($DCC<=48){
			$sector=$DCC-39;
		    } else {
			$sector=$DCC-48;
		    }
		} 
		my $lmr_chan=0;

		if($side==1){
		    $lmr_chan=$DCC+600-573;
		    if(($DCC==53&&$ix>50)||$DCC==54) {
			$lmr_chan=$lmr_chan+1;
		    }
		} else {
		    $lmr_chan=$DCC+600-518;
		    if(($DCC==8&&$ix>50)||$DCC==9) {
			$lmr_chan=$lmr_chan+1;
		    }
		}

		my $logic_id = sprintf "20603000%02d", $lmr_chan;
		

		#xtal number ids: side, ix, iy
		push @channel_ids, [$side, $ix, $iy];
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}

sub define_EE_crystal_number_to_EE_LMM {

	my $name = "EE_crystal_number";
	my $maps_to = "EE_LMM";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables

	my $ix;
	my $iy;
	my $side;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;
		my $Dee = $channels[14];
		my $lmm = $channels[17];

                my $logic_id = sprintf "20601%02d0%02d",$Dee, $lmm;

		

		#xtal number ids: side, ix, iy
		push @channel_ids, [$side, $ix, $iy];
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}



sub define_EE_crystal_number_to_EE_LM_PN {

	my $name = "EE_crystal_number";
	my $maps_to = "EE_LM_PN";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;

	#temp variables

	my $ix;
	my $iy;
	my $side;
	my $logic_id;


	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

		$ix = $channels[0];
		$iy = $channels[1];
		$side = $channels[2];
		my $sideIndex = $side + 1;
		my $Dee = $channels[14];
		my $lmm = $channels[17];
		my $ccu_id=$channels[13];
		my $pns=0;

		if($lmm==1) { $pns=100;}
		if($lmm==2)  { $pns=101;}
		if($lmm==3)  { $pns=102;}
		if($lmm==4)  { $pns=103;}
		if($lmm==5)  { $pns=207;}
		if($lmm==6)  { $pns=205;}
		if($lmm==7)  { $pns=206;}
		if($lmm==8)  { $pns=209;}
		if($lmm==9)  { $pns=200;}
		if($lmm==10) { $pns=201;}
		if($lmm==11) { $pns=202;}
		if($lmm==12) { $pns=203;}
		if($lmm==13) { $pns=204;}
		if($lmm==14) { $pns=107;}
		if($lmm==15) { $pns=105;}
		if($lmm==16) { $pns=106;}
		if($lmm==17) { $pns=109;}
		if($lmm==18) { $pns=104;}
		if($lmm==19) { $pns=108;}


		my $logic_id = sprintf "20602%02d%03d", $Dee,$pns;
		if($lmm==18 && ($ccu_id==37 || $ccu_id==40 ||  $ccu_id==40 ) ){
		    $logic_id = sprintf "20602%02d%03d", $Dee,209;
		}
		#xtal number ids: side, ix, iy
		push @channel_ids, [$side, $ix, $iy];
		push @logic_ids, $logic_id;

	}

	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};

}







sub define_EB_crystal_number_to_EB_trigger_tower {
  my $tt_def = define_EB_trigger_tower();
  my $tt_logic_ids = $tt_def->{logic_ids};
  my $tt_channel_ids = $tt_def->{channel_ids};
  my $count = scalar @{$tt_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_trigger_tower";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      # calculate the tt channel indexes
      my $ttj = POSIX::floor($j/5.0);
      my $tti = POSIX::floor($i/5.0);
      
      # the trigger tower
      my $tt = $ttj + 4*$tti + 1;
      
      # get the logic_id for this tt channel
      my $tt_id;
      
      for my $i (0..$count-1) {
	my @ids = @{$$tt_channel_ids[$i]};
	if ($ids[0] == $SM && $ids[1] == $tt) {
	  $tt_id = $$tt_logic_ids[$i];
	  last;
	}
      }

      if (!defined $tt_id) {
	die "Cannot determine logic_id of TT channel SM=$SM, ch=$tt\n";
      }
      
      # set the mapping
      push @logic_ids, $tt_id;
      push @channel_ids, [ $SM, $xtal ];

      # print "SM $SM xtal $xtal -> TT $tt\n";
    }
  }

	return { 
	  name => $name, maps_to => $maps_to, 
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};
}

sub define_EB_crystal_number_to_EB_module {
  my $M_def = define_EB_module();
  my $M_logic_ids = $M_def->{logic_ids};
  my $M_channel_ids = $M_def->{channel_ids};
  my $count = scalar @{$M_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_module";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      my $M;
      if ($xtal <= 500) {
	$M = 1;
      } else {
	$M = POSIX::floor(($xtal - 1 - 500)/400.0) + 2;
      }
      
      # get the logic_id for this M channel
      my $M_id;

      for my $i (0..$count-1) {
	my @ids = @{$$M_channel_ids[$i]};
	if ($ids[0] == $SM && $ids[1] == $M) {
	  $M_id = $$M_logic_ids[$i];
	  last;
	}
      }

      if (!defined $M_id) {
	die "Cannot determine logic_id of M channel SM=$SM, ch=$M\n";
      }
      
      # set the mapping
      push @logic_ids, $M_id;
      push @channel_ids, [ $SM, $xtal ];

      # print "SM $SM xtal $xtal -> M $M\n";
    }
  }

  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_constr_supermodule_to_EB_supermodule {
  my $name = "EB_constr_supermodule";
  my $maps_to = "EB_supermodule";

  my @logic_ids;
  my @channel_ids;
  my @slot_to_constr={-1,12,17,10,1,8,4,27,20,23,25,6,34,35,15,18,30,21,9
			      ,24,22,13,31,26,16,2,11,5,0,29,28,14,33,32,3,7,19};
  foreach my $SM (1..36) {
      my $constSM=$slot_to_constr[$SM];
    my $logic_id = sprintf "1041%02d00%02d", $SM, $SM;
    push @logic_ids, $logic_id;
    push @channel_ids, [ $constSM ];
  }

  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_constr_crystal_number_to_EB_crystal_number {
  my $name = "EB_constr_crystal_number";
  my $maps_to = "EB_crystal_number";

  my @logic_ids;
  my @channel_ids;
  my @slot_to_constr={-1,12,17,10,1,8,4,27,20,23,25,6,34,35,15,18,30,21,9
			      ,24,22,13,31,26,16,2,11,5,0,29,28,14,33,32,3,7,19};

  foreach my $SM (1..36) {
    foreach my $cn (1..1700) {
      my $constSM=$slot_to_constr[$SM];
      my $logic_id = sprintf "1011%02d%04d", $SM, $cn;

      # set the mapping
      push @logic_ids, $logic_id;
      push @channel_ids, [$constSM, $cn];          
    }
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_elec_crystal_number {
  my $ecn_def = define_EB_elec_crystal_number();
  my $ecn_logic_ids = $ecn_def->{logic_ids};
  my $ecn_channel_ids = $ecn_def->{channel_ids};
  my $count = scalar @{$ecn_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_elec_crystal_number";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    foreach my $cn (1..1700) {
      my $ecn = cn_to_ecn($cn);
      
      # get the logic_id for this ecn channel
      my $ecn_id;
      for my $i (0..$count-1) {
	my @ids = @{$$ecn_channel_ids[$i]};
	if ($ids[0] == $SM && $ids[1] == $ecn) {
	  $ecn_id = $$ecn_logic_ids[$i];
	  last;
	}
      }
      if (!defined $ecn_id) {
	die "Cannot determine logic_id of crystal channel SM=$SM, ecn=$ecn\n";
      }
      
      # set the mapping
      push @logic_ids, $ecn_id;
      push @channel_ids, [$SM, $cn];          
    }
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}



sub define_EB_crystal_number_to_EB_fe_crystal_number {
  my $fecn_def = define_EB_fe_crystal_number();
  my $fecn_logic_ids = $fecn_def->{logic_ids};
  my $fecn_channel_ids = $fecn_def->{channel_ids};
  my $count = scalar @{$fecn_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_fe_crystal_number";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    foreach my $cn (1..1700) {
      my ($tt, $fecn) = cn_to_fecn($cn);
      
      # get the logic_id for this fecn channel
      my $fecn_id;
      for my $i (0..$count-1) {
	my @ids = @{$$fecn_channel_ids[$i]};
	if ($ids[0] == $SM && $ids[1] == $tt && $ids[2] == $fecn) {
	  $fecn_id = $$fecn_logic_ids[$i];
	  last;
	}
      }
      if (!defined $fecn_id) {
	die "Cannot determine logic_id of crystal channel SM=$SM, fecn=$fecn\n";
      }
      
      # set the mapping
      push @logic_ids, $fecn_id;
      push @channel_ids, [$SM, $cn];          
    }
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}


sub define_EB_fe_crystal_number_to_EB_crystal_number {
  my $cn_def = define_EB_crystal_number();
  my $cn_logic_ids = $cn_def->{logic_ids};
  my $cn_channel_ids = $cn_def->{channel_ids};
  my $count = scalar @{$cn_logic_ids};

  my $name = "EB_fe_crystal_number";
  my $maps_to = "EB_crystal_number";

  my @logic_ids;
  my @channel_ids;

  # get the logic_id for this cn channel
  my $cn_id;
  for my $i (0..$count-1) {
    my @ids = @{$$cn_channel_ids[$i]};
    my ($SM, $cn) = @ids[0..1];
    my ($tt, $fecn) = cn_to_fecn($cn);	
    $cn_id = $$cn_logic_ids[$i];
    # set the mapping
    push @logic_ids, $cn_id;
    push @channel_ids, [$SM, $tt, $fecn];          
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_ECAL_crystal_number_fedccuxt_to_EB_crystal_number {
  my $cn_def = define_EB_crystal_number();
  my $cn_logic_ids = $cn_def->{logic_ids};
  my $cn_channel_ids = $cn_def->{channel_ids};
  my $count = scalar @{$cn_logic_ids};

  my $name = "ECAL_crystal_number_fedccuxt";
  my $maps_to = "EB_crystal_number";

  my @logic_ids;
  my @channel_ids;

  # get the logic_id for this cn channel
  my $cn_id;
  for my $i (0..$count-1) {
    my @ids = @{$$cn_channel_ids[$i]};
    my ($SM, $cn) = @ids[0..1];
    my ($tt, $fecn) = cn_to_fecn($cn);	
    $cn_id = $$cn_logic_ids[$i];


#   now convert SM in FED 
    if($SM>0){
    my $FED=0; 
    if($SM <=18 ){
	$FED=627+$SM;
    }
    if($SM >=19 ){
	$FED=609+($SM-18);
    }

    # set the mapping
    push @logic_ids, $cn_id;
    push @channel_ids, [$FED, $tt, $fecn];
}          
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}


sub define_EB_elec_crystal_number_to_EB_crystal_number {
  my $cn_def = define_EB_crystal_number();
  my $cn_logic_ids = $cn_def->{logic_ids};
  my $cn_channel_ids = $cn_def->{channel_ids};
  my $count = scalar @{$cn_logic_ids};

  my $name = "EB_elec_crystal_number";
  my $maps_to = "EB_crystal_number";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    foreach my $ecn (0..1699) {
      my $cn = ecn_to_cn($ecn);
      
      # get the logic_id for this ecn channel
      my $cn_id;
      for my $i (0..$count-1) {
	my @ids = @{$$cn_channel_ids[$i]};
	if ($ids[0] == $SM && $ids[1] == $cn) {
	  $cn_id = $$cn_logic_ids[$i];
	  last;
	}
      }
      if (!defined $cn_id) {
	die "Cannot determine logic_id of crystal channel SM=$SM, cn=$cn\n";
      }
      
      # set the mapping
      push @logic_ids, $cn_id;
      push @channel_ids, [$SM, $ecn];          
    }
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}



sub define_EB_crystal_number_to_EB_HV_channel {
  my $hv_def = define_EB_HV_channel();
  my $hv_logic_ids = $hv_def->{logic_ids};
  my $hv_channel_ids = $hv_def->{channel_ids};
  my $count = scalar @{$hv_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_HV_channel";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      # calculate the hv channel indexes
      my $hvj = POSIX::floor($j/10.0);
      my $hvi = POSIX::floor($i/5.0);
      
      # the high voltage channel
      my $hv = $hvj + 2*$hvi + 1;
      
      # get the logic_id for this hv channel
      my $hv_id;
      for my $i (0..$count-1) {
	my @ids = @{$$hv_channel_ids[$i]};
	if ($ids[0] == $SM && $ids[1] == $hv) {
	  $hv_id = $$hv_logic_ids[$i];
	  last;
	}
      }
      if (!defined $hv_id) {
	die "Cannot determine logic_id of HV channel SM=$SM, ch=$hv\n";
      }
      
      # set the mapping
      push @logic_ids, $hv_id;
      push @channel_ids, [$SM, $xtal];

      # print "SM $SM xtal -> HV $hv\n";
    }
  }

  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_LV_channel {
  my $lv_def = define_EB_LV_channel();
  my $lv_logic_ids = $lv_def->{logic_ids};
  my $lv_channel_ids = $lv_def->{channel_ids};
  my $count = scalar @{$lv_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_LV_channel";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      my $lv;
      if ($i < 5) {
	$lv = 1;
      } else {
	# calculate the lv channel indexes
	my $lvj = POSIX::floor($j/10.0);
	my $lvi = POSIX::floor(($i-5)/10.0);
      
	$lv = (2*$lvi) + $lvj + 2;
      }
      
      # get the logic_id for this lv channel
      my $lv_id;
      for my $i (0..$count-1) {
	my @ids = @{ $$lv_channel_ids[$i] };
	if ($ids[0] == $SM && $ids[1] == $lv) {
	  $lv_id = $$lv_logic_ids[$i];
	  last;
	}
      }
      if (!defined $lv_id) {
	die "Cannot determine logic_id of LV channel SM=$SM, ch=$lv\n";
      }
      
      # set the mapping
      push @logic_ids, $lv_id;
      push @channel_ids, [ $SM, $xtal ];

      # print "SM $SM xtal $xtal -> LV $lv\n";
    }
  }
  
  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_LM_channel {
  my $lm_def = define_EB_LM_channel();
  my $lm_logic_ids = $lm_def->{logic_ids};
  my $lm_channel_ids = $lm_def->{channel_ids};
  my $count = scalar @{$lm_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_LM_channel";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      my $lm;
      if ($i < 5) {
	$lm = 1;
      } else {
	# calculate the lm channel indexes
	my $lmj = POSIX::floor($j/10.0);
	my $lmi = POSIX::floor(($i-5)/20.0);
      
	$lm = (2*$lmi) + $lmj + 2;
      }
      
      # get the logic_id for this lm channel
      my $lm_id;
      for my $i (0..$count-1) {
	my @ids = @{ $$lm_channel_ids[$i] };
	if ($ids[0] == $SM && $ids[1] == $lm) {
	  $lm_id = $$lm_logic_ids[$i];
	  last;
	}
      }
      if (!defined $lm_id) {
	die "Cannot determine logic_id of LM channel SM=$SM, ch=$lm\n";
      }

      # set the mapping
      push @logic_ids, $lm_id;
      push @channel_ids, [ $SM, $xtal ];

      # print "SM $SM xtal $xtal -> LM $lm\n";
    }
  }

  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_LM_side {
  my $lm_def = define_EB_LM_side();
  my $lm_logic_ids = $lm_def->{logic_ids};
  my $lm_channel_ids = $lm_def->{channel_ids};
  my $count = scalar @{$lm_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_LM_side";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      my $j = POSIX::floor(($xtal-1)/20.0);
      my $i = ($xtal-1) - 20*$j;

      my $lm;
      if ($j < 5) {
	$lm = 1;
      } else {
	$lm=1;
	if($i<10) 
	{ 
	    $lm=0;
	}
      }

      my $logic_id = sprintf "1132%02d00%02d", $SM, $lm;
      
      # set the mapping
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $xtal ];

      # print "SM $SM xtal $xtal -> LM $lm\n";
    }
  }

  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_LM_side_to_EB_crystal_number {
  my $maps_to = "EB_crystal_number";
  my $name = "EB_LM_side";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (1..36) {
    for my $xtal (1..1700) {
      my $j = POSIX::floor(($xtal-1)/20.0);
      my $i = ($xtal-1) - 20*$j;

      my $lm;
      if ($j < 5) {
	$lm = 1;
      } else {
	$lm=1;
	if($i<10) 
	{ 
	    $lm=0;
	}
      }

      my $logic_id = sprintf "1011%02d%04d", $SM, $xtal;
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $lm ];

    }
  }

  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}



sub define_EB_crystal_number_to_ECAL_LMR {

  my $name = "EB_crystal_number";
  my $maps_to = "ECAL_LMR";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (1..36) {
    for my $xtal (1..1700) {
      my $j = POSIX::floor(($xtal-1)/20.0);
      my $i = ($xtal-1) - 20*$j;

      my $lm;
      if ($j < 5) {
	$lm = 0;
      } else {
	  if($i<10) { 
	      $lm=1;
	  }else {
	      $lm=0;
	  }
      }
      my $sm_ord=0;
      if($SM>18) {
	  $sm_ord=$SM-18;
      } else {
	  $sm_ord=$SM+18;
      }
      
      my $chan=($sm_ord-1)*2+$lm+1;

      my $logic_id = sprintf "10603000%02d", $chan;
      
      # set the mapping
      push @logic_ids, $logic_id;
      push @channel_ids, [ $SM, $xtal ];

      # print "SM $SM xtal $xtal -> LM $lm\n";
    }
  }

  return { 
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_LM_PN {
  my $pn_def = define_EB_LM_PN();
  my $pn_logic_ids = $pn_def->{logic_ids};
  my $pn_channel_ids = $pn_def->{channel_ids};
  my $count = scalar @{$pn_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_LM_PN";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      # crystal indexes
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      # LM channel
      my $lm;
      if ($i < 5) {
	  $lm = 1;
      } else {
	  # calculate the lm channel indexes
	  my $lmj = POSIX::floor($j/10.0);
	  my $lmi = POSIX::floor(($i-5)/20.0);
	  
	  $lm = (2*$lmi) + $lmj + 2;
      }

      # PN channel
      my $pn;
      if ($lm == 1) { 
	  if ($j < 10 ) { $pn = 0; }
	  else          { $pn = 5; }
      } else {
	  if ($lm % 2 == 0) { $pn = $lm/2; }
	  else              { $pn = (($lm-1)/2) + 5; }
      }

      # get the logic_id for this PN
      my $pn_id;
      for my $n (0..$count-1) {
	my @ids = @{ $$pn_channel_ids[$n] };
	if ($ids[0] == $SM && $ids[1] == $pn) {
	  $pn_id = $$pn_logic_ids[$n];
	  last;
	}
      }
      if (!defined $pn_id) {
	die "Cannot determine logic_id of PN SM=$SM, pn=$pn\n";
      }

      # set the mapping
      push @logic_ids, $pn_id;
      push @channel_ids, [ $SM, $xtal ];

      print "SM $SM xtal $xtal -> LM_channel $lm -> PN $pn\n";
    }
  }

  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_T_capsule {
  my $t_def = define_EB_T_capsule();
  my $t_logic_ids = $t_def->{logic_ids};
  my $t_channel_ids = $t_def->{channel_ids};
  my $count = scalar @{$t_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_T_capsule";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      # crystal indexes
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      # T_capsule channel
      my $ti = POSIX::floor($i/5.0);
      my $tj = POSIX::floor($j/2.0);
      my $t = ($ti * 10) + $tj + 1;

      # get the logic_id for this vfe channel
      my $t_id;
      for my $n (0..$count-1) {
	my @ids = @{ $$t_channel_ids[$n] };
	if ($ids[0] == $SM && $ids[1] == $t) {
	  $t_id = $$t_logic_ids[$n];
	  last;
	}
      }
      if (!defined $t_id) {
	die "Cannot determine logic_id of T_capsule channel SM=$SM, t=$t\n";
      }

      # set the mapping
      push @logic_ids, $t_id;
      push @channel_ids, [ $SM, $xtal ];

      print "SM $SM xtal $xtal ($i, $j) -> t $t ($ti, $tj)\n";
    }
  }

  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_T_capsule_to_EB_crystal_number {
  my $cn_def = define_EB_crystal_number();
  my $cn_logic_ids = $cn_def->{logic_ids};
  my $cn_channel_ids = $cn_def->{channel_ids};
  my $count = scalar @{$cn_logic_ids};

  my $name = "EB_T_capsule";
  my $maps_to = "EB_crystal_number";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $tc (1..170) {
      # calculate the tc channel indexes
      my $tci = POSIX::floor(($tc-1)/10.0);
      my $tcj = $tc - ($tci * 10) - 1;

      # calculate the crystal indexes
      my $i = ($tci*5) + 2;
      my $j = ($tcj*2);

      # calculate the crystal number
      my $cn = ($i*20) + $j + 1;

      # get the logic_id for this tc channel
      my $cn_id;
      for my $n (0..$count-1) {
	my @ids = @{ $$cn_channel_ids[$n] };
	if ($ids[0] == $SM && $ids[1] == $cn) {
	  $cn_id = $$cn_logic_ids[$n];
	  last;
	}
      }
      if (!defined $cn_id) {
	die "Cannot determine logic_id of crystal SM=$SM xtal=$cn";
      }

      # set the mapping
      push @logic_ids, $cn_id;
      push @channel_ids, [ $SM, $tc ];

#      print "SM $SM T_capsule $tc ($tci, $tcj) -> xtal $cn ($i, $j)\n";
    }
  }

  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_EB_crystal_number_to_EB_VFE {
  my $vfe_def = define_EB_VFE();
  my $vfe_logic_ids = $vfe_def->{logic_ids};
  my $vfe_channel_ids = $vfe_def->{channel_ids};
  my $count = scalar @{$vfe_logic_ids};

  my $name = "EB_crystal_number";
  my $maps_to = "EB_VFE";

  my @logic_ids;
  my @channel_ids;

  foreach my $SM (0..36) {
    for my $xtal (1..1700) {
      # crystal indexes
      my $i = POSIX::floor(($xtal-1)/20.0);
      my $j = ($xtal-1) - 20*$i;

      # calculate the tt channel indexes
      my $ttj = POSIX::floor($j/5.0);
      my $tti = POSIX::floor($i/5.0);

      # the trigger tower
      my $tt = $ttj + 4*$tti + 1;

      # electronics crystal number
      my $ecn = &cn_to_ecn($xtal);

      # VFE channel
      my $vfe = POSIX::floor(($ecn - 25*($tt-1))/5.0) + 1;

      # get the logic_id for this vfe channel
      my $vfe_id;
      for my $n (0..$count-1) {
	my @ids = @{ $$vfe_channel_ids[$n] };
	if ($ids[0] == $SM && $ids[1] == $tt && $ids[2] == $vfe) {
	  $vfe_id = $$vfe_logic_ids[$n];
	  last;
	}
      }
      if (!defined $vfe_id) {
	die "Cannot determine logic_id of VFE channel SM=$SM, tt=$tt, vfe=$vfe\n";
      }

      # set the mapping
      push @logic_ids, $vfe_id;
      push @channel_ids, [ $SM, $xtal ];

#      print "SM $SM xtal $xtal -> tt $tt vfe $vfe\n";
    }
  }

  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_ECAL_readout_strip_to_EB_VFE {
  my $vfe_def = define_EB_VFE();
  my $vfe_logic_ids = $vfe_def->{logic_ids};
  my $vfe_channel_ids = $vfe_def->{channel_ids};
  my $count = scalar @{$vfe_logic_ids};

  my $name = "ECAL_readout_strip";
  my $maps_to = "EB_VFE";

  my @logic_ids;
  my @channel_ids;


  for my $n (0..$count-1) {
      my @ids = @{ $$vfe_channel_ids[$n] };
      
      my $SM=$ids[0];
      my $tt=$ids[1];
      my $vfe=$ids[2] ;
      my $vfe_id = $$vfe_logic_ids[$n];
      
#   now convert SM in FED 
      if($SM>0){
	  my $FED=0; 
	  if($SM <=18 ){
	      $FED=627+$SM;
	  }
	  if($SM >=19 ){
	      $FED=609+($SM-18);
	  }
      
	  # set the mapping
	  push @logic_ids, $vfe_id;
	  push @channel_ids, [ $FED, $tt, $vfe ];
      }
    }



  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}
sub define_ECAL_readout_strip_to_EE_readout_strip {
  my $vfe_def = define_EE_readout_strip();
  my $vfe_logic_ids = $vfe_def->{logic_ids};
  my $vfe_channel_ids = $vfe_def->{channel_ids};
  my $count = scalar @{$vfe_logic_ids};

  my $name = "ECAL_readout_strip";
  my $maps_to = "EE_readout_strip";

  my @logic_ids;
  my @channel_ids;


  for my $n (0..$count-1) {
      my @ids = @{ $$vfe_channel_ids[$n] };
      
      my $dcc=$ids[0]+600;
      my $tt=$ids[1];
      my $vfe=$ids[2] ;
      my $vfe_id = $$vfe_logic_ids[$n];
      
      
      # set the mapping
      push @logic_ids, $vfe_id;
      push @channel_ids, [ $dcc, $tt, $vfe ];



  }

  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}

sub define_ECAL_readout_strip_to_EE_offline_stripid {

	my $name = "ECAL_readout_strip";	
	my $maps_to = "EE_offline_stripid";

	my @channel_ids;
	my @logic_ids;

	#opening file
	open (FILE , "CMSSW.txt") || die ("could not open EE numbering file");
	#reading it into an array
	my @lines = <FILE>;
	#getting the first line out	
	shift @lines;
	#temp variables
	my $DCC;
	my $readout_tower;
	my $readout_strip;
	my @ids;

	foreach my $line (@lines) {

	        my @channels = split (/ /, $line);

	        #id =DCC TTower readout_strip;
		my $FED=$channels[4]+600;
	        my $id = sprintf "%03d %02d %02d %06d", $FED,$channels[5],$channels[6],$channels[19];

	        push @ids, $id;
	}

	#perlish - returns unique entries using internal references AND a hash
	#(actually, not an original idea)

	undef my %saw;
	my @unique = grep(!$saw{$_}++, @ids);
	
	foreach my $id (@unique) {

		my @channels = split (/ /, $id);
		$DCC = $channels[0];
		$readout_tower = $channels[1];
		$readout_strip = $channels[2];
		my $logic_id = $channels[3];
		push @logic_ids, $logic_id;
		push @channel_ids, [$DCC, $readout_tower,$readout_strip];
	}


	return { 
		name => $name, maps_to => $maps_to, 
		logic_ids => \@logic_ids, channel_ids => \@channel_ids
	};


}

sub define_ECAL_trigger_tower_to_EB_trigger_tower {
  my $vfe_def = define_EB_trigger_tower();
  my $vfe_logic_ids = $vfe_def->{logic_ids};
  my $vfe_channel_ids = $vfe_def->{channel_ids};
  my $count = scalar @{$vfe_logic_ids};

  my $name = "ECAL_trigger_tower";
  my $maps_to = "EB_trigger_tower";

  my @logic_ids;
  my @channel_ids;


  for my $n (0..$count-1) {
      my @ids = @{ $$vfe_channel_ids[$n] };
      
      my $SM=$ids[0];
      my $tt=$ids[1];
      my $vfe_id = $$vfe_logic_ids[$n];
      
#   now convert SM in FED 
      if($SM>0){
	  my $TCC=0; 
	  if($SM <=18 ){
	      $TCC=54+$SM;
	  }
	  if($SM >=19 ){
	      $TCC=36+($SM-18);
	  }
      
	  # set the mapping
	  push @logic_ids, $vfe_id;
	  push @channel_ids, [ $TCC, $tt ];
      }
    }


  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}
sub define_EB_readout_tower_to_EB_trigger_tower {
  my $vfe_def = define_EB_trigger_tower();
  my $vfe_logic_ids = $vfe_def->{logic_ids};
  my $vfe_channel_ids = $vfe_def->{channel_ids};
  my $count = scalar @{$vfe_logic_ids};

  my $name = "EB_readout_tower";
  my $maps_to = "EB_trigger_tower";

  my @logic_ids;
  my @channel_ids;


  for my $n (0..$count-1) {
      my @ids = @{ $$vfe_channel_ids[$n] };
      
      my $SM=$ids[0];
      my $tt=$ids[1];
      my $vfe_id = $$vfe_logic_ids[$n];
      
#   now convert SM in FED 
      if($SM>0){
	  my $TCC=0; 
	  if($SM <=18 ){
	      $TCC=627+$SM;
	  }
	  if($SM >=19 ){
	      $TCC=609+($SM-18);
	  }
      
	  # set the mapping
	  push @logic_ids, $vfe_id;
	  push @channel_ids, [ $TCC, $tt ];
      }
    }


  return {
	  name => $name, maps_to => $maps_to,
	  logic_ids => \@logic_ids, channel_ids => \@channel_ids
	 };
}


sub ecn_to_cn {
  my $ecn = shift;

  # the trigger tower - 1
  my $tt = POSIX::floor($ecn/25.0);
  
  # the tt indexes
  my $tti = POSIX::floor($tt/4.0);
  my $ttj = $tt - 4*$tti;
  
  # the minimum ecn for the trigger tower
  my $min_ecn = $tt*25;
  # the minimum ecn for first row in the tt column
  my $min_ecn_col = $tti*100;
  
  # determine whether this is a bottom up or a top down tower
  my $cn;
  my $tt_xtal_col;
  if ($tt < 12 ||
      $tt >= 20 && $tt < 28 ||
      $tt >= 36 && $tt < 44 ||
      $tt >= 52 && $tt < 60) {
    # bottom up "S" pattern
    $tt_xtal_col = 4 - POSIX::floor(($ecn - $min_ecn)/5.0);
    if ($tt_xtal_col % 2 == 0) {
      # even column
      $cn = $min_ecn_col + 5*$ttj + (25 + 15*$tt_xtal_col) - ($ecn - $min_ecn);
    } else {
      # odd column
      $cn = $min_ecn_col + 5*$ttj + $ecn - $min_ecn + 6 + (($tt_xtal_col-1)*25);
    }
  } else {
    # top down "S" pattern
    $tt_xtal_col = POSIX::floor(($ecn - $min_ecn)/5.0);
    if ($tt_xtal_col % 2 == 0) {
      # even column
      $cn = $min_ecn_col + 5*$ttj + $ecn - $min_ecn + (15 * $tt_xtal_col) + 1;
    } else {
      # odd column
      $cn = $min_ecn_col + 5*$ttj + 30 + (($tt_xtal_col-1)*25) - ($ecn - $min_ecn);
    }
  }
 # printf "ecn_to_cn %4d -> %4d ... tti %2d ttj %2d tt %2d min_ecn %4d min_ecn_col %4d tt_xtal_col %1d\n", 
 #   $ecn, $cn, $tti, $ttj, $tt, $min_ecn, $min_ecn_col, $tt_xtal_col;

  return $cn;

}

sub cn_to_ecn {
  my $cn = shift;

  # get the tt number
  my $i = POSIX::floor(($cn-1)/20.0);
  my $j = ($cn-1) - 20*$i;
  
  # calculate the tt channel indexes
  my $ttj = POSIX::floor($j/5.0);
  my $tti = POSIX::floor($i/5.0);
      
  # the trigger tower - 1
  my $tt = $ttj + 4*$tti;
     
  # the minimum ecn for the trigger tower
  my $min_ecn = $tt*25;
  # the minimum ecn for first row in the tt column
  my $min_ecn_col = $tti*100;
  # the column within the trigger tower
  my $tt_xtal_col = $i - 5*$tti;

  # determine whether this is a bottom up or a top down tower
  my $ecn;
  if ($tt < 12 ||
      $tt >= 20 && $tt < 28 ||
      $tt >= 36 && $tt < 44 ||
      $tt >= 52 && $tt < 60) {
    # bottom up "S" pattern
    if ($tt_xtal_col % 2 == 0) {
      # even column
      $ecn = $min_ecn + ((25 + 15*$tt_xtal_col) - ($cn - $min_ecn_col - 5*$ttj));
    } else {
      # odd column
      $ecn = $min_ecn + ($cn - $min_ecn_col - (5*$ttj) - 6 - (($tt_xtal_col - 1) * 25));
    }
  } else {
    # top down "S" pattern
    if ($tt_xtal_col % 2 == 0) {
      # even column
      $ecn = $min_ecn + ($cn - $min_ecn_col - (5*$ttj) - (15*$tt_xtal_col) - 1);
    } else {
      # odd column
      $ecn = $min_ecn + (30 + (($tt_xtal_col - 1) * 25) - ($cn - $min_ecn_col - (5*$ttj)));
    }
  }

# printf "cn_to_ecn %4d -> %4d ... tti %2d ttj %2d tt %2d min_ecn %4d min_ecn_col %4d tt_xtal_col %1d\n", 
#  $cn, $ecn, $tti, $ttj, $tt, $min_ecn, $min_ecn_col, $tt_xtal_col;

  return $ecn;
}



sub cn_to_fecn {
  my $cn = shift;

  # get the tt number
  my $i = POSIX::floor(($cn-1)/20.0);
  my $j = ($cn-1) - 20*$i;
  
  # calculate the tt channel indexes
  my $ttj = POSIX::floor($j/5.0);
  my $tti = POSIX::floor($i/5.0);
      
  # the trigger tower - 1
  my $tt = $ttj + 4*$tti;
     
  # the minimum ecn for the trigger tower
  my $min_ecn = $tt*25;
  # the minimum ecn for first row in the tt column
  my $min_ecn_col = $tti*100;
  # the column within the trigger tower
  my $tt_xtal_col = $i - 5*$tti;

  # determine whether this is a bottom up or a top down tower
  my $ecn;
  if ($tt < 12 ||
      $tt >= 20 && $tt < 28 ||
      $tt >= 36 && $tt < 44 ||
      $tt >= 52 && $tt < 60) {
    # bottom up "S" pattern
    if ($tt_xtal_col % 2 == 0) {
      # even column
      $ecn = $min_ecn + ((25 + 15*$tt_xtal_col) - ($cn - $min_ecn_col - 5*$ttj));
    } else {
      # odd column
      $ecn = $min_ecn + ($cn - $min_ecn_col - (5*$ttj) - 6 - (($tt_xtal_col - 1) * 25));
    }
  } else {
    # top down "S" pattern
    if ($tt_xtal_col % 2 == 0) {
      # even column
      $ecn = $min_ecn + ($cn - $min_ecn_col - (5*$ttj) - (15*$tt_xtal_col) - 1);
    } else {
      # odd column
      $ecn = $min_ecn + (30 + (($tt_xtal_col - 1) * 25) - ($cn - $min_ecn_col - (5*$ttj)));
    }
  }

  my $fecn = $ecn - ($tt*25);

# printf "cn_to_fecn %4d -> %4d ... tti %2d ttj %2d tt %2d min_ecn %4d min_ecn_col %4d tt_xtal_col %1d\n", 
#   $cn, $fecn, $tti, $ttj, $tt, $min_ecn, $min_ecn_col, $tt_xtal_col;

  return ($tt+1, $fecn);
}

1;

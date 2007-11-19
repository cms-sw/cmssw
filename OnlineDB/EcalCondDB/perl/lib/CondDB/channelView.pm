#!/usr/bin/perl

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
		   'EB_crystal_index' => \&define_EB_crystal_index,
		   'EB_trigger_tower' => \&define_EB_trigger_tower,
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
		   'EB_LM_PN' => \&define_EB_LM_PN,
		   'EB_T_capsule' => \&define_EB_T_capsule,
		   'EB_VFE' => \&define_EB_VFE,
		   'EB_LVRB_DCU' => \&define_EB_LVRB_DCU,
		   'EB_LVRB_T_sensor' => \&define_EB_LVRB_T_sensor,
		   'EB_mem_TT' => \&define_EB_mem_TT,
		   'EB_mem_channel' => \&define_EB_mem_channel,
		   'EB_crystal_number_to_EB_trigger_tower'
		   => \&define_EB_crystal_number_to_EB_trigger_tower,
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
		   'EB_fe_crystal_number_to_EB_crystal_number',
		   => \&define_EB_fe_crystal_number_to_EB_crystal_number
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
  my @logic_ids = (2000000001);
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

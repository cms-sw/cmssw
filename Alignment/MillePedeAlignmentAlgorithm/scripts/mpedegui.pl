#!/usr/bin/perl

##################
# 
# Project: Millipede Production Enviroment GUI
# File:    /afs/cern.ch/user/s/sanchl/public/ZooZ-1.2/mpedegui.zooz
# Authors: Luis Sanchez, Andrea Parenti, Silvia Miglioranzi
# Contact: sanchl@mail.cern.ch
#
##################

#
# Headers
#
use strict;
use warnings;

use Tk 804;
require Tk::ROText;

#
# Global variables
#
my (
     # MainWindow
     $MW,

     # Hash of all widgets
     %ZWIDGETS,
     

    );

#
# User-defined variables (if any)
#
    my $types = [ ['All Files',  '*'],
                  ['zsh scripts', '.sh'],
                  ['cfg files',   '.cfg'],
                  ['py files',    '.py'],
                ];

    my $pathmillescript_variable;
    my $pathcfg_variable;
    my $pathdata_variable;
    my $pathpedescript_variable;
    my $pathcastor_variable;
    my $jobname_variable;
    my $setuppedejob_variable;
    my $njobs_variable;
    my $appendmillejob_variable;
    my $batchclass_variable;
    my $mpssetup_output;
    my $status_output;
    my $firemerge_variable;
    my $njobsfir_variable;
    my $mpsfire_output;
    my $fetch_output;
    my $row_offset;
    
######################
#
# Create the MainWindow
#
######################

$MW = MainWindow->new;

######################
#
# Load any images and fonts
#
######################
ZloadImages();
ZloadFonts ();

### THIS IS THE TITLE ###

# Widget title_label isa Label
$ZWIDGETS{'title_label'} = $MW->Label(
   -text => 'Millipede Production System GUI',
  )->grid(
   -row        => 0,
   -column     => 0,
   -columnspan => 4,
   -sticky     => 'ew',
  );

### THIS IS THE MPS_SETUP SECTION ###

# Widget pathmillescript_label isa Label
$ZWIDGETS{'pathmillescript_label'} = $MW->Label(
   -text => 'Path to milleScript:',
  )->grid(
   -row    => 1,
   -column => 0,
   -sticky => 'ew',
  );


# Widget pathmillescript_entry isa Entry
$ZWIDGETS{'pathmillescript_entry'} = $MW->Entry(   
	-textvariable => \$pathmillescript_variable,
        -background => 'white',
	)
   ->grid(
   -row    => 2,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathcfg_label isa Label
$ZWIDGETS{'pathcfg_label'} = $MW->Label(
   -text => 'Path to cfg/py file:',
  )->grid(
   -row    => 3,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathcfg_entry isa Entry
$ZWIDGETS{'pathcfg_entry'} = $MW->Entry(
	-textvariable => \$pathcfg_variable,
        -background => 'white',
	)
	->grid(
   -row    => 4,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathdata_label isa Label
$ZWIDGETS{'pathdata_label'} = $MW->Label(
   -text => 'Path to data:',
  )->grid(
   -row    => 5,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathdata_entry isa Entry
$ZWIDGETS{'pathdata_entry'} = $MW->Entry(
	-textvariable => \$pathdata_variable,
        -background => 'white',
	)->grid(
   -row    => 6,
   -column => 0,
   -sticky => 'ew',
  );


# Widget pathpedescript_label isa Label
$ZWIDGETS{'pathpedescript_label'} = $MW->Label(
   -text => 'Path to pedeScript:',
  )->grid(
   -row    => 7,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathpedescript_entry isa Entry
$ZWIDGETS{'pathpedescript_entry'} = $MW->Entry(
	-textvariable => \$pathpedescript_variable,
        -background => 'white',
	)->grid(
   -row    => 8,
   -column => 0,
   -sticky => 'ew',
  );


# Widget pathcastor_label isa Label
$ZWIDGETS{'pathcastor_label'} = $MW->Label(
   -text => 'Path to castor directory:',
  )->grid(
   -row    => 9,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathcastor_entry isa Entry
$ZWIDGETS{'pathcastor_entry'} = $MW->Entry(
	-textvariable => \$pathcastor_variable,
        -background => 'white',
	)->grid(
   -row    => 10,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathmillescript_button isa Button
$ZWIDGETS{'pathmillescript_button'} = $MW->Button(
   -anchor  => 'center',
   -justify => 'center',
   -text    => 'Browse',
   -command => sub{$pathmillescript_variable = &open_file},
  )->grid(
   -row    => 2,
   -column => 1,
   -sticky => 'w',
  );

# Widget pathcfg_button isa Button
$ZWIDGETS{'pathcfg_button'} = $MW->Button(
   -text => 'Browse',
   -command => sub{$pathcfg_variable = &open_file},
  )->grid(
   -row    => 4,
   -column => 1,
   -sticky => 'w',
  );

# Widget pathdata_button isa Button
$ZWIDGETS{'pathdata_button'} = $MW->Button(
   -text => 'Browse',
   -command => sub{$pathdata_variable = &open_file},
  )->grid(
   -row    => 6,
   -column => 1,
   -sticky => 'w',
  );

# Widget pathpedescript_button isa Button
$ZWIDGETS{'pathpedescript_button'} = $MW->Button(
   -text => 'Browse',
   -command => sub{$pathpedescript_variable = &open_file},   
  )->grid(
   -row    => 8,
   -column => 1,
   -sticky => 'w',
  );

# Widget njobs_label isa Label
$ZWIDGETS{'njobs_label'} = $MW->Label(
   -text => 'Number of jobs:',
  )->grid(
   -row        => 1,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget njobs_menu isa Optionmenu
$ZWIDGETS{'njobs_menu'} = $MW->Optionmenu(
        -options => [[1 =>1], [10 =>10], [100=>100], [1000=>1000]],
        -variable => \$njobs_variable,
	)->grid(
   -row        => 2,
   -column     => 2,
   -sticky     => 'ew',
  );
  
# Widget jobaname_entry isa Entry
$ZWIDGETS{'njobs_entry'} = $MW->Entry(
	-textvariable => \$njobs_variable,
        -background => 'white',
	)->grid(

   -row        => 3,
   -column     => 2,
   -sticky => 'ew',
  );

# Widget batchclass_label isa Label
$ZWIDGETS{'batchclass_label'} = $MW->Label(
   -text => 'Batch system queue/class:',
  )->grid(
   -row        => 4,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget batchclass_menu isa Optionmenu
$ZWIDGETS{'batchclass_menu'} = $MW->Optionmenu(
        -options => [["8nm"=>"8nm"], ["1nh"=>"1nh"], ["8nh"=>"8nh"], ["1nd"=>"1nd"], ["2nd"=>"2nd"], ["1nw"=>"1nw"], ["2nw"=>"2nw"], ["cmscaf"=>"cmscaf"], ["cmscaf:cmscafspec"=>"cmscaf:cmscafspec"] ],
        -variable => \$batchclass_variable,
	)->grid(
   -row        => 5,
   -column     => 2,
   -sticky     => 'ew',
  );
  
# Widget batchclass_entry isa Entry
$ZWIDGETS{'batchclass_entry'} = $MW->Entry(
	-textvariable => \$batchclass_variable,
        -background => 'white',
	)->grid(

   -row        => 6,
   -column     => 2,
   -sticky => 'ew',
  );


# Widget jobname_label isa Label
$ZWIDGETS{'jobname_label'} = $MW->Label(
   -text => 'Jobname for batch system:',
  )->grid(
   -row    => 7,
   -column => 2,
   -sticky => 'ew',
  );


# Widget jobaname_entry isa Entry
$ZWIDGETS{'jobaname_entry'} = $MW->Entry(
	-textvariable => \$jobname_variable,
        -background => 'white',
	)->grid(
   -row    => 8,
   -column => 2,
   -sticky => 'ew',
  );

# Widget setuppedejob_label isa Label
$ZWIDGETS{'setuppedejob_label'} = $MW->Label(
   -text => 'Setup a Pede job?',
  )->grid(
   -row        => 1,
   -column     => 3,
   -sticky     => 'ew',
  );

# Widget setuppedejob_menu isa Optionmenu
$ZWIDGETS{'setuppedejob_menu'} = $MW->Optionmenu(
        -options => [[yes=>1], [no=>2]],
        -variable => \$setuppedejob_variable,
	)->grid(
   -row        => 2,
   -column     => 3,
   -sticky     => 'ew',
  );
  
# Widget appendmillejob_label isa Label
$ZWIDGETS{'appendmillejob_label'} = $MW->Label(
   -text => 'Set up additional Mille jobs?',
  )->grid(
   -row        => 3,
   -column     => 3,
   -sticky     => 'ew',
  );

# Widget appendmillejob_menu isa Optionmenu
$ZWIDGETS{'appendmillejob_menu'} = $MW->Optionmenu(
        -options => [[no=>2], [yes=>1]],
        -variable => \$appendmillejob_variable,
	)->grid(
   -row        => 4,
   -column     => 3,
   -sticky     => 'ew',
  );


# Widget setup_button isa Button
$ZWIDGETS{'setup_button'} = $MW->Button(
   -text => 'Run mps_setup',
   -command => \&setup_cmd,
   -background => 'cyan',
  )->grid(
   -row        => 14,
   -column     => 0,
   -columnspan => 4,
   -sticky     => 'ew',
  );


### THIS IS THE MPS_FIRE SECTION ###
$row_offset=15;

# Widget njobsfir_label isa Label
$ZWIDGETS{'njobsfir_label'} = $MW->Label(
   -text => 'Numbers of jobs to fire: ',
  )->grid(
   -row        => $row_offset,
   -column     => 0,
   -sticky     => 'ew',
  ); 
  
# Widget jobaname_entry isa Entry
$ZWIDGETS{'njobsfir_entry'} = $MW->Entry(
	-textvariable => \$njobsfir_variable,
        -background => 'white',
	)->grid(

   -row        => $row_offset+1,
   -column     => 0,
   -sticky => 'ew',
  );


# Widget firemerge_label isa Label
$ZWIDGETS{'firemerge_label'} = $MW->Label(
   -text => 'Fire merge job:',
  )->grid(
   -row        => $row_offset,
   -column     => 1,
   -sticky     => 'ew',
  );

 # Widget firemerge_menu isa Optionmenu
$ZWIDGETS{'firemerge_menu'} = $MW->Optionmenu(
        -options => [["no"=>""], ["yes"=>"-m"], ["force merge job"=>"-mf"]],
        -variable => \$firemerge_variable,
	)->grid(
   -row        => $row_offset+1,
   -column     => 1,
   -sticky     => 'ew',
  );

# Widget status_button isa Button
$ZWIDGETS{'runmpsfire_button'} = $MW->Button(
   -text => 'Run mps_fire',
   -command => \&mpsfire_cmd,
   -background => 'cyan',
  )->grid(
   -row        => $row_offset+1,
   -column     => 2,
   -columnspan => 2,
   -sticky     => 'ew',
  );

### THIS IS THE MPS_FETCH/MPS_STAT SECTION ###
$row_offset=17;

 # Widget runmpsfetch_button isa Button
$ZWIDGETS{'runmpsfetch_button'} = $MW->Button(
   -text => 'Run mps_fetch',
   -command => \&mpsfetch_cmd,
   -background => 'cyan',
  )->grid(
   -row        => $row_offset,
   -column     => 0,
   -columnspan => 2,
   -sticky     => 'ew',
  );  

# Widget status_button isa Button
$ZWIDGETS{'status_button'} = $MW->Button(
   -text => 'Run mps_stat',
   -command => \&mpsstat_cmd,
   -background => 'cyan',
  )->grid(
   -row        => $row_offset,
   -column     => 2,
   -columnspan => 2,
   -sticky     => 'ew',
  );

### THIS IS THE OUTPUT SECTION ###
$row_offset=19;

# Widget output_label isa Label
$ZWIDGETS{'output_label'} = $MW->Label(
   -text => 'Command output:',
  )->grid(
   -row        => $row_offset,
   -column     => 0,
   -columnspan => 4,
   -sticky     => 'ew',
  );

# Widget MPS Setup Output isa ROText
$ZWIDGETS{'ROText1'} = $MW->Scrolled('ROText',
  -scrollbars => 'e',
  -height => 15,
  -width => 40,
  -wrap => 'char',
  -background => 'white',
  -selectbackground => 'blue'
  )
  ->grid(
   -row        => $row_offset+1,
   -column     => 0,
   -columnspan => 4,
   -sticky     => 'ew',
  );


### QUIT SECTION ###
$row_offset=21;

# Widget quit_button isa Button
$ZWIDGETS{'quit_button'} = $MW->Button(
   -anchor  => 'center',
   -justify => 'center',
   -text    => 'Quit',
   -command => sub{exit(0)},
  )->grid(
   -row        => $row_offset,
   -column     => 0,
   -columnspan => 4,
   -sticky     => 'ew',
  );

###############
#
# MainLoop
#
###############
$pathmillescript_variable="";
$pathcfg_variable="";
$pathdata_variable="";
$pathpedescript_variable="";
$pathcastor_variable="";
$jobname_variable="";
$njobs_variable="1"; #The default is 1
$setuppedejob_variable="1"; #The default is "yes"
$appendmillejob_variable="2"; #The default is "no". 
$batchclass_variable="8nm";
$mpssetup_output="";
$status_output="";
$firemerge_variable=""; #The default is to merge nothing
$njobsfir_variable="1"; #This is the default anyway
$mpsfire_output="";
$fetch_output="";
MainLoop;

#######################
#
# Subroutines
#
#######################

sub ZloadImages {
}

sub ZloadFonts {
}

sub open_file {
  my $open = $MW->getOpenFile(-filetypes => $types,
                              -defaultextension => '*');
  return $open;
}

sub setup_cmd {
  my $setup_cmd = "";
  my $setup_opt = "";

  if ($appendmillejob_variable == 1) {
    $setup_opt .= " -a";
  }

  if ($setuppedejob_variable != 1) {
# Do not setup Pede job
    $setup_cmd = sprintf "mps_setup.pl %s %s %s %s %d %s %s",
      $setup_opt,$pathmillescript_variable,$pathcfg_variable,$pathdata_variable,
	$njobs_variable,$batchclass_variable,$jobname_variable;
	$mpssetup_output=`$setup_cmd 2>&1`;
  } else {
# Setup Mille and Pede
    $setup_opt .= " -m";
    $setup_cmd = sprintf "mps_setup.pl %s %s %s %s %d %s %s %s %s",
      $setup_opt,$pathmillescript_variable,$pathcfg_variable,$pathdata_variable,
	$njobs_variable,$batchclass_variable,$jobname_variable,
	  $pathpedescript_variable,$pathcastor_variable;
	  $mpssetup_output=`$setup_cmd 2>&1`;
  }

## This 2 lines put the output in the main terminal (not in the gui window) and should probably be removed after gui-development is over
#  printf $setup_cmd . "\n"; 
#  printf $mpssetup_output ."\n";

  $ZWIDGETS{'ROText1'}->insert('end',"$mpssetup_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}


sub mpsstat_cmd {

  my $status_cmd = "";
  
    $status_cmd = sprintf "mps_stat.pl";
	$status_output=`$status_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$status_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}


sub mpsfire_cmd {

  my $fire_cmd = "";
  
    $fire_cmd = sprintf "mps_fire.pl %s %s", $firemerge_variable, $njobsfir_variable;
	$mpsfire_output=`$fire_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$mpsfire_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}

sub mpsfetch_cmd {

  my $fetch_cmd = "";
  
    $fetch_cmd = sprintf "mps_fetch.pl";
	$fetch_output=`$fetch_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$fetch_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}

#This should clean the text output window, right now is not implemented in any button or call
sub cleanwindow{
  $ZWIDGETS{'ROText1'}->delete("1.0", 'end');
}

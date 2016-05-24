#!/usr/bin/env perl

##################
# 
# Project: Millipede Production Enviroment GUI
# File:    /afs/cern.ch/user/s/sanchl/public/ZooZ-1.2/mpedegui.zooz
# Authors: Luis Sanchez, Andrea Parenti, Silvia Miglioranzi
# Contact: parenti@mail.cern.ch
#
##################

#
# Headers
#
use strict;
use warnings;

use Tk 804;
use Tk::DialogBox;
require Tk::ROText;
use Switch;

# Needed in order to read mps.db
BEGIN {
use File::Basename;
unshift(@INC, dirname($0)."/mpslib");
}
use Mpslib;

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

# Working directory
    my $workdir;

# Variables for mps_setup
    my $pathmillescript_variable;
    my $pathcfg_variable;
    my $pathdata_variable;
    my $pathpedescript_variable;
    my $pathcastor_variable;
    my $jobname_variable;
    my $setuppedejob_variable;
    my $njobs_variable;
    my $appendmillejob_variable;
    my $milleclass_variable;
    my $pedeclass_variable;
    my $batchclass_variable;
    my $pedemem_variable;
# Variables for mps_fire
    my $firemerge_variable;
    my $njobsfir_variable;
# Variables for mps_kill

# Variables for mps_retry
    my $retryID_variable;
    my $retrymerge_variable;
    my $retryforce_variable;

# Variables for mps_save
    my $pathsave_variable;

# Standard outputs
    my $mpssetup_output;
    my $mpsfire_output;
    my $mpskill_output;
    my $mpsretry_output;
    my $mpsfetch_output;
    my $mpsstat_output;
    my $mpssave_output;

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

# Widget workdir_label isa Label
$ZWIDGETS{'workdir_label'} = $MW->Label(
   -text => 'Working Directory:',
  )->grid(
   -row        => 1,
   -column     => 0,
   -sticky     => 'ew',
  );

# Widget workDir isa ROText
$ZWIDGETS{'ROText2'} = $MW->ROText(
   -height       => 1,
   -background => 'white',
  )
   ->grid(
   -row        => 1,
   -column     => 1,
   -columnspan => 3,
  );


### THIS IS THE MPS_SETUP SECTION ###
$row_offset=2;

# Widget pathmillescript_label isa Label
$ZWIDGETS{'pathmillescript_label'} = $MW->Label(
   -text => 'Path to milleScript:',
  )->grid(
   -row    => $row_offset,
   -column => 0,
   -sticky => 'ew',
  );


# Widget pathmillescript_entry isa Entry
$ZWIDGETS{'pathmillescript_entry'} = $MW->Entry(   
	-textvariable => \$pathmillescript_variable,
        -background => 'white',
	)
   ->grid(
   -row    => $row_offset+1,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathcfg_label isa Label
$ZWIDGETS{'pathcfg_label'} = $MW->Label(
   -text => 'Path to cfg/py file:',
  )->grid(
   -row    => $row_offset+2,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathcfg_entry isa Entry
$ZWIDGETS{'pathcfg_entry'} = $MW->Entry(
	-textvariable => \$pathcfg_variable,
        -background => 'white',
	)
	->grid(
   -row    => $row_offset+3,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathdata_label isa Label
$ZWIDGETS{'pathdata_label'} = $MW->Label(
   -text => 'Path to data:',
  )->grid(
   -row    => $row_offset+4,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathdata_entry isa Entry
$ZWIDGETS{'pathdata_entry'} = $MW->Entry(
	-textvariable => \$pathdata_variable,
        -background => 'white',
	)->grid(
   -row    => $row_offset+5,
   -column => 0,
   -sticky => 'ew',
  );


# Widget pathpedescript_label isa Label
$ZWIDGETS{'pathpedescript_label'} = $MW->Label(
   -text => 'Path to pedeScript:',
  )->grid(
   -row    => $row_offset+6,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathpedescript_entry isa Entry
$ZWIDGETS{'pathpedescript_entry'} = $MW->Entry(
	-textvariable => \$pathpedescript_variable,
        -background => 'white',
	)->grid(
   -row    => $row_offset+7,
   -column => 0,
   -sticky => 'ew',
  );


# Widget pathcastor_label isa Label
$ZWIDGETS{'pathcastor_label'} = $MW->Label(
   -text => 'Path to castor directory:',
  )->grid(
   -row    => $row_offset+8,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathcastor_entry isa Entry
$ZWIDGETS{'pathcastor_entry'} = $MW->Entry(
	-textvariable => \$pathcastor_variable,
        -background => 'white',
	)->grid(
   -row    => $row_offset+9,
   -column => 0,
   -sticky => 'ew',
  );

# Widget pathmillescript_button isa Button
$ZWIDGETS{'pathmillescript_button'} = $MW->Button(
   -anchor  => 'center',
   -justify => 'center',
   -text    => 'Browse',
   -command => sub{$pathmillescript_variable = &open_file_v3($pathmillescript_variable,"Choose Mille script","sh")},
  )->grid(
   -row    => $row_offset+1,
   -column => 1,
   -sticky => 'w',
  );

# Widget pathcfg_button isa Button
$ZWIDGETS{'pathcfg_button'} = $MW->Button(
   -text => 'Browse',
   -command => sub{$pathcfg_variable = &open_file_v3($pathcfg_variable,"Choose cfg/py file","py")},
  )->grid(
   -row    => $row_offset+3,
   -column => 1,
   -sticky => 'w',
  );

# Widget pathdata_button isa Button
$ZWIDGETS{'pathdata_button'} = $MW->Button(
   -text => 'Browse',
   -command => sub{$pathdata_variable = &open_file_v3($pathdata_variable,"Choose data file","*")},
  )->grid(
   -row    => $row_offset+5,
   -column => 1,
   -sticky => 'w',
  );

# Widget pathpedescript_button isa Button
$ZWIDGETS{'pathpedescript_button'} = $MW->Button(
   -text => 'Browse',
   -command => sub{$pathpedescript_variable = &open_file_v3($pathpedescript_variable,"Choose Pede script","sh")},
  )->grid(
   -row    => $row_offset+7,
   -column => 1,
   -sticky => 'w',
  );

# Widget njobs_label isa Label
$ZWIDGETS{'njobs_label'} = $MW->Label(
   -text => 'Number of jobs:',
  )->grid(
   -row        => $row_offset,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget njobs_menu isa Optionmenu
$ZWIDGETS{'njobs_menu'} = $MW->Optionmenu(
        -options => [[1 =>1], [10 =>10], [100=>100], [1000=>1000]],
        -variable => \$njobs_variable,
	)->grid(
   -row        => $row_offset+1,
   -column     => 2,
   -sticky     => 'ew',
  );
  
# Widget jobaname_entry isa Entry
$ZWIDGETS{'njobs_entry'} = $MW->Entry(
	-textvariable => \$njobs_variable,
        -background => 'white',
	)->grid(

   -row        => $row_offset+2,
   -column     => 2,
   -sticky => 'ew',
  );

# Widget batchclass_label isa Label
$ZWIDGETS{'batchclass_label'} = $MW->Label(
   -text => 'Batch system queue/class:',
  )->grid(
   -row        => $row_offset+3,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget milleclass_menu isa Optionmenu
$ZWIDGETS{'milleclass_menu'} = $MW->Optionmenu(
        -options => [ ["(Choose mille queue)"=>""], ["8nm"=>"8nm"], ["1nh"=>"1nh"], ["8nh"=>"8nh"], ["1nd"=>"1nd"], ["2nd"=>"2nd"], ["1nw"=>"1nw"], ["2nw"=>"2nw"], ["cmscaf1nh"=>"cmscaf1nh"], ["cmscaf1nd"=>"cmscaf1nd"], ["cmscaf1nw"=>"cmscaf1nw"] ],
        -variable => \$milleclass_variable,
        -command => \&createbatchclass,
	)->grid(
   -row        => $row_offset+4,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget pedeclass_menu isa Optionmenu
$ZWIDGETS{'pedeclass_menu'} = $MW->Optionmenu(
        -options => [ ["(Choose pede queue)"=>""], ["8nm"=>"8nm"], ["1nh"=>"1nh"], ["8nh"=>"8nh"], ["1nd"=>"1nd"], ["2nd"=>"2nd"], ["1nw"=>"1nw"], ["2nw"=>"2nw"], ["cmscaf1nh"=>"cmscaf1nh"], ["cmscaf1nd"=>"cmscaf1nd"], ["cmscaf1nw"=>"cmscaf1nw"], ["cmscafspec1nh"=>"cmscafspec1nh"], ["cmscafspec1nd"=>"cmscafspec1nd"], ["cmscafspec1nw"=>"cmscafspec1nw"] ],
        -variable => \$pedeclass_variable,
        -command => \&createbatchclass,
	)->grid(
   -row        => $row_offset+5,
   -column     => 2,
   -sticky     => 'ew',
  );

  
# Widget batchclass_entry isa Entry
$ZWIDGETS{'batchclass_entry'} = $MW->Entry(
	-textvariable => \$batchclass_variable,
        -background => 'white',
	)->grid(

   -row        => $row_offset+6,
   -column     => 2,
   -sticky => 'ew',
  );


# Widget jobname_label isa Label
$ZWIDGETS{'jobname_label'} = $MW->Label(
   -text => 'Jobname for batch system:',
  )->grid(
   -row    => $row_offset+7,
   -column => 2,
   -sticky => 'ew',
  );


# Widget jobaname_entry isa Entry
$ZWIDGETS{'jobaname_entry'} = $MW->Entry(
	-textvariable => \$jobname_variable,
        -background => 'white',
	)->grid(
   -row    => $row_offset+8,
   -column => 2,
   -sticky => 'ew',
  );

# Widget setuppedejob_label isa Label
$ZWIDGETS{'setuppedejob_label'} = $MW->Label(
   -text => 'Setup a Pede job?',
  )->grid(
   -row        => $row_offset,
   -column     => 3,
   -sticky     => 'ew',
  );

# Widget setuppedejob_menu isa Optionmenu
$ZWIDGETS{'setuppedejob_menu'} = $MW->Optionmenu(
        -options => [[yes=>1], [no=>2]],
        -variable => \$setuppedejob_variable,
	)->grid(
   -row        => $row_offset+1,
   -column     => 3,
   -sticky     => 'ew',
  );


# Widget pede_mem isa Label
$ZWIDGETS{'pede_mem'} = $MW->Label(
   -text => 'Memory for pede job:',
  )->grid(
   -row        => $row_offset+2,
   -column     => 3,
   -sticky     => 'ew',
  );
  
# Widget pede_mem_entry isa Entry
$ZWIDGETS{'pede_mem_entry'} = $MW->Entry(
	-textvariable => \$pedemem_variable,
        -background => 'white',
	)->grid(

   -row        => $row_offset+3,
   -column     => 3,
   -sticky => 'ew',
  );


# Widget appendmillejob_label isa Label
$ZWIDGETS{'appendmillejob_label'} = $MW->Label(
   -text => 'Set up additional Mille jobs?',
  )->grid(
   -row        => $row_offset+4,
   -column     => 3,
   -sticky     => 'ew',
  );

# Widget appendmillejob_menu isa Optionmenu
$ZWIDGETS{'appendmillejob_menu'} = $MW->Optionmenu(
        -options => [[no=>2], [yes=>1]],
        -variable => \$appendmillejob_variable,
	)->grid(
   -row        => $row_offset+5,
   -column     => 3,
   -sticky     => 'ew',
  );


# Widget setup_button isa Button
$ZWIDGETS{'setup_button'} = $MW->Button(
   -text => 'Run mps_setup',
   -command => \&setup_cmd,
   -background => 'cyan',
  )->grid(
   -row        => $row_offset+9,
   -column     => 2,
   -columnspan => 2,
   -sticky     => 'ew',
  );


### THIS IS THE MPS_FIRE SECTION ###
$row_offset=12;

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

### THIS IS THE MPS_KILL SECTION ###
$row_offset=14;

### THIS IS THE MPS_RETRY SECTION ###
$row_offset=16;

# Widget retryID_label isa Label
$ZWIDGETS{'retryID_label'} = $MW->Label(
   -text => 'jobSpec to retry:',
  )->grid(
   -row        => $row_offset,
   -column     => 0,
   -sticky     => 'ew',
  );

# Widget retryID_entry isa Entry
$ZWIDGETS{'retryID_entry'} = $MW->Entry(
	-textvariable => \$retryID_variable,
        -background => 'white',
	)->grid(
   -row        => $row_offset+1,
   -column     => 0,
   -sticky => 'ew',
  );


# Widget retrymerge_label isa Label
$ZWIDGETS{'retrymerge_label'} = $MW->Label(
   -text => 'Retry merge job: ',
  )->grid(
   -row        => $row_offset,
   -column     => 1,
   -sticky     => 'ew',
  );

# Widget retrymerge_menu isa Optionmenu
$ZWIDGETS{'retrymerge_menu'} = $MW->Optionmenu(
        -options => [["no"=>""], ["yes"=>"-m"]],
        -variable => \$retrymerge_variable,
	)->grid(
   -row        => $row_offset+1,
   -column     => 1,
   -sticky     => 'ew',
  );

# Widget retryforce_label isa Label
$ZWIDGETS{'retryforce_label'} = $MW->Label(
   -text => 'Force retry on OK jobs: ',
  )->grid(
   -row        => $row_offset,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget retryforce_menu isa Optionmenu
$ZWIDGETS{'retryforce_menu'} = $MW->Optionmenu(
        -options => [["no"=>""], ["yes"=>"-f"]],
        -variable => \$retryforce_variable,
	)->grid(
   -row        => $row_offset+1,
   -column     => 2,
   -sticky     => 'ew',
  );

# Widget runmpsretry_button isa Button
$ZWIDGETS{'runmpsretry_button'} = $MW->Button(
   -text => 'Run mps_retry',
   -command => \&mpsretry_cmd,
   -background => 'cyan',
  )->grid(
   -row        => $row_offset+1,
   -column     => 3,
   -columnspan => 1,
   -sticky     => 'ew',
  );


### THIS IS THE MPS_FETCH/MPS_STAT SECTION ###
$row_offset=18;

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

### THIS IS THE MPS_SAVE SECTION ###
$row_offset=19;

# Widget savedir_label isa Label
$ZWIDGETS{'savedir_label'} = $MW->Label(
   -text => 'Save to Directory:',
  )->grid(
   -row        => $row_offset,
   -column     => 0,
   -sticky     => 'ew',
  );

# Widget pathsave_entry isa Entry
$ZWIDGETS{'pathsave_entry'} = $MW->Entry(   
	-textvariable => \$pathsave_variable,
        -background => 'white',
	)
   ->grid(
   -row    => $row_offset+1,
   -column => 0,
   -columnspan => 2,
   -sticky => 'ew',
  );

# Widget save_button isa Button
$ZWIDGETS{'save_button'} = $MW->Button(
   -text => 'Run mps_save',
   -command => \&mpssave_cmd,
   -background => 'cyan',
  )->grid(
   -row        => $row_offset+1,
   -column     => 2,
   -columnspan => 2,
   -sticky     => 'ew',
  );


### THIS IS THE OUTPUT SECTION ###
$row_offset=21;

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
$row_offset=23;

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


reset_var(); # Reset all variables
MainLoop;
printf "Prova\n";

#######################
#
# Subroutines
#
#######################

sub ZloadImages {
}

sub ZloadFonts {
}

sub open_file_v3 ($$$) {
  my $infile = shift; # Original input file name
  my $title  = shift; # Title of the browse window
  my $fext   = shift; # File extension
  my $ftyp;

  switch ($fext) {
    case "sh" {$ftyp = [ ['zsh scripts', '.sh'], ['All Files',  '*'] ];}
    case "py" {$ftyp = [ ['py files',    '.py'], ['All Files',  '*'] ];}
    case "cfg" {$ftyp = [ ['cfg files',   '.cfg'], ['All Files',  '*'] ];}
    else {$ftyp = [ ['All Files',  '*'] ];}
  }

  my $open = $MW->getOpenFile(-filetypes => $ftyp,
                              -title => $title,
			     );

  if (!defined($open)) {
    $open = $infile;
  }

  return $open;
}

sub reset_var {

  $workdir = `pwd`;
  $ZWIDGETS{'ROText2'}->delete("1.0", 'end');
  $ZWIDGETS{'ROText2'}->insert('end',$workdir);

# Read mps.db
  if (-e "mps.db") {
    read_db();
#
    $pathmillescript_variable =$batchScript;
    $pathcfg_variable         =$cfgTemplate;
    $pathdata_variable        =$infiList;
    $pathpedescript_variable  =$mergeScript;
    $pathcastor_variable      =$mssDir;
    if ($mssDirPool ne "") {
      $pathcastor_variable    =$mssDirPool.":".$pathcastor_variable;
    }
    $jobname_variable         =$addFiles;
    $njobs_variable           =$nJobs;
#
    $batchclass_variable =$class;
    $milleclass_variable ="";
    $pedeclass_variable  ="";
    $pedemem_variable = $pedeMem;
  }
#
  $setuppedejob_variable    ="1"; #The default is "yes".
  $appendmillejob_variable  ="2"; #The default is "no".
  $firemerge_variable       ="";  #The default is to merge nothing
  $njobsfir_variable        ="1"; #This is the default anyway
#
  $retryID_variable=0;
  $retrymerge_variable="";
  $retryforce_variable="";
#
  $mpssetup_output="";
  $mpsfire_output="";
  $mpskill_output="";
  $mpsretry_output="";
  $mpsfetch_output="";
  $mpsstat_output="";
  $mpssave_output="";
}

sub setup_cmd {
  my $setup_cmd = "";
  my $setup_opt = "";


#  if (-e "jobData") {
  if (-e "jobData" && $appendmillejob_variable != 1) {

    my $popup=$MW->DialogBox(-title=>"Confirm Setup",
			     -buttons=>["Confirm","Cancel"],);
    $popup->add("Label", -text=>"Are you sure? This will erase the existing jobData directory.")->pack;

    my $button = $popup->Show;

    if ($button ne "Confirm") {
      return;
    }
  }

  if ($appendmillejob_variable == 1) {
    $setup_opt .= " -a";
  }

  if ($setuppedejob_variable != 1) {
# Do not setup Pede job
    $setup_cmd = sprintf "mps_setup.pl %s %s %s %s %d %s %s",$setup_opt,
      $pathmillescript_variable,$pathcfg_variable,$pathdata_variable,
      $njobs_variable,$batchclass_variable,$jobname_variable;
    $mpssetup_output=`$setup_cmd 2>&1`;
  } else {
# Setup Mille and Pede
    $setup_opt .= " -m";
    if (length($pedemem_variable)) {
      $setup_opt .= " -M ".$pedemem_variable;
    }
    $setup_cmd = sprintf "mps_setup.pl %s %s %s %s %d %s %s %s %s",$setup_opt,
      $pathmillescript_variable,$pathcfg_variable,$pathdata_variable,
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

sub mpssave_cmd {
  my $save_cmd = "";
  
  $save_cmd = sprintf "mps_save.pl %s",$pathsave_variable;
  $mpssave_output=`$save_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$mpssave_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}

sub mpsstat_cmd {
  my $status_cmd = "";
  
  $status_cmd = sprintf "mps_stat.py";
  $mpsstat_output=`$status_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$mpsstat_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}


sub mpsfire_cmd {
  my $fire_cmd = "";
  
  $fire_cmd = sprintf "mps_fire.py %s %s",$firemerge_variable,
    $njobsfir_variable;
  $mpsfire_output=`$fire_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$mpsfire_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}

sub mpsretry_cmd {
  my $mpsretry_cmd = "";
  
  $mpsretry_cmd = sprintf "mps_retry.pl %s %s %s",$retrymerge_variable,
    $retryforce_variable,$retryID_variable;
  $mpsretry_output=`$mpsretry_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$mpsretry_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}

sub mpsfetch_cmd {

  my $fetch_cmd = "";
  
    $fetch_cmd = sprintf "mps_fetch.py";
	$mpsfetch_output=`$fetch_cmd 2>&1`;

  $ZWIDGETS{'ROText1'}->insert('end',"$mpsfetch_output \n");
  $ZWIDGETS{'ROText1'}->see('end');
}

#This should clean the text output window, right now is not implemented in any button or call
sub cleanwindow{
  $ZWIDGETS{'ROText1'}->delete("1.0", 'end');
}

#Create $batchclass_variable from $milleclass_variable and $pedeclass_variable
sub createbatchclass{
#  if (!defined($milleclass_variable)) {$milleclass_variable="";}
#  if (!defined($pedeclass_variable))  {$pedeclass_variable="";}

  if (!defined($milleclass_variable)) {return;}
  if (!defined($pedeclass_variable))  {return;}

  if ($milleclass_variable ne "" && $pedeclass_variable ne "") {
    $batchclass_variable=$milleclass_variable .":". $pedeclass_variable;
  } elsif ($milleclass_variable ne "") {
    $batchclass_variable=$milleclass_variable;
  } elsif ($pedeclass_variable ne "") {
    $batchclass_variable=$pedeclass_variable;
  }
}

#Test routine
sub prova{
  printf "Prova.\n";
}
